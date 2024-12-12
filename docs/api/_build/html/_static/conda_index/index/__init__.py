# Copyright (C) 2018-2022 Anaconda, Inc

from __future__ import annotations

import bz2
import copy
import functools
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from numbers import Number
from os.path import abspath, basename, getmtime, getsize, isfile, join
from typing import NamedTuple
from uuid import uuid4

import zstandard
from conda.base.context import context

#  BAD BAD BAD - conda internals
from conda.core.subdir_data import SubdirData
from conda.exports import MatchSpec, Resolve, VersionOrder, human_bytes
from conda.models.channel import Channel
from conda_package_streaming import package_streaming
from jinja2 import Environment, PackageLoader

from .. import utils
from ..utils import (
    CONDA_PACKAGE_EXTENSION_V1,
    CONDA_PACKAGE_EXTENSION_V2,
    CONDA_PACKAGE_EXTENSIONS,
)
from . import rss, sqlitecache

log = logging.getLogger(__name__)

# zstd -T0 -b15 -e17 repodata.json
# level 16 gives a nice jump in ratio and decompress speed
# 15#repodata.json     : 229527083 ->  27834591 (x8.246),  112.3 MB/s, 2726.7 MB/s
# 16#repodata.json     : 229527083 ->  24457586 (x9.385),   47.6 MB/s, 3797.3 MB/s
# 17#repodata.json     : 229527083 ->  23358438 (x9.826),   30.2 MB/s, 3977.2 MB/s
ZSTD_COMPRESS_LEVEL = 16
ZSTD_COMPRESS_THREADS = -1  # automatic


def logging_config():
    """Called by package extraction subprocesses to re-configure logging."""
    import conda_index.index.logutil

    conda_index.index.logutil.configure()


# use this for debugging, because ProcessPoolExecutor isn't pdb/ipdb friendly
class DummyExecutor(Executor):
    def map(self, func, *iterables):
        for iterable in iterables:
            for thing in iterable:
                yield func(thing)

    def submit(self, func, *args, **kwargs):
        class future:
            def result(self):
                return func(*args, **kwargs)

        return future()


local_index_timestamp = 0
cached_index = None
local_subdir = ""
local_output_folder = ""
cached_channels = []
channel_data = {}

# os.cpu_count() "Return the number of CPUs in the system. Returns None if
# undetermined."
MAX_THREADS_DEFAULT = os.cpu_count() or 1
if (
    sys.platform == "win32"
):  # see https://github.com/python/cpython/commit/8ea0fd85bc67438f679491fae29dfe0a3961900a
    MAX_THREADS_DEFAULT = min(48, MAX_THREADS_DEFAULT)  # pragma: no cover
LOCK_TIMEOUT_SECS = 3 * 3600
LOCKFILE_NAME = ".lock"

# XXX conda-build calls its version of get_build_index. Appears to combine
# remote and local packages, updating the local index based on mtime. Standalone
# conda-index removes get_build_index() for now.


def _ensure_valid_channel(local_folder, subdir):
    for folder in {subdir, "noarch"}:
        path = os.path.join(local_folder, folder)
        if not os.path.isdir(path):
            os.makedirs(path)


class FileInfo(NamedTuple):
    """
    Filename and a bit of stat information.
    """

    fn: str
    st_mtime: Number
    st_size: Number


def update_index(
    dir_path,
    output_dir=None,
    check_md5=False,
    channel_name=None,
    patch_generator=None,
    threads: (int | None) = MAX_THREADS_DEFAULT,
    verbose=False,
    progress=False,
    subdirs=None,
    warn=True,
    current_index_versions=None,
    debug=False,
    write_bz2=True,
    write_zst=False,
    write_run_exports=False,
):
    """
    High-level interface to ``ChannelIndex``. Index all subdirs under
    ``dir_path``. Output to `output_dir`, or under the input directory if
    `output_dir` is not given. Writes updated ``channeldata.json``.

    The input ``dir_path`` should at least contain a directory named ``noarch``.
    The path tree therein is treated as a full channel, with a level of subdirs,
    each subdir having an update to repodata.json. The full channel will also
    have a channeldata.json file.
    """
    _, dirname = os.path.split(dir_path)
    if dirname in utils.DEFAULT_SUBDIRS:
        if warn:
            log.warn(
                "The update_index function has changed to index all subdirs at once.  You're pointing it at a single subdir.  "
                "Please update your code to point it at the channel root, rather than a subdir. "
                "Use -s=<subdir> to update a single subdir."
            )
        raise ValueError(
            "Does not accept a single subdir, or a path named "
            "like one of the standard subdirs."
        )

    channel_index = ChannelIndex(
        dir_path,
        channel_name,
        subdirs=subdirs,
        threads=threads,
        deep_integrity_check=check_md5,
        debug=debug,
        output_root=output_dir,
        write_bz2=write_bz2,
        write_zst=write_zst,
        write_run_exports=write_run_exports,
    )

    channel_index.index(
        patch_generator=patch_generator,
        verbose=verbose,
        progress=progress,
        current_index_versions=current_index_versions,
    )

    channel_index.update_channeldata()


def _make_seconds(timestamp):
    timestamp = int(timestamp)
    if timestamp > 253402300799:  # 9999-12-31
        timestamp //= (
            1000  # convert milliseconds to seconds; see conda/conda-build#1988
        )
    return timestamp


# ==========================================================================


REPODATA_VERSION = 1
CHANNELDATA_VERSION = 1
RUN_EXPORTS_VERSION = 1
REPODATA_JSON_FN = "repodata.json"
REPODATA_FROM_PKGS_JSON_FN = "repodata_from_packages.json"
RUN_EXPORTS_JSON_FN = "run_exports.json"
CHANNELDATA_FIELDS = (
    "description",
    "dev_url",
    "doc_url",
    "doc_source_url",
    "home",
    "license",
    "reference_package",
    "source_url",
    "source_git_url",
    "source_git_tag",
    "source_git_rev",
    "summary",
    "version",
    "subdirs",
    "icon_url",
    "icon_hash",  # "md5:abc123:12"
    "run_exports",
    "binary_prefix",
    "text_prefix",
    "activate.d",
    "deactivate.d",
    "pre_link",
    "post_link",
    "pre_unlink",
    "tags",
    "identifiers",
    "keywords",
    "recipe_origin",
    "commits",
)


def _apply_instructions(subdir, repodata, instructions):
    repodata.setdefault("removed", [])
    utils.merge_or_update_dict(
        repodata.get("packages", {}),
        instructions.get("packages", {}),
        merge=False,
        add_missing_keys=False,
    )
    # we could have totally separate instructions for .conda than .tar.bz2, but it's easier if we assume
    #    that a similarly-named .tar.bz2 file is the same content as .conda, and shares fixes
    new_pkg_fixes = {
        k.replace(CONDA_PACKAGE_EXTENSION_V1, CONDA_PACKAGE_EXTENSION_V2): v
        for k, v in instructions.get("packages", {}).items()
    }

    utils.merge_or_update_dict(
        repodata.get("packages.conda", {}),
        new_pkg_fixes,
        merge=False,
        add_missing_keys=False,
    )
    utils.merge_or_update_dict(
        repodata.get("packages.conda", {}),
        instructions.get("packages.conda", {}),
        merge=False,
        add_missing_keys=False,
    )

    for fn in instructions.get("revoke", ()):
        for key in ("packages", "packages.conda"):
            if fn.endswith(CONDA_PACKAGE_EXTENSION_V1) and key == "packages.conda":
                fn = fn.replace(CONDA_PACKAGE_EXTENSION_V1, CONDA_PACKAGE_EXTENSION_V2)
            if fn in repodata[key]:
                repodata[key][fn]["revoked"] = True
                repodata[key][fn]["depends"].append("package_has_been_revoked")

    for fn in instructions.get("remove", ()):
        for key in ("packages", "packages.conda"):
            if fn.endswith(CONDA_PACKAGE_EXTENSION_V1) and key == "packages.conda":
                fn = fn.replace(CONDA_PACKAGE_EXTENSION_V1, CONDA_PACKAGE_EXTENSION_V2)
            popped = repodata[key].pop(fn, None)
            if popped:
                repodata["removed"].append(fn)
    repodata["removed"].sort()

    return repodata


def _get_jinja2_environment():
    def _filter_strftime(dt, dt_format):
        if isinstance(dt, (int, float)):
            if dt > 253402300799:  # 9999-12-31
                dt //= 1000  # convert milliseconds to seconds; see #1988
            dt = datetime.fromtimestamp(dt, tz=timezone.utc)
        return dt.strftime(dt_format)

    def _filter_add_href(text, link, **kwargs):
        if link:
            kwargs_list = [f'href="{link}"']
            kwargs_list += [f'{k}="{v}"' for k, v in kwargs.items()]
            return "<a {}>{}</a>".format(" ".join(kwargs_list), text)
        else:
            return text

    environment = Environment(
        loader=PackageLoader("conda_index", "templates"),
    )
    environment.filters["human_bytes"] = human_bytes
    environment.filters["strftime"] = _filter_strftime
    environment.filters["add_href"] = _filter_add_href
    environment.trim_blocks = True
    environment.lstrip_blocks = True

    return environment


def _make_subdir_index_html(channel_name, subdir, repodata_packages, extra_paths):
    environment = _get_jinja2_environment()
    template = environment.get_template("subdir-index.html.j2")
    rendered_html = template.render(
        title="{}/{}".format(channel_name or "", subdir),
        packages=repodata_packages,
        current_time=datetime.now(timezone.utc),
        extra_paths=extra_paths,
    )
    return rendered_html


def _make_channeldata_index_html(channel_name, channeldata):
    environment = _get_jinja2_environment()
    template = environment.get_template("channeldata-index.html.j2")
    rendered_html = template.render(
        title=channel_name,
        packages=channeldata["packages"],
        subdirs=channeldata["subdirs"],
        current_time=datetime.now(timezone.utc),
    )
    return rendered_html


def _make_rss(channel_name, channeldata):
    return rss.get_rss(channel_name, channeldata)


def _get_resolve_object(subdir, precs=None, repodata=None):
    packages = {}
    conda_packages = {}
    if not repodata:
        repodata = {
            "info": {
                "subdir": subdir,
                "arch": context.arch_name,
                "platform": context.platform,
            },
            "packages": packages,
            "packages.conda": conda_packages,
        }

    channel = Channel(f"https://conda.anaconda.org/dummy-channel/{subdir}")
    sd = SubdirData(channel)

    # repodata = copy.deepcopy(repodata) # slower than json.dumps/load loop
    repodata_copy = json.loads(json.dumps(repodata))

    # adds url, Channel objects to each repodata package
    sd._process_raw_repodata(repodata_copy)

    sd._loaded = True
    SubdirData._cache_[channel.url(with_credentials=True)] = sd

    index = {prec: prec for prec in precs or sd._package_records}
    r = Resolve(index, channels=(channel,))
    return r


def _add_missing_deps(new_r, original_r):
    """For each package in new_r, if any deps are not satisfiable, backfill them from original_r."""

    expanded_groups = copy.deepcopy(new_r.groups)
    seen_specs = set()
    for g_name, g_recs in new_r.groups.items():
        for g_rec in g_recs:
            for dep_spec in g_rec.depends:
                if dep_spec in seen_specs:
                    continue
                ms = MatchSpec(dep_spec)
                if not new_r.find_matches(ms):
                    matches = original_r.find_matches(ms)
                    if matches:
                        version = matches[0].version
                        expanded_groups[ms.name] = set(
                            expanded_groups.get(ms.name, [])
                        ) | set(
                            original_r.find_matches(MatchSpec(f"{ms.name}={version}"))
                        )
                seen_specs.add(dep_spec)
    return [pkg for group in expanded_groups.values() for pkg in group]


def _add_prev_ver_for_features(new_r, orig_r):
    expanded_groups = copy.deepcopy(new_r.groups)
    for g_name in new_r.groups:
        if not any(m.track_features or m.features for m in new_r.groups[g_name]):
            # no features so skip
            continue

        # versions are sorted here so this is the latest
        latest_version = VersionOrder(str(new_r.groups[g_name][0].version))
        if g_name in orig_r.groups:
            # now we iterate through the list to find the next to latest
            # without a feature
            keep_m = None
            for i in range(len(orig_r.groups[g_name])):
                _m = orig_r.groups[g_name][i]
                if VersionOrder(str(_m.version)) <= latest_version and not (
                    _m.track_features or _m.features
                ):
                    keep_m = _m
                    break
            if keep_m is not None:
                expanded_groups[g_name] = {keep_m} | set(
                    expanded_groups.get(g_name, [])
                )

    return [pkg for group in expanded_groups.values() for pkg in group]


def _shard_newest_packages(subdir, r, pins=None):
    """Captures only the newest versions of software in the resolve object.

    For things where more than one version is supported simultaneously (like Python),
    pass pins as a dictionary, with the key being the package name, and the value being
    a list of supported versions.  For example:

    {'python': ["2.7", "3.6"]}
    """
    groups = {}
    pins = pins or {}
    for g_name, g_recs in r.groups.items():
        # always do the latest implicitly
        version = r.groups[g_name][0].version
        matches = set(r.find_matches(MatchSpec(f"{g_name}={version}")))
        if g_name in pins:
            for pin_value in pins[g_name]:
                version = r.find_matches(MatchSpec(f"{g_name}={pin_value}"))[0].version
                matches.update(r.find_matches(MatchSpec(f"{g_name}={version}")))
        groups[g_name] = matches

    # add the deps of the stuff in the index
    new_r = _get_resolve_object(
        subdir, precs=[pkg for group in groups.values() for pkg in group]
    )
    new_r = _get_resolve_object(subdir, precs=_add_missing_deps(new_r, r))

    # now for any pkg with features, add at least one previous version
    # also return
    return set(_add_prev_ver_for_features(new_r, r))


def _build_current_repodata(subdir, repodata, pins):
    r = _get_resolve_object(subdir, repodata=repodata)
    keep_pkgs = _shard_newest_packages(subdir, r, pins)
    new_repodata = {
        k: repodata[k] for k in set(repodata.keys()) - {"packages", "packages.conda"}
    }
    packages = {}
    conda_packages = {}
    for keep_pkg in keep_pkgs:
        if keep_pkg.fn.endswith(CONDA_PACKAGE_EXTENSION_V2):
            conda_packages[keep_pkg.fn] = repodata["packages.conda"][keep_pkg.fn]
            # in order to prevent package churn we consider the md5 for the .tar.bz2 that matches the .conda file
            #    This holds when .conda files contain the same files as .tar.bz2, which is an assumption we'll make
            #    until it becomes more prevalent that people provide only .conda files and just skip .tar.bz2
            counterpart = keep_pkg.fn.replace(
                CONDA_PACKAGE_EXTENSION_V2, CONDA_PACKAGE_EXTENSION_V1
            )
            conda_packages[keep_pkg.fn]["legacy_bz2_md5"] = (
                repodata["packages"].get(counterpart, {}).get("md5")
            )
        elif keep_pkg.fn.endswith(CONDA_PACKAGE_EXTENSION_V1):
            packages[keep_pkg.fn] = repodata["packages"][keep_pkg.fn]
    new_repodata["packages"] = packages
    new_repodata["packages.conda"] = conda_packages
    return new_repodata


def thread_executor_factory(debug, threads):
    return (
        DummyExecutor()
        if (debug or threads == 1)
        else ProcessPoolExecutor(
            threads,
            initializer=logging_config,
            mp_context=multiprocessing.get_context("spawn"),
        )
    )  # "fork" start method may cause hangs even on Linux?


class ChannelIndex:
    """
    Class implementing ``update_index``. Allows for more fine-grained control of
    output.

    See the implementation of ``conda_index.cli`` for usage.
    """

    def __init__(
        self,
        channel_root,
        channel_name,
        subdirs=None,
        threads: (int | None) = MAX_THREADS_DEFAULT,
        deep_integrity_check=False,
        debug=False,
        output_root=None,  # write repodata.json etc. to separate folder?
        cache_class=sqlitecache.CondaIndexCache,
        write_bz2=False,
        write_zst=False,
        write_run_exports=False,
        compact_json=True,
    ):
        if threads is None:
            threads = MAX_THREADS_DEFAULT

        self.cache_class = cache_class
        self.channel_root = abspath(channel_root)
        self.output_root = abspath(output_root) if output_root else self.channel_root
        self.channel_name = channel_name or basename(channel_root.rstrip("/"))
        self._subdirs = subdirs
        # no lambdas in pickleable
        self.thread_executor_factory = functools.partial(
            thread_executor_factory, debug, threads
        )
        self.debug = debug
        self.deep_integrity_check = deep_integrity_check
        self.write_bz2 = write_bz2
        self.write_zst = write_zst
        self.write_run_exports = write_run_exports
        self.compact_json = compact_json

    def index(
        self,
        patch_generator,
        verbose=False,
        progress=False,
        current_index_versions=None,
    ):
        """
        Examine all changed packages under ``self.channel_root``, updating
        ``index.html`` for each subdir.
        """
        if verbose:
            log.debug(
                "ChannelIndex.index(verbose=...) is a no-op. Alter log levels for %s to control verbosity.",
                __name__,
            )

        subdirs = self.detect_subdirs()

        # Lock local channel.
        with utils.try_acquire_locks([utils.get_lock(self.channel_root)], timeout=900):
            # begin non-stop "extract packages into cache";
            # extract_subdir_to_cache manages subprocesses. Keeps cores busy
            # during write/patch/update channeldata steps.
            def extract_subdirs_to_cache():
                executor = ThreadPoolExecutor(max_workers=1)

                def extract_args():
                    for subdir in subdirs:
                        # .cache is currently in channel_root not output_root
                        _ensure_valid_channel(self.channel_root, subdir)
                        subdir_path = join(self.channel_root, subdir)
                        yield (subdir, verbose, progress, subdir_path)

                def extract_wrapper(args: tuple):
                    # runs in thread
                    subdir, verbose, progress, subdir_path = args
                    cache = self.cache_for_subdir(subdir)
                    return self.extract_subdir_to_cache(
                        subdir, verbose, progress, subdir_path, cache
                    )

                # map() gives results in order passed, not in order of
                # completion. If using multiple threads, switch to
                # submit() / as_completed().
                return executor.map(extract_wrapper, extract_args())

            # Collect repodata from packages, save to
            # REPODATA_FROM_PKGS_JSON_FN file
            with self.thread_executor_factory() as index_process:
                futures = [
                    index_process.submit(
                        functools.partial(
                            self.index_prepared_subdir,
                            subdir=subdir,
                            verbose=verbose,
                            progress=progress,
                            patch_generator=patch_generator,
                            current_index_versions=current_index_versions,
                        )
                    )
                    for subdir in extract_subdirs_to_cache()
                ]
                # limited API to support DummyExecutor
                for future in futures:
                    result = future.result()
                    log.info(f"Completed {result}")

    def index_prepared_subdir(
        self,
        subdir: str,
        verbose: bool,
        progress: bool,
        patch_generator,
        current_index_versions,
    ):
        """
        Create repodata_from_packages.json by calling index_subdir, then apply
        any patches to create repodata.json.
        """
        log.info("Subdir: %s Gathering repodata", subdir)

        repodata_from_packages = self.index_subdir(
            subdir, verbose=verbose, progress=progress
        )

        log.info("%s Writing pre-patch repodata", subdir)
        self._write_repodata(
            subdir,
            repodata_from_packages,
            REPODATA_FROM_PKGS_JSON_FN,
        )

        # Apply patch instructions.
        log.info("%s Applying patch instructions", subdir)
        patched_repodata, _ = self._patch_repodata(
            subdir, repodata_from_packages, patch_generator
        )

        # Save patched and augmented repodata. If the contents
        # of repodata have changed, write a new repodata.json.
        # Create associated index.html.

        log.info("%s Writing patched repodata", subdir)

        log.debug("%s write patched repodata", subdir)
        self._write_repodata(subdir, patched_repodata, REPODATA_JSON_FN)

        log.info("%s Building current_repodata subset", subdir)

        log.debug("%s build current_repodata", subdir)
        current_repodata = _build_current_repodata(
            subdir, patched_repodata, pins=current_index_versions
        )

        log.info("%s Writing current_repodata subset", subdir)

        log.debug("%s write current_repodata", subdir)
        self._write_repodata(
            subdir,
            current_repodata,
            json_filename="current_repodata.json",
        )

        if self.write_run_exports:
            log.info("%s Building run_exports data", subdir)
            run_exports_data = self.build_run_exports_data(subdir)

            log.info("%s Writing run_exports.json", subdir)
            self._write_repodata(
                subdir,
                run_exports_data,
                json_filename=RUN_EXPORTS_JSON_FN,
            )

        log.info("%s Writing index HTML", subdir)

        log.debug("%s write index.html", subdir)
        self._write_subdir_index_html(subdir, patched_repodata)

        log.debug("%s finish", subdir)

        return subdir

    def update_channeldata(self, rss=False):
        """
        Update channeldata based on re-reading output `repodata.json` and existing
        `channeldata.json`. Call after index() if channeldata is needed.
        """
        subdirs = self.detect_subdirs()

        # Skip locking; only writes the channeldata.

        # Keep channeldata in memory, update with each subdir.
        channel_data = {}
        channeldata_file = self.channeldata_path()
        if os.path.isfile(channeldata_file):
            with open(channeldata_file) as f:
                channel_data = json.load(f)

        for subdir in subdirs:
            log.info("Channeldata subdir: %s", subdir)
            log.debug("%s read repodata", subdir)
            with open(
                os.path.join(self.output_root, subdir, REPODATA_JSON_FN)
            ) as repodata:
                patched_repodata = json.load(repodata)

            self._update_channeldata(channel_data, patched_repodata, subdir)

            log.debug("%s channeldata finished", subdir)

        # Create and write the rss feed.
        if rss:
            self._write_rss(channel_data)

        # Create and write channeldata.
        self._write_channeldata_index_html(channel_data)
        log.debug("write channeldata")
        self._write_channeldata(channel_data)

    def detect_subdirs(self):
        if not self._subdirs:
            detected_subdirs = {
                subdir.name
                for subdir in os.scandir(self.channel_root)
                if subdir.name in utils.DEFAULT_SUBDIRS and subdir.is_dir()
            }
            log.debug("found subdirs %s", detected_subdirs)
            self.subdirs = sorted(detected_subdirs | {"noarch"})
        else:
            self.subdirs = sorted(set(self._subdirs))
            if "noarch" not in self.subdirs:
                log.warn("Indexing %s does not include 'noarch'", self.subdirs)
        return self.subdirs

    def channeldata_path(self):
        channeldata_file = os.path.join(self.output_root, "channeldata.json")
        return channeldata_file

    def index_subdir(self, subdir, verbose=False, progress=False):
        """
        Return repodata from the cache without reading old repodata.json

        Must call `extract_subdir_to_cache()` first or will be outdated.
        """
        subdir_path = join(self.channel_root, subdir)

        cache = self.cache_for_subdir(subdir)

        log.debug("Building repodata for %s", subdir_path)

        new_repodata_packages, new_repodata_conda_packages = cache.indexed_packages()

        new_repodata = {
            "packages": new_repodata_packages,
            "packages.conda": new_repodata_conda_packages,
            "info": {
                "subdir": subdir,
            },
            "repodata_version": REPODATA_VERSION,
            "removed": [],  # can be added by patch/hotfix process
        }

        return new_repodata

    def cache_for_subdir(self, subdir):
        cache: sqlitecache.CondaIndexCache = self.cache_class(
            channel_root=self.channel_root, subdir=subdir
        )
        if cache.cache_is_brand_new:
            # guaranteed to be only thread doing this?
            cache.convert()
        return cache

    def extract_subdir_to_cache(
        self, subdir, verbose, progress, subdir_path, cache: sqlitecache.CondaIndexCache
    ):
        """
        Extract all changed packages into the subdir cache.

        Return name of subdir.
        """
        # exactly these packages (unless they are un-indexable) will be in the
        # output repodata
        cache.save_fs_state(subdir_path)

        log.debug("%s find packages to extract", subdir)

        # list so tqdm can show progress
        extract = [
            FileInfo(
                fn=cache.plain_path(row["path"]),
                st_mtime=row["mtime"],
                st_size=row["size"],
            )
            for row in cache.changed_packages()
        ]

        log.debug("%s extract %d packages", subdir, len(extract))

        # now updates own stat cache
        extract_func = functools.partial(
            cache.extract_to_cache_info_object, self.channel_root, subdir
        )

        start_time = time.time()
        size_processed = 0

        with self.thread_executor_factory() as thread_executor:
            for fn, mtime, size, index_json in thread_executor.map(
                extract_func, extract
            ):
                size_processed += size  # even if processed incorrectly
                # fn can be None if the file was corrupt or no longer there
                if fn and mtime:
                    if index_json:
                        pass  # correctly indexed a package! index_subdir will fetch.
                    else:
                        log.error(
                            "Package at %s did not contain valid index.json data.  Please"
                            " check the file and remove/redownload if necessary to obtain "
                            "a valid package.",
                            os.path.join(subdir_path, fn),
                        )
            end_time = time.time()
            try:
                bytes_sec = size_processed / (end_time - start_time)
            except ZeroDivisionError:  # pragma: no cover
                bytes_sec = 0
        log.info(
            "%s cached %s from %s packages at %s/second",
            subdir,
            human_bytes(size_processed),
            len(extract),
            human_bytes(bytes_sec),
        )

        return subdir

    def _write_repodata(self, subdir, repodata, json_filename):
        """
        Write repodata to :json_filename, but only if changed.
        """
        repodata_json_path = join(self.channel_root, subdir, json_filename)
        new_repodata = self.json_dumps(repodata)
        write_result = self._maybe_write(
            repodata_json_path, new_repodata, write_newline_end=False
        )
        # write repodata.json.bz2 if it doesn't exist, even if repodata.json has
        # not changed
        repodata_bz2_path = repodata_json_path + ".bz2"
        repodata_zst_path = repodata_json_path + ".zst"
        if write_result or not os.path.exists(repodata_bz2_path):
            if self.write_bz2:
                bz2_content = bz2.compress(new_repodata.encode("utf-8"))
                self._maybe_write(repodata_bz2_path, bz2_content)
            else:
                self._maybe_remove(repodata_bz2_path)
            if self.write_zst:
                repodata_zst_content = zstandard.ZstdCompressor(
                    level=ZSTD_COMPRESS_LEVEL, threads=ZSTD_COMPRESS_THREADS
                ).compress(new_repodata.encode("utf-8"))
                self._maybe_write(repodata_zst_path, repodata_zst_content)
            else:
                self._maybe_remove(repodata_zst_path)
        return write_result

    def _write_subdir_index_html(self, subdir, repodata):
        repodata_legacy_packages = repodata["packages"]
        repodata_conda_packages = repodata["packages.conda"]

        repodata_packages = {}
        repodata_packages.update(repodata_legacy_packages)
        repodata_packages.update(repodata_conda_packages)

        subdir_path = join(self.channel_root, subdir)

        def _add_extra_path(extra_paths, path):
            if isfile(join(self.channel_root, path)):
                md5sum, sha256sum = utils.checksums(path, ("md5", "sha256"))
                extra_paths[basename(path)] = {
                    "size": getsize(path),
                    "timestamp": int(getmtime(path)),
                    "sha256": sha256sum,
                    "md5": md5sum,
                }

        extra_paths = {}
        _add_extra_path(extra_paths, join(subdir_path, REPODATA_JSON_FN))
        if self.write_bz2:
            _add_extra_path(extra_paths, join(subdir_path, REPODATA_JSON_FN + ".bz2"))
        _add_extra_path(extra_paths, join(subdir_path, REPODATA_FROM_PKGS_JSON_FN))
        if self.write_bz2:
            _add_extra_path(
                extra_paths, join(subdir_path, REPODATA_FROM_PKGS_JSON_FN + ".bz2")
            )

        _add_extra_path(extra_paths, join(subdir_path, "patch_instructions.json"))
        rendered_html = _make_subdir_index_html(
            self.channel_name, subdir, repodata_packages, extra_paths
        )
        assert rendered_html
        index_path = join(subdir_path, "index.html")
        return self._maybe_write(index_path, rendered_html)

    def _write_rss(self, channeldata):
        log.info("Build RSS")
        rss = _make_rss(self.channel_name, channeldata)
        rss_path = join(self.channel_root, "rss.xml")
        self._maybe_write(rss_path, rss)
        log.info("Built RSS")

    def _write_channeldata_index_html(self, channeldata):
        rendered_html = _make_channeldata_index_html(self.channel_name, channeldata)
        assert rendered_html
        index_path = join(self.channel_root, "index.html")
        self._maybe_write(index_path, rendered_html)

    def _update_channeldata(self, channel_data, repodata, subdir):

        cache = self.cache_for_subdir(subdir)

        legacy_packages = repodata["packages"]
        conda_packages = repodata["packages.conda"]

        use_these_legacy_keys = set(legacy_packages.keys()) - {
            k[:-6] + CONDA_PACKAGE_EXTENSION_V1 for k in conda_packages.keys()
        }
        all_repodata_packages = conda_packages.copy()
        all_repodata_packages.update(
            {k: legacy_packages[k] for k in use_these_legacy_keys}
        )
        package_data = channel_data.get("packages", {})

        # Pay special attention to groups that have run_exports - we
        # need to process each version group by version; take newest per
        # version group.  We handle groups that are not in the index at
        # all yet similarly, because we can't check if they have any
        # run_exports.

        # This is more deterministic than, but slower than the old "newest
        # timestamp across all versions if no run_exports", unsatisfying
        # when old versions get new builds. When channeldata.json is not
        # being built from scratch the speed difference is not noticable.

        def newest_by_name_and_version(all_repodata_packages):
            namever = {}

            for fn, package in all_repodata_packages.items():
                key = (package["name"], package["version"])
                timestamp = package.get("timestamp", 0)
                existing = namever.get(key)
                if not existing or existing[1].get("timestamp", 0) < timestamp:
                    namever[key] = (fn, package)

            return list(namever.values())

        groups = newest_by_name_and_version(all_repodata_packages)

        def _replace_if_newer_and_present(pd, data, existing_record, data_newer, k):
            if data.get(k) and (data_newer or not existing_record.get(k)):
                pd[k] = data[k]
            else:
                pd[k] = existing_record.get(k)

        # unzipping
        fns, fn_dicts = [], []
        if groups:
            fns, fn_dicts = zip(*groups)

        load_func = cache.load_all_from_cache
        with self.thread_executor_factory() as thread_executor:
            for fn_dict, data in zip(fn_dicts, thread_executor.map(load_func, fns)):
                # not reached when older channeldata.json matches
                if data:
                    data.update(fn_dict)
                    name = data["name"]
                    # existing record
                    existing_record = package_data.get(name, {})
                    data_v = data.get("version", "0")
                    erec_v = existing_record.get("version", "0")
                    # are timestamps already normalized to seconds?
                    data_newer = VersionOrder(data_v) > VersionOrder(erec_v) or (
                        data_v == erec_v
                        and _make_seconds(data.get("timestamp", 0))
                        > _make_seconds(existing_record.get("timestamp", 0))
                    )

                    package_data[name] = package_data.get(name, {})
                    # keep newer value for these
                    for k in (
                        "description",
                        "dev_url",
                        "doc_url",
                        "doc_source_url",
                        "home",
                        "license",
                        "source_url",
                        "source_git_url",
                        "summary",
                        "icon_url",
                        "icon_hash",
                        "tags",
                        "identifiers",
                        "keywords",
                        "recipe_origin",
                        "version",
                    ):
                        _replace_if_newer_and_present(
                            package_data[name], data, existing_record, data_newer, k
                        )

                    # keep any true value for these, since we don't distinguish subdirs
                    for k in (
                        "binary_prefix",
                        "text_prefix",
                        "activate.d",
                        "deactivate.d",
                        "pre_link",
                        "post_link",
                        "pre_unlink",
                    ):
                        package_data[name][k] = any(
                            (data.get(k), existing_record.get(k))
                        )

                    package_data[name]["subdirs"] = sorted(
                        list(set(existing_record.get("subdirs", []) + [subdir]))
                    )
                    # keep one run_exports entry per version of the package, since these vary by version
                    run_exports = existing_record.get("run_exports", {})
                    exports_from_this_version = data.get("run_exports")
                    if exports_from_this_version:
                        run_exports[data_v] = data.get("run_exports")
                    package_data[name]["run_exports"] = run_exports
                    package_data[name]["timestamp"] = _make_seconds(
                        max(
                            data.get("timestamp", 0),
                            channel_data.get(name, {}).get("timestamp", 0),
                        )
                    )

        channel_data.update(
            {
                "channeldata_version": CHANNELDATA_VERSION,
                "subdirs": sorted(
                    list(set(channel_data.get("subdirs", []) + [subdir]))
                ),
                "packages": package_data,
            }
        )

    def json_dumps(self, data):
        """
        Format json based on class policy.
        """
        if self.compact_json:
            return json.dumps(data, sort_keys=True, separators=(",", ":"))
        else:
            return json.dumps(data, sort_keys=True, indent=2) + "\n"

    def _write_channeldata(self, channeldata):
        # trim out commits, as they can take up a ton of space.  They're really only for the RSS feed.
        for pkg, pkg_dict in channeldata.get("packages", {}).items():
            channeldata["packages"][pkg] = {
                k: v for k, v in pkg_dict.items() if v is not None and k != "commits"
            }
        channeldata_path = join(self.channel_root, "channeldata.json")
        content = self.json_dumps(channeldata)
        self._maybe_write(channeldata_path, content, True)

    def build_run_exports_data(self, subdir, verbose=False, progress=False):
        """
        Return CEP-12 compliant run_exports metadata from the db cache.

        Must call `extract_subdir_to_cache()` first or will be outdated.
        """
        subdir_path = join(self.channel_root, subdir)

        cache = self.cache_for_subdir(subdir)

        log.debug("Building run_exports for %s", subdir_path)

        run_exports_packages = {}
        run_exports_conda_packages = {}

        # load cached packages
        for row in cache.db.execute(
            """
            SELECT path, run_exports FROM stat
            LEFT JOIN run_exports USING (path)
            WHERE stat.stage = ?
            ORDER BY path
            """,
            (cache.upstream_stage,),
        ):
            path, run_exports_data = row
            run_exports_data = {"run_exports": json.loads(run_exports_data or "{}")}
            if path.endswith(CONDA_PACKAGE_EXTENSION_V1):
                run_exports_packages[path] = run_exports_data
            elif path.endswith(CONDA_PACKAGE_EXTENSION_V2):
                run_exports_conda_packages[path] = run_exports_data
            else:
                log.warn("%s doesn't look like a conda package", path)

        new_run_exports_data = {
            "packages": run_exports_packages,
            "packages.conda": run_exports_conda_packages,
            "info": {
                "subdir": subdir,
                "version": RUN_EXPORTS_VERSION,
            },
        }

        return new_run_exports_data

    def _load_patch_instructions_tarball(self, subdir, patch_generator):
        instructions = {}

        target = "/".join((subdir, "patch_instructions.json"))
        for tar, member in package_streaming.stream_conda_component(
            patch_generator, component="pkg"
        ):
            if member.name == target:
                reader = tar.extractfile(member)
                assert reader, "tar member was not a regular file"
                instructions = json.load(reader)
        return instructions

    def _create_patch_instructions(self, subdir, repodata, patch_generator=None):
        gen_patch_path = patch_generator or join(self.channel_root, "gen_patch.py")
        if isfile(gen_patch_path):
            log.debug(f"using patch generator {gen_patch_path} for {subdir}")

            # https://stackoverflow.com/a/41595552/2127762
            from importlib.util import module_from_spec, spec_from_file_location

            spec = spec_from_file_location("a_b", gen_patch_path)
            if spec and spec.loader:
                mod = module_from_spec(spec)
                spec.loader.exec_module(mod)
            else:
                raise ImportError()

            instructions = mod._patch_repodata(repodata, subdir)

            if instructions.get("patch_instructions_version", 0) > 1:
                raise RuntimeError("Incompatible patch instructions version")

            return instructions
        else:
            if patch_generator:
                raise ValueError(
                    "Specified metadata patch file '{}' does not exist.  Please try an absolute "
                    "path, or examine your relative path carefully with respect to your cwd.".format(
                        patch_generator
                    )
                )
            return {}

    def _write_patch_instructions(self, subdir, instructions):
        new_patch = self.json_dumps(instructions)
        patch_instructions_path = join(
            self.channel_root, subdir, "patch_instructions.json"
        )
        self._maybe_write(patch_instructions_path, new_patch, True)

    def _load_instructions(self, subdir):
        patch_instructions_path = join(
            self.channel_root, subdir, "patch_instructions.json"
        )
        if isfile(patch_instructions_path):
            log.debug("using patch instructions %s", patch_instructions_path)
            with open(patch_instructions_path) as fh:
                instructions = json.load(fh)
                if instructions.get("patch_instructions_version", 0) > 1:
                    raise RuntimeError("Incompatible patch instructions version")
                return instructions
        return {}

    def _patch_repodata(self, subdir, repodata, patch_generator: str | None = None):
        if patch_generator and patch_generator.endswith(CONDA_PACKAGE_EXTENSIONS):
            instructions = self._load_patch_instructions_tarball(
                subdir, patch_generator
            )
        else:
            instructions = self._create_patch_instructions(
                subdir, repodata, patch_generator
            )
        if instructions:
            self._write_patch_instructions(subdir, instructions)
        else:
            instructions = self._load_instructions(subdir)
        if instructions.get("patch_instructions_version", 0) > 1:
            raise RuntimeError("Incompatible patch instructions version")

        return _apply_instructions(subdir, repodata, instructions), instructions

    def _maybe_write(self, path, content: str | bytes, write_newline_end=False):
        # Create the temp file next "path" so that we can use an atomic move, see
        # https://github.com/conda/conda-build/issues/3833
        temp_path = f"{path}.{uuid4()}"

        # intercept to support separate output_directory
        output_path = os.path.join(
            self.output_root, (os.path.relpath(path, self.channel_root))
        )

        output_temp_path = os.path.join(
            self.output_root, (os.path.relpath(temp_path, self.channel_root))
        )

        os.makedirs(os.path.dirname(output_temp_path), exist_ok=True)

        log.debug(f"_maybe_write {path} to {output_path}")

        return self._maybe_write_output_paths(
            content, output_path, output_temp_path, write_newline_end
        )

    def _maybe_write_output_paths(
        self, content: str | bytes, output_path, output_temp_path, write_newline_end
    ):
        """
        Internal to _maybe_write.
        """

        if isinstance(content, str):
            mode = "w"
            encoding = "utf-8"
            newline = "\n"
            newline_option = "\n"
        else:
            mode = "wb"
            encoding = None
            newline = b"\n"
            newline_option = None

        # always use \n line separator
        with open(
            output_temp_path,
            mode=mode,
            encoding=encoding,
            newline=newline_option,
        ) as fh:
            fh.write(content)
            if write_newline_end:
                fh.write(newline)

        if isfile(output_path):
            if utils.file_contents_match(output_temp_path, output_path):
                # No need to change mtimes. The contents already match.
                os.unlink(output_temp_path)
                return False

        utils.move_with_fallback(output_temp_path, output_path)
        return True

    def _maybe_remove(self, path):
        """
        Remove path if it exists, rewriting to respect self.output_root.
        """

        # intercept to support separate output_directory
        output_path = os.path.join(
            self.output_root, (os.path.relpath(path, self.channel_root))
        )

        log.debug(f"_maybe_remove {path} from {output_path}")

        if isfile(output_path):
            os.unlink(output_path)
