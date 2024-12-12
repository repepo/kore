# -*- coding: utf-8 -*-

"""Utilities to parse repodata cache."""

from __future__ import annotations

__all__ = ['RepoDataCacheRecord', 'RepoDataCache', 'REPO_CACHE']

import contextlib
import functools
import hashlib
import itertools
import os
import re
import typing

import ujson

from navigator_updater.utils import logs
from navigator_updater.utils import workers
from navigator_updater.utils import worker_utils


RepoData = typing.TypedDict(
    'RepoData',
    {
        'packages': typing.Any,
        'packages.conda': typing.Any,
        'url': str,
    },
    total=False,
)


class ValidCacheFiles(typing.Container[str]):  # pylint: disable=too-few-public-methods
    """
    Helper container used to validate repodata cache file names.

    It only emulates the collection so it might be used instead of lists or sets of expected names.
    """

    __slots__ = ()

    PATTERN: typing.Final[typing.Pattern[str]] = re.compile(r'^[a-z0-9]{8}\.json$')

    def __contains__(self, item: typing.Any) -> bool:
        """Check if file is a repodata cache file by its name."""
        return isinstance(item, str) and bool(self.PATTERN.fullmatch(item))


class RepoDataCacheRecord(typing.NamedTuple):
    """Single stored value in :class:`~RepoDataCache`."""

    content: 'RepoData'
    modification_time: float

    def merge(self, other: 'RepoDataCacheRecord') -> 'RepoDataCacheRecord':
        """Combine values of two cache records into a single one."""
        return RepoDataCacheRecord(
            content=typing.cast('RepoData', {**self.content, **other.content}),
            modification_time=max(self.modification_time, other.modification_time),
        )


class RepoDataCache:  # pylint: disable=too-few-public-methods
    """Helper class which caches details from repodata files."""

    __slots__ = ('__content',)

    def __init__(self) -> None:
        """Initialize new :class:`~RepoDataCache` instance."""
        self.__content: typing.Final[typing.Dict[str, RepoDataCacheRecord]] = {}

    @staticmethod
    def name(url: str) -> str:
        """Get repodata cache file name from the URL of the corresponding channel."""
        if not url.endswith('/'):
            url += '/'
        return hashlib.md5(url.encode('utf-8')).hexdigest()[:8] + '.json'  # nosec

    @staticmethod
    def _normalize(content: typing.Mapping[str, typing.Any]) -> 'RepoData':
        """
        Clean and normalize parsed repodata cache file content.

        This method adapts content of the repodata cache files generated by different versions of conda to a single
        unified form.
        """
        result: 'RepoData' = {}

        # normalize the `_url` (old style) and `url` (new style) values, so they might be safely overwritten without
        # leaving any "ghost" data in collected output
        with contextlib.suppress(KeyError):
            result['url'] = content['_url']
        with contextlib.suppress(KeyError):
            result['url'] = content['url']

        with contextlib.suppress(KeyError):
            result['packages'] = content['packages']
        with contextlib.suppress(KeyError):
            result['packages.conda'] = content['packages.conda']

        return result

    @workers.Task
    def file(self, path: str, *, force: bool = False) -> RepoDataCacheRecord:
        """Read content of the single repodata cache file."""
        stream: typing.TextIO
        modification_time: float = os.path.getmtime(path)
        result: typing.Optional[RepoDataCacheRecord] = self.__content.get(path, None)
        if (result is None) or (result.modification_time < modification_time) or force:
            with open(path, 'rt', encoding='utf-8') as stream:
                result = self.__content[path] = RepoDataCacheRecord(
                    content=self._normalize(ujson.load(stream)),
                    modification_time=modification_time,
                )
        return result

    @workers.Task
    def files(self, *path: str, force: bool = False) -> RepoDataCacheRecord:
        """
        Combine content of the multiple repodata cache files.

        Each next file in the request will overwrite the collected content. I.e. the further in the argument list is,
        the higher priority of its content is. This is useful when you have multiple files of the metadata, and only one
        file with the actual content, especially since the latter one may contain metadata as well with older conda
        releases.
        """
        return worker_utils.parallel_reduce(
            RepoDataCacheRecord.merge,
            (
                self.file.worker(path=item, force=force)  # pylint: disable=no-member
                for item in path
            ),
            RepoDataCacheRecord(content={}, modification_time=0.0),
        )

    def _find_repodata(
            self,
            directories: typing.Union[str, typing.Iterable[str]],
            channels: typing.Union[None, str, typing.Iterable[str]] = None,
    ) -> typing.List[typing.List[str]]:
        """
        Detect files that contain repodata cache content.

        Method returns list of groups of files. Files in each group are sorted in order expected by
        :meth:`~RepoDataCache.files`.
        """
        result: typing.List[typing.List[str]] = []

        # prepare a set of expected repodata cache file names
        channel_filter: typing.Container[str]
        if isinstance(channels, str):
            channels = [channels]
        if channels:
            channel_filter = set(map(self.name, channels))
        else:
            channel_filter = ValidCacheFiles()

        # look for core repodata cache files in primary directories
        if isinstance(directories, str):
            directories = [directories]
        directory: str
        for directory in directories:
            directory = os.path.abspath(directory)
            if not os.path.isdir(directory):
                continue
            result.extend(
                [os.path.join(directory, name)]
                for name in os.listdir(directory)
                if name in channel_filter
            )

        # extend with extra files where necessary
        # core file should remain in the end to have the highest priority and overwrite all previous content
        #
        # code below expects each core file to have a name in form of `<hex:8>.json`
        item: typing.List[str]
        for item in result:
            base: str = item[-1]

            # *.state.json files for conda 23.1.0+
            option: str = base[:-5] + '.state.json'
            if os.path.isfile(option):
                item.insert(-1, option)

            # *.info.json files for conda 23.5.0+
            option = base[:-5] + '.info.json'
            if os.path.isfile(option):
                item.insert(-1, option)

        # finish
        return result

    @workers.Task
    def collect(
            self,
            directories: typing.Union[str, typing.Iterable[str]],
            channels: typing.Union[None, str, typing.Iterable[str]] = None,
            *,
            force: bool = False,
    ) -> typing.Mapping[str, 'RepoData']:
        """Collect repodata cache from :code:`directories`, that is applicable to the :code:`channels`."""
        result: typing.Dict[str, 'RepoData'] = {}
        modification_time: typing.Dict[str, float] = {}

        thread: workers.TaskThread
        threads: typing.List[workers.TaskThread] = [
            self.files.worker(*args, force=force).thread()  # pylint: disable=no-member
            for args in self._find_repodata(directories=directories, channels=channels)
        ]
        for thread in threads:
            thread.wait()

            try:
                record: RepoDataCacheRecord = typing.cast(workers.TaskResult, thread.result).result
            except workers.TaskCanceledError:
                continue
            except Exception:  # pylint: disable=broad-except
                logs.conda_logger.exception('unable to load repodata cache file: %r', thread.call.args)
                continue

            if not (record.content.get('packages', None) or record.content.get('packages.conda', None)):
                continue

            url: typing.Optional[str] = record.content.get('url', None)
            if url is None:
                logs.conda_logger.warning('unable to parse repodata cache file: %r', thread.call.args)
                continue
            if record.modification_time > modification_time.get(url, 0.0):
                result[url], modification_time[url] = record

        return result

    @workers.Task
    def modification_time(
            self,
            directories: typing.Union[str, typing.Iterable[str]],
            channels: typing.Union[None, str, typing.Iterable[str]] = None,
    ) -> float:
        """
        Detect when repodata cache was updates last time.

        Interface is similar to the :meth:`~RepoDataCache.collect`, but returns modification time instead of the
        repodata content.
        """
        return functools.reduce(
            max,
            map(
                os.path.getmtime,
                itertools.chain.from_iterable(
                    self._find_repodata(directories=directories, channels=channels),
                )
            ),
            0.0,
        )


REPO_CACHE: typing.Final[RepoDataCache] = RepoDataCache()