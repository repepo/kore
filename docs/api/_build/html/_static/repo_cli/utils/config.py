from __future__ import absolute_import, print_function, unicode_literals

import collections
import itertools
import logging
import os
import stat
import warnings
from os.path import abspath, dirname, exists, expanduser, isdir, isfile, join
from string import Template

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


try:
    from urllib import quote_plus
except ImportError:
    from urllib.parse import quote_plus

# from binstar_client.utils.appdirs import AppDirs, EnvAppDirs
# from binstar_client.errors import BinstarError
from ..errors import RepoCLIError
from .appdirs import AppDirs, EnvAppDirs
from .conda import CONDA_PREFIX, CONDA_ROOT
from .yaml import yaml_dump, yaml_load

logger = logging.getLogger("repo_cli")


def expandvars(path):
    environ = dict(CONDA_ROOT=CONDA_ROOT, CONDA_PREFIX=CONDA_PREFIX)
    environ.update(os.environ)
    return Template(path).safe_substitute(**environ)


def expand(path):
    return abspath(expanduser(expandvars(path)))


CONFIG_FILENAME = "repo-cli-config.yaml"
if "REPO_CONFIG_DIR" in os.environ:
    dirs = EnvAppDirs(".conda", os.environ["REPO_CONFIG_DIR"])
    USER_CONFIG = join(dirs.user_data_dir, CONFIG_FILENAME)
else:
    # we need to keep it to binstar to be compatible with conda. ContinuuIO is important for
    # Windows users
    dirs = AppDirs("binstar", "ContinuumIO")
    USER_CONFIG = expand("~/.conda/" + CONFIG_FILENAME)
# if 'BINSTAR_CONFIG_DIR' in os.environ:
#     dirs = EnvAppDirs('binstar', 'ContinuumIO', os.environ['BINSTAR_CONFIG_DIR'])
#     USER_CONFIG = join(dirs.user_data_dir, 'config.yaml')
# else:
#     dirs = AppDirs('binstar', 'ContinuumIO')
#     USER_CONFIG = expand('~/.continuum/anaconda-client/config.yaml')


# Package types used in upload/download
PACKAGE_TYPES = {
    "env": "Environment",
    "ipynb": "Notebook",
    "conda": "Conda Package",
    "pypi": "Python Package",
    "project": "Anaconda Project",
}

UPLOAD_TYPE_MAPPING = {
    "conda": "conda1",
    "env": "anaconda_env",
    "ipynb": "jupyter_notebook",
    "project": "anaconda_project",
    "pypi": "bdist_wheel",
    "sbom": "sbom",
}

USER_LOGDIR = dirs.user_log_dir
SITE_CONFIG = expand("$CONDA_ROOT/etc/anaconda-client/%s" % (CONFIG_FILENAME,))
SYSTEM_CONFIG = SITE_CONFIG

OIDC_CLIENT_ID = "repo-service-cli"

DEFAULT_CONFIG = {
    "sites": {},
    "auto_register": True,
    "default_site": None,
    "url": None,
    "ssl_verify": True,
    "oauth2": False,
}

CONFIGURATION_KEYS = [
    "auto_register",
    "default_site",
    "sites",
    "url",
    "ssl_verify",
    "oauth2",
]

SEARCH_PATH = (
    "~/.conda/",
    dirs.site_data_dir,
    "/etc/anaconda-client/",
    "$CONDA_ROOT/etc/anaconda-client/",
    dirs.user_data_dir,
    "~/.continuum/anaconda-client/",
    "$CONDA_PREFIX/etc/anaconda-client/",
)


def recursive_update(config, update_dict):
    for update_key, updated_value in update_dict.items():
        if isinstance(updated_value, Mapping):
            updated_value_dict = recursive_update(
                config.get(update_key, {}), updated_value
            )
            config[update_key] = updated_value_dict
        else:
            config[update_key] = update_dict[update_key]

    return config


TOKEN_DIRS = [
    dirs.user_data_dir,
    join(dirname(USER_CONFIG), "tokens"),
]
TOKEN_DIR = TOKEN_DIRS[-1]


def store_token(token, args):
    config = get_config(site=args and args.site)

    for token_dir in TOKEN_DIRS:
        url = config.get("url")

        if not isdir(token_dir):
            os.makedirs(token_dir)
        tokenfile = join(token_dir, "%s.token" % quote_plus(url))

        if isfile(tokenfile):
            os.unlink(tokenfile)

        logger.debug(f"Saving token {token} to {tokenfile}")
        with open(tokenfile, "w") as fd:
            logger.debug(f"Saving token {token} to {tokenfile}")
            fd.write(token)
        os.chmod(tokenfile, stat.S_IWRITE | stat.S_IREAD)

        # TODO: This is a hack
        tokenfile = join(token_dir, "%s.token" % quote_plus(join(url, "repo")))
        logger.debug(f"Saving token {token} to {tokenfile}")
        with open(tokenfile, "w") as fd:
            logger.debug(f"Saving token {token} to {tokenfile}")
            fd.write(token)
        os.chmod(tokenfile, stat.S_IWRITE | stat.S_IREAD)


def load_token(site):
    config = get_config(site=site)
    url = config.get("url")
    if not url:
        logger.debug(
            "Repository URL is not set. Please configure. Skipping token check"
        )
        return

    for token_dir in TOKEN_DIRS:
        logger.debug(
            "[LOAD TOKEN] Looking for token in {token_dir}".format(token_dir=token_dir)
        )
        tokenfile = join(token_dir, "%s.token" % quote_plus(url))
        _is_file = exists(tokenfile)

        if _is_file:
            logger.debug("[LOAD TOKEN] Found login token: {}".format(tokenfile))
            with open(tokenfile) as fd:
                token = fd.read().strip()

            if token:
                return token
            else:
                logger.debug("Token file is empty: {}".format(tokenfile))
                logger.debug("Removing file: {}".format(tokenfile))
                os.unlink(tokenfile)
        else:
            logger.debug(
                "[LOAD TOKEN] Token file {tokenfile} not found.".format(
                    tokenfile=tokenfile
                )
            )


def remove_token(args):
    config = get_config(site=args and args.site)
    url = config.get("url")

    for token_dir in TOKEN_DIRS:
        tokenfile = join(token_dir, "%s.token" % quote_plus(url))
        if isfile(tokenfile):
            os.unlink(tokenfile)

        tokenfile = join(token_dir, "%s.token" % quote_plus(join(url, "repo")))
        if isfile(tokenfile):
            os.unlink(tokenfile)


def load_config(config_file):
    if exists(config_file):
        with open(config_file) as fd:
            data = yaml_load(fd)
            if data:
                return data

    return {}


def load_file_configs(search_path):
    def _file_yaml_loader(fullpath):
        assert (
            fullpath.endswith(".yml")
            or fullpath.endswith(".yaml")
            or fullpath.endswith("anacondarc")
        ), fullpath
        yield fullpath, load_config(fullpath)

    def _dir_yaml_loader(fullpath):
        for filename in os.listdir(fullpath):
            if filename == CONFIG_FILENAME:
                filepath = join(fullpath, filename)
                yield filepath, load_config(filepath)

    # map a stat result to a file loader or a directory loader
    _loader = {
        stat.S_IFREG: _file_yaml_loader,
        stat.S_IFDIR: _dir_yaml_loader,
    }

    def _get_st_mode(path):
        # stat the path for file type, or None if path doesn't exist
        try:
            return stat.S_IFMT(os.stat(path).st_mode)
        except OSError:
            return None

    expanded_paths = [expand(path) for path in search_path]
    stat_paths = (_get_st_mode(path) for path in expanded_paths)
    load_paths = (
        _loader[st_mode](path)
        for path, st_mode in zip(expanded_paths, stat_paths)
        if st_mode is not None
    )
    raw_data = collections.OrderedDict(
        kv for kv in itertools.chain.from_iterable(load_paths)
    )

    return raw_data


def get_config(site=None):
    config = DEFAULT_CONFIG.copy()

    file_configs = load_file_configs(SEARCH_PATH)
    for fn in file_configs:
        recursive_update(config, file_configs[fn])

    site = site or config.get("default_site")
    sites = config.get("sites", {})

    if site:
        site = str(site)

        if site not in sites:
            logger.warning('Site alias "%s" does not exist in the config file', site)
        else:
            # This takes whatever keys are set for the site into the top level of the config dict
            recursive_update(config, sites.get(site, {}))

    return config


def save_config(data, config_file):
    data_dir = dirname(config_file)

    try:
        if not exists(data_dir):
            os.makedirs(data_dir)

        with open(config_file, "w") as fd:
            yaml_dump(data, stream=fd)
    except EnvironmentError as exc:
        raise RepoCLIError(
            "%s: %s"
            % (
                exc.filename,
                exc.strerror,
            )
        )


def set_config(data, user=True):
    warnings.warn("Use save_config instead of set_config", DeprecationWarning)
    save_config(data, USER_CONFIG if user else SYSTEM_CONFIG)
