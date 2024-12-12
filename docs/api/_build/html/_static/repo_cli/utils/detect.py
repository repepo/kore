"""
Package type detection and meta-data extraction
"""
from __future__ import print_function, unicode_literals

import json
import logging
import tarfile
import zipfile
from os import path

# from binstar_client.inspect_package.conda import inspect_conda_package
# from binstar_client.inspect_package import conda_installer
# from binstar_client.inspect_package.pypi import inspect_pypi_package
# from binstar_client.inspect_package.r import inspect_r_package
# from binstar_client.inspect_package.ipynb import inspect_ipynb_package
# from binstar_client.inspect_package.env import inspect_env_package

logger = logging.getLogger("binstar.detect")


def file_handler(filename, fileobj, *args, **kwargs):
    return ({}, {"description": ""}, {"basename": path.basename(filename), "attrs": {}})


detectors = {
    # 'conda': inspect_conda_package,
    # 'pypi': inspect_pypi_package,
    # 'r': inspect_r_package,
    # 'ipynb': inspect_ipynb_package,
    # 'env': inspect_env_package,
    # conda_installer.PACKAGE_TYPE: conda_installer.inspect_package,
    "file": file_handler,
}


def is_environment(filename):
    logger.debug("Testing if environment file ..")
    if filename.endswith(".yml") or filename.endswith(".yaml"):
        return True
    logger.debug("No environment file")


def is_ipynb(filename):
    logger.debug("Testing if ipynb file ..")
    if filename.endswith(".ipynb"):
        return True
    logger.debug("No ipynb file")


def is_anaconda_project_yaml(filename):
    return filename == "anaconda-project.yml" or filename.endswith(
        "/anaconda-project.yml"
    )


def is_project(filename):
    logger.debug("Testing if project ..")

    def is_python_file():
        return filename.endswith(".py")

    def is_directory():
        return path.isdir(filename)

    if is_directory() or is_python_file():
        return True

    if filename.endswith(".tar.gz") or filename.endswith(".tar.bz2"):
        compression = filename.rsplit(".", maxsplit=1)[1]
        with tarfile.open(filename, mode="r|%s" % compression) as tf:
            for name in tf.getnames():
                if is_anaconda_project_yaml(name):
                    return True

    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename) as zf:
            for name in zf.namelist():
                if is_anaconda_project_yaml(name):
                    return True

    logger.debug("Not a project")


def is_conda(filename):
    logger.debug("Testing if conda package ..")

    if filename.endswith(".tar.bz2"):  # Could be a conda package
        try:
            with tarfile.open(filename, mode="r|bz2") as tf:
                for info in tf:
                    if info.name == "info/index.json":
                        break
                else:
                    raise KeyError
        except KeyError:
            logger.debug("Not conda  package no 'info/index.json' file in the tarball")
            return False
        else:
            logger.debug("This is a conda package")
            return True
    logger.debug("Not conda package (file ext is not .tar.bz2)")


def is_pypi(filename):
    logger.debug("Testing if pypi package ..")
    if filename.endswith(".whl"):
        logger.debug("This is a pypi wheel package")
        return True
    if filename.endswith(".tar.gz") or filename.endswith(
        ".tgz"
    ):  # Could be a setuptools sdist or r source package
        with tarfile.open(filename) as tf:
            if any(name.endswith("/PKG-INFO") for name in tf.getnames()):
                return True
            else:
                logger.debug("This not is a pypi package (no '/PKG-INFO' in tarball)")
                return False

    logger.debug("This not is a pypi package (expected .tgz, .tar.gz or .whl)")


def is_r(filename):
    logger.debug("Testing if R package ..")
    if filename.endswith(".tar.gz") or filename.endswith(
        ".tgz"
    ):  # Could be a setuptools sdist or r source package
        with tarfile.open(filename) as tf:

            if any(name.endswith("/DESCRIPTION") for name in tf.getnames()) and any(
                name.endswith("/NAMESPACE") for name in tf.getnames()
            ):
                return True
            else:
                logger.debug(
                    "This not is an R package (no '*/DESCRIPTION' and '*/NAMESPACE' files)."
                )
    else:
        logger.debug("This not is an R package (expected .tgz, .tar.gz).")


def is_sbom(filename):
    # Fail fast if filename does not have the expected extension.
    if not filename.endswith(".spdx.json"):
        return False

    logger.debug("Testing if SBOM document ..")
    with open(filename) as sbom:
        content = json.load(sbom)

    # Look for the spdx version
    if not content.get("spdxVersion"):
        logger.warn("Document contains no version information!")
        return False

    # Make sure there's a package section defined.
    packages = content.get("packages")
    if not packages or len(packages) < 1:
        logger.warn("Document contains no package information!")
        return False

    # Look for the package file sha256, which is needed to correlate this
    # sbom with the package internally.
    checksums = packages[0].get("checksums")
    if not checksums:
        logger.warn("Document contains no package checksums!")
        return False
    sha256s = [x for x in checksums if x.get("algorithm") == "SHA256"]
    if sha256s and sha256s[0].get("checksumValue"):
        return True
    logger.warn("Document does not contain the needed package sha256 hash!")
    return False


def detect_package_type(filename):
    if isinstance(filename, bytes):
        filename = filename.decode("utf-8", errors="ignore")

    if is_conda(filename):
        return "conda"
    if is_project(filename):
        return "project"
    if is_pypi(filename):
        return "pypi"
    if is_r(filename):
        return "r"
    if is_ipynb(filename):
        return "ipynb"
    if is_environment(filename):
        return "env"
    if is_sbom(filename):
        return "sbom"
    return None


def get_attrs(package_type, filename, *args, **kwargs):
    with open(filename, "rb") as fileobj:
        return detectors[package_type](filename, fileobj, *args, **kwargs)
