# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""
This module provides user configuration file management features.

It's based on the ConfigParser module (present in the standard library).
"""

from __future__ import print_function

import ast
import configparser
import contextlib
import copy
import io
import os
import re
import shutil
import time
from anaconda_navigator.config.base import get_home_dir


# -----------------------------------------------------------------------------
# --- Auxiliary classes
# -----------------------------------------------------------------------------
class NoDefault:  # pylint: disable=too-few-public-methods
    """NoDefault object."""


# -----------------------------------------------------------------------------
# --- Defaults class
# -----------------------------------------------------------------------------
class DefaultsConfig(configparser.ConfigParser):
    """Class used to save defaults to a file."""

    def __init__(self, name, subfolder):
        """
        Class used to save defaults to a file.

        Parameters
        -----------
        name: str
            Name of the configuration.
        subfolder: str
            Path to folder location of configuration file.
        """
        configparser.ConfigParser.__init__(self)
        self.name = name
        self.subfolder = subfolder

    def _write(self, fp):  # pylint: disable=invalid-name
        """Private write method for Python 2.

        The one from configparser fails for non-ascii Windows accounts.
        """
        if self._defaults:
            fp.write(f'[{configparser.DEFAULTSECT}]\n')
            for (key, value) in self._defaults.items():
                write_value = str(value).replace('\n', '\n\t')
                fp.write(f'{key} = {write_value}\n')
            fp.write('\n')
        for section in self._sections:
            fp.write(f'[{section}]\n')
            for (key, value) in self._sections[section].items():
                if key == '__name__':
                    continue
                if (value is not None) or (self._optcre == self.OPTCRE):
                    value = str(value)
                    key = ' = '.join((key, value.replace('\n', '\n\t')))
                fp.write(f'{key}\n')
            fp.write('\n')

    def _set(self, section, option, value, verbose):
        """Private set method."""
        if not self.has_section(section):
            self.add_section(section)
        if not isinstance(value, str):
            value = repr(value)
        if verbose:
            print(f'{section}[ {option} ] = {value}')
        configparser.ConfigParser.set(self, section, option, value)

    def _save(self):
        """Save config into the associated .ini file."""
        fname = self.filename()

        def _write_file(fname):
            with open(fname, 'w', encoding='utf-8') as configfile:
                self.write(configfile)

        try:  # the "easy" way
            _write_file(fname)
        except IOError:
            try:  # the "delete and sleep" way
                if os.path.isfile(fname):
                    os.remove(fname)
                time.sleep(0.05)
                _write_file(fname)
            except Exception:
                print('Failed to write user configuration file.')
                print('Please submit a bug report.')
                raise

    def filename(self):
        """Create a .ini filename located in user home directory."""
        folder = os.path.join(get_home_dir(), self.subfolder)
        # Save defaults in a "defaults" dir of .anaconda_navigator to not
        # pollute it
        if 'defaults' in self.name:
            folder = os.path.join(folder, 'defaults')
        try:
            os.makedirs(folder)
        except os.error:
            # Folder (or one of its parents) already exists
            pass
        ini_file = os.path.join(folder, f'{self.name}.ini')
        return ini_file

    def set_defaults(self, defaults):
        """Set default configuration."""
        for section, options in defaults:
            for option in options:
                new_value = copy.deepcopy(options[option])
                self._set(section, option, new_value, False)


# -----------------------------------------------------------------------------
# --- User config class
# -----------------------------------------------------------------------------
class UserConfig(DefaultsConfig):
    """
    User configuration class.

    Note that 'get' and 'set' arguments number and type differ from the
    overriden methods.
    """

    DEFAULT_SECTION_NAME = 'main'  # pylint: disable=invalid-name

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name,
        defaults=None,
        load=True,
        version=None,
        subfolder=None,
        backup=False,
        raw_mode=False,
        remove_obsolete=False
    ):
        """
        User configuration class.

        Parameters
        ----------
        name: str
            Name of the configuration.
        defaults: dict
            dictionnary containing options *or* list of tuples (section_name,
            options).
        version: str
            Version of the configuration file (X.Y.Z format).
        subfolder: str
            Configuration file will be saved in HOME/subfolder/name.ini.
        """
        DefaultsConfig.__init__(self, name, subfolder)

        self.raw = 1 if raw_mode else 0
        if (version is not None) and (re.match(r'^(\d+).(\d+).(\d+)$', version) is None):
            raise ValueError(f'Version number {version} is incorrect - must be in X.Y.Z format')

        if isinstance(defaults, dict):
            defaults = [(self.DEFAULT_SECTION_NAME, defaults)]
        self.defaults = defaults
        if defaults is not None:
            self.reset_to_defaults(save=False)

        fname = self.filename()
        if backup:
            try:
                shutil.copyfile(fname, f'{fname}.bak')
            except IOError:
                pass

        if load:
            # If config file already exists, it overrides Default options:
            self.load_from_ini()
            old_ver = self.get_version(version)

            # Save new defaults
            self.__save_new_defaults(defaults, version, subfolder)

            # Updating defaults only if major/minor version is different
            if self._minor(version) != self._minor(old_ver):
                if backup:
                    try:
                        shutil.copyfile(fname, f'{fname}-{old_ver}.bak')
                    except IOError:
                        pass
                self.__update_defaults(defaults, old_ver)

                # Remove deprecated options if major version has changed
                if (remove_obsolete or self._major(version) != self._major(old_ver)):
                    self.__remove_deprecated_options(old_ver)

                # Set new version number
                self.set_version(version, save=False)
            if defaults is None:
                # If no defaults are defined, set .ini file settings as default
                self.set_as_defaults()

    @staticmethod
    def _major(_t):
        """Return major component in config versions."""
        return _t[:_t.find('.')]

    @staticmethod
    def _minor(_t):
        """Return minor component in config versions."""
        return _t[:_t.rfind('.')]

    def get_version(self, version='0.0.0'):
        """Return configuration (not application!) version."""
        return self.get(self.DEFAULT_SECTION_NAME, 'version', version)

    def set_version(self, version='0.0.0', save=True):
        """Set configuration (not application!) version."""
        self.set(self.DEFAULT_SECTION_NAME, 'version', version, save=save)

    def set_logged_data(self, url=None, brand=None):
        """Set `logged_api_url` and `logged_brand` config options."""
        self.set(self.DEFAULT_SECTION_NAME, 'logged_api_url', url)
        self.set(self.DEFAULT_SECTION_NAME, 'logged_brand', brand)

    def get_logged_data(self):
        """Set `logged_brand` and `logged_api_url` config options."""
        return self.get(self.DEFAULT_SECTION_NAME, 'logged_brand',),\
            self.get(self.DEFAULT_SECTION_NAME, 'logged_api_url')

    def is_logged_brand(self, url=None, brand=None):
        """Check if given `url` or `brand` match current repo."""
        logged_brand, logged_url = self.get_logged_data()

        if brand and logged_brand:
            return brand == logged_brand

        if logged_url and url:
            return url == logged_url

        return False

    def load_from_ini(self):
        """Load config from the associated .ini file."""
        try:
            self.read(self.filename(), encoding='utf-8')
        except configparser.MissingSectionHeaderError:
            print('Warning: File contains no section headers.')

    def __load_old_defaults(self, old_version):
        """Read old defaults."""
        old_defaults = configparser.ConfigParser()
        path = os.path.dirname(self.filename())
        path = os.path.join(path, 'defaults')
        old_defaults.read(os.path.join(path, 'defaults-' + old_version + '.ini'))
        return old_defaults

    @staticmethod
    def __save_new_defaults(defaults, new_version, subfolder):
        """Save new defaults."""
        new_defaults = DefaultsConfig(
            name='defaults-' + new_version,
            subfolder=subfolder,
        )
        if not os.path.isfile(new_defaults.filename()):
            new_defaults.set_defaults(defaults)
            new_defaults._save()  # pylint: disable=protected-access

    def __update_defaults(self, defaults, old_version, verbose=False):
        """Update defaults after a change in version."""
        old_defaults = self.__load_old_defaults(old_version)
        for section, options in defaults:
            for option in options:
                new_value = copy.deepcopy(options[option])
                try:
                    old_value = old_defaults.get(section, option)
                except (configparser.NoSectionError, configparser.NoOptionError):
                    old_value = None
                if (old_value is None) or (str(new_value) != old_value):
                    self._set(section, option, new_value, verbose)

    def __remove_deprecated_options(self, old_version):
        """Remove options present in the .ini file but not in defaults."""
        old_defaults = self.__load_old_defaults(old_version)
        for section in old_defaults.sections():
            for option, _ in old_defaults.items(section, raw=self.raw):
                if self.get_default(section, option) is NoDefault:
                    self.remove_option(section, option)
                    if len(self.items(section, raw=self.raw)) == 0:
                        self.remove_section(section)

    def cleanup(self):
        """Remove .ini file associated to config."""
        os.remove(self.filename())

    def set_as_defaults(self):
        """Set defaults from the current config."""
        self.defaults = []
        for section in self.sections():
            secdict = {}
            for option, value in self.items(section, raw=self.raw):
                secdict[option] = value
            self.defaults.append((section, secdict))

    def get_defaults(self):  # pylint: disable=missing-function-docstring
        self.reset_to_defaults(save=False)

        with io.StringIO() as fd:  # pylint: disable=invalid-name
            self.write(fd)
            fd.seek(0)
            return fd.read()

    def reset_to_defaults(self, save=True, verbose=False, section=None):
        """Reset config to Default values."""
        for sec, options in self.defaults:
            if section is None or section == sec:
                for option in options:
                    value = copy.deepcopy(options[option])
                    self._set(sec, option, value, verbose)
        if save:
            self._save()

    def __check_section_option(self, section, option):
        """Private method to check section and option types."""
        if section is None:
            section = self.DEFAULT_SECTION_NAME
        elif not isinstance(section, str):
            raise RuntimeError("Argument 'section' must be a string")
        if not isinstance(option, str):
            raise RuntimeError("Argument 'option' must be a string")
        return section

    def get_default(self, section, option):
        """
        Get Default value for a given (section, option).

        -> useful for type checking in 'get' method
        """
        section = self.__check_section_option(section, option)
        for sec, options in self.defaults:
            if (sec == section) and (option in options):
                return copy.deepcopy(options[option])
        return NoDefault

    def get(self, section, option, default=NoDefault):  # pylint: disable=arguments-differ
        """
        Get an option.

        section=None: attribute a default section name
        default: default value (if not specified, an exception
        will be raised if option doesn't exist)
        """
        section = self.__check_section_option(section, option)

        if not self.has_section(section):
            if default is NoDefault:
                raise configparser.NoSectionError(section)
            self.add_section(section)

        if not self.has_option(section, option):
            if default is NoDefault:
                raise configparser.NoOptionError(option, section)
            self.set(section, option, default)
            return default

        value = configparser.ConfigParser.get(self, section, option, raw=self.raw)
        default_value = self.get_default(section, option)

        if isinstance(default_value, bool):
            value = ast.literal_eval(value)
        elif isinstance(default_value, float):
            value = float(value)
        elif isinstance(default_value, int):
            value = int(value)
        else:
            with contextlib.suppress(BaseException):
                # lists, tuples, ...
                value = ast.literal_eval(value)
        return value

    def set_default(self, section, option, default_value):
        """
        Set Default value for a given (section, option).

        -> called when a new (section, option) is set and no default exists
        """
        section = self.__check_section_option(section, option)
        for sec, options in self.defaults:
            if sec == section:
                options[option] = default_value

    def set(  # pylint: disable=arguments-differ,too-many-arguments
            self, section, option, value, verbose=False, save=True,
    ):
        """
        Set an option.

        section=None: attribute a default section name
        """
        section = self.__check_section_option(section, option)
        default_value = self.get_default(section, option)
        if default_value is NoDefault:
            default_value = value
            self.set_default(section, option, default_value)

        if isinstance(default_value, bool):
            value = bool(value)
        elif isinstance(default_value, float):
            value = float(value)
        elif isinstance(default_value, int):
            value = int(value)
        elif not isinstance(default_value, str):
            value = repr(value)
        self._set(section, option, value, verbose)
        if save:
            self._save()

    def remove_section(self, section):
        """Remove the specified section from the configuration."""
        configparser.ConfigParser.remove_section(self, section)
        self._save()

    def remove_option(self, section, option):
        """Remove the specified option from the specified section."""
        configparser.ConfigParser.remove_option(self, section, option)
        self._save()
