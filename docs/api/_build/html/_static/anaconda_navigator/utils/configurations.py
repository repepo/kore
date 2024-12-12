# -*- coding: utf-8 -*-

"""Utilities to simplify configuration files management."""

from __future__ import annotations

__all__ = ['Updater', 'JsonUpdater', 'YamlUpdater']

import abc
import contextlib
import types
import typing
import json
import yaml


KEEP: typing.Any = object()


class Updater(contextlib.AbstractContextManager, metaclass=abc.ABCMeta):
    """
    Abstract helper that allows updating configuration files of general formats.

    It is used to hide general actions of reading and writing files with corresponding serialization.

    This class might also be used as a context manager. On successful exit - content of the configuration will be saved
    automatically.

    .. code-block:: python

        with JsonUpdater(path='/some/file') as configuration:
            configuration.content['some_property'] = 'new_value'

        # configuration file is saved automatically, unless an exception raised in main body

    :param path: Path to the configuration file.
    """

    __slots__ = ('__content', '__path')

    def __init__(self, path: str) -> None:
        """Initialize new :class:`~Updater` instance."""
        self.__path: str = path
        self.__content: typing.Any = self._read()

    @property
    def content(self) -> typing.Any:  # noqa: D401
        """Content of the configuration file."""
        return self.__content

    @content.setter
    def content(self, value: str) -> None:
        """Update `content` value."""
        self.__content = value

    @property
    def path(self) -> str:  # noqa: D401
        """Path to the configuration."""
        return self.__path

    @abc.abstractmethod
    def _read(self) -> typing.Any:
        """
        Read content of the configuration file.

        .. note::

            You can use value of the :attr:`~Updater.path` to locate the configuration file.

        .. warning::

            Do not use :attr:`~Updater.content`! This is the value that must be set by current function, and which must
            not rely on the previous configuration content.

        :return: Content of the configuration file.
        """

    @abc.abstractmethod
    def _write(self) -> None:
        """
        Write new configuration to the file.

        .. note::

            You can use value of the :attr:`~Updater.path` to locate the configuration file, and
            :attr:`~Updater.content` for the content of the configuration that should be saved.
        """

    def load(self) -> None:
        """Load configuration from the file."""
        self.__content = self._read()

    def save(self, path: str = KEEP) -> None:
        """
        Save configuration to the file.

        :param path: New path to save configuration to. This will also update value of the :attr:`~Updater.path` and
                     affect subsequent :meth:`~Updater.load` calls (unless there would be an error writing to file).

                     If not provided - current :attr:`~Updater.path` will be used.
        """
        path_backup: str = self.__path
        if path is not KEEP:
            self.__path = path
        try:
            self._write()
        except BaseException:
            self.__path = path_backup
            raise

    def __enter__(self) -> Updater:
        """Enter into the configuration updating context."""
        return self

    def __exit__(
            self,
            exc_type: typing.Optional[typing.Type[BaseException]],
            exc_val: typing.Optional[BaseException],
            exc_tb: typing.Optional[types.TracebackType],
    ) -> None:
        """
        Exit from the configuration updating context.

        If no exception raised - configuration would be saved.
        """
        if exc_val is None:
            self.save()


class EncodingAwareUpdater(Updater, metaclass=abc.ABCMeta):
    """
    Modified :class:`~Updater`, which also allows setting encoding to read configuration with.

    :param path: Path to the configuration file.
    :param encoding: Encoding to use for the configuration file.
    """

    __slots__ = ('__encoding',)

    def __init__(self, path: str, encoding: typing.Optional[str] = None) -> None:
        """Initialize new :class:`~EncodingAwareUpdater` instance."""
        self.__encoding: typing.Optional[str] = encoding
        super().__init__(path=path)

    @property
    def encoding(self) -> typing.Optional[str]:  # noqa: D401
        """Current encoding of the configuration file."""
        return self.__encoding

    def save(self, path: str = KEEP, encoding: typing.Optional[str] = KEEP) -> None:  # pylint: disable=arguments-differ
        """
        Save configuration to the file.

        :param path: New path to save configuration to. This will also update value of the
                     :attr:`~EncodingAwareUpdater.path` and affect subsequent :meth:`~Updater.load` calls (unless there
                     would be an error writing to file).

                     If not provided - current :attr:`~EncodingAwareUpdater.path` will be used.

        :param encoding: New encoding to save configuration file with. This also updates
                         :attr:`~EncodingAwareUpdater.encoding` in the way similar to `path` argument.
        """
        encoding_backup: typing.Optional[str] = self.__encoding
        if encoding is not KEEP:
            self.__encoding = encoding
        try:
            super().save(path=path)
        except BaseException:
            self.__encoding = encoding_backup
            raise


class JsonUpdater(EncodingAwareUpdater):
    """
    Configuration updater, that supports configuration files with JSON content.

    :param path: Path to the configuration file.
    :param encoding: Encoding to use for the configuration file.
    """

    __slots__ = ()

    def _read(self) -> typing.Any:
        """
        Read content of the configuration file.

        :return: Content of the configuration file.
        """
        stream: typing.TextIO
        with open(self.path, 'rt', encoding=self.encoding) as stream:
            return json.load(stream)

    def _write(self) -> None:
        """Write new configuration to the file."""
        stream: typing.TextIO
        with open(self.path, 'wt', encoding=self.encoding) as stream:
            json.dump(self.content, stream)


class YamlUpdater(EncodingAwareUpdater):
    """
    Configuration updater, that supports configuration files with YAML content.

    :param path: Path to the configuration file.
    :param encoding: Encoding to use for the configuration file.
    """

    __slots__ = ()

    def _read(self) -> typing.Any:
        """
        Read content of the configuration file.

        :return: Content of the configuration file.
        """
        stream: typing.TextIO
        with open(self.path, 'rt', encoding=self.encoding) as stream:
            return yaml.load(stream, yaml.FullLoader)  # nosec

    def _write(self) -> None:
        """Write new configuration to the file."""
        stream: typing.TextIO
        with open(self.path, 'wt', encoding=self.encoding) as stream:
            yaml.dump(self.content, stream)
