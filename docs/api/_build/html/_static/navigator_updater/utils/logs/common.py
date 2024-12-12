# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Logger utilities."""

__all__ = ['log_files', 'clean_logs', 'load_log', 'JSONFormatter']

import datetime
import logging.handlers
import os
import types
import typing

import ujson

from navigator_updater.config import LOG_FILENAME, LOG_FOLDER

LOG_TIME_FORMAT: str = '%Y-%m-%d %H:%M:%S'

_SysExcInfoType = typing.Union[
    typing.Tuple[typing.Type[BaseException], BaseException, typing.Optional[types.TracebackType]],
    typing.Tuple[None, None, None]
]

_LogRecordAttrType = typing.Union[int, float, str, _SysExcInfoType, None]


class JSONFormatter(logging.Formatter):
    """
    Simple logs formatter using JSON serialization.
    """

    def __init__(
            self,
            fields: typing.Optional[typing.Dict[str, str]] = None,
            always_extra: typing.Optional[typing.Dict[str, str]] = None,
            datefmt: typing.Optional[str] = LOG_TIME_FORMAT,
    ) -> None:
        """
        :param fields: A dictionary of fields to use in the log.
            The keys in the dictionary are keys that will be used in the
            final log form, and its values are the names of the attributes
            from the log record to use as final log values. Defaults to
            None, which is interpreted as an empty dict.
        :param always_extra: A dictionary of additional static
            values written to the final log. Defaults to None, which is
            interpreted as an empty dict.
        :param datefmt: strftime date format. For more details
            check logs.Formatter documentation. Defaults to None.
        """
        super().__init__(fmt=None, datefmt=datefmt, style='%')
        self.fields: typing.Dict[str, str] = fields or {}
        self._uses_time: bool = 'asctime' in self.fields.values()
        self.always_extra: typing.Dict[str, typing.Any] = always_extra or {}

    def usesTime(self) -> bool:
        """
        Check if the format uses the creation time of the record. For more
        information about the method see logs.Formatter.
        """
        return self._uses_time

    def format(self, record: logging.LogRecord) -> str:
        """
        Build a JSON serializable dict starting from `self.always_extra`,
        adding the data from the LogRecord specified in `self.fields`, and
        finally adding the record specific extra data.

        :param record: log record to be converted to string
        :returns: JSON serialized log record
        """
        data: typing.MutableMapping[str, typing.Any] = self.always_extra.copy()
        record.message = record.getMessage()

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        if record.stack_info:
            record.stack_info = self.formatStack(record.stack_info)

        key: str
        field: str
        for key, field in self.fields.items():
            value: _LogRecordAttrType = record.__dict__.get(field, None)

            if not value:
                continue

            if field == 'exc_text':
                value = record.exc_text if any(record.exc_info or []) else None

            data[key] = value

        return ujson.dumps(data)  # pylint: disable=c-extension-no-member


def log_files(log_folder: str = LOG_FOLDER, log_filename: str = LOG_FILENAME) -> typing.List[str]:
    """
    Return all available log files located inside the logs folder.

    Files starting with a `.` are ignored as well as files not including the
    `log_filename` as part of the name.
    """
    paths: typing.List[str] = []
    if not os.path.isdir(log_folder):
        return paths

    log_file: str
    for log_file in sorted(os.listdir(log_folder)):
        log_file_path: str = os.path.join(log_folder, log_file)

        if os.path.isfile(log_file_path) and (log_filename in log_file) and (not log_file.startswith('.')):
            paths.append(log_file_path)

    return paths


def clean_logs(log_folder: str = LOG_FOLDER) -> None:
    """Remove logs in old plain text format."""
    week_ago: datetime.datetime = datetime.datetime.now() - datetime.timedelta(days=7)
    to_datetime: typing.Callable[[str, str], datetime.datetime] = datetime.datetime.strptime

    path: str
    lines: typing.List[typing.Mapping[str, typing.Any]]
    for path in log_files(log_folder):
        lines = load_log(path)

        line: typing.Any
        new_lines: typing.List[str] = []
        for line in lines:
            try:
                if to_datetime(line['time'], LOG_TIME_FORMAT) < week_ago:
                    continue
            except (ValueError, TypeError, KeyError):
                pass
            new_lines.append(ujson.dumps(line, ensure_ascii=False) + '\n')  # pylint: disable=c-extension-no-member

        with open(path, 'w', encoding='utf-8') as stream:
            stream.writelines(new_lines)


def load_log(log_file_path: str) -> typing.List[typing.Mapping[str, typing.Any]]:
    """Load log file and return list of items."""
    if not os.path.isfile(log_file_path):
        return []

    json_lines: typing.List[typing.Mapping[str, typing.Any]] = []
    with open(log_file_path, 'r', encoding='utf-8') as stream:
        for line in stream:
            try:
                json_lines.append(ujson.loads(line))  # pylint: disable=c-extension-no-member
            except ValueError:
                pass

    return json_lines
