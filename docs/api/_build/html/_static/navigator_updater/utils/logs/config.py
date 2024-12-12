"""Logging configurations."""

__all__ = ('LOGGER_CONFIG',)

from dataclasses import dataclass
import logging
import os

from navigator_updater.config import LOG_FILENAME, LOG_FOLDER, MAX_LOG_FILE_SIZE

if not os.path.isdir(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)


@dataclass
class LoggerConfig:
    """Config variables and logging dictionary configuration. """
    file_path: str = os.path.join(LOG_FOLDER, LOG_FILENAME)
    backup_count: int = 5
    level: int = logging.INFO
    max_log_file_size = MAX_LOG_FILE_SIZE

    @property
    def dict_config(self):
        """Logging dict config."""
        return {
            'version': 1,
            'formatters': {
                'root_console': {
                    'format': '%(asctime)s - %(levelname)s %(module)s.%(funcName)s:%(lineno)d\n%(message)s\n'
                },
                'root_json': {
                    '()': 'navigator_updater.utils.logs.common.JSONFormatter',
                    'fields': {
                        'time': 'asctime',
                        'level': 'levelname',
                        'module': 'module',
                        'method': 'funcName',
                        'line': 'lineno',
                        'path': 'pathname',
                        'message': 'message',
                        'exception': 'exc_text',
                        'traceback': 'stack_info'
                    }
                },
                'conda_json': {
                    '()': 'navigator_updater.utils.logs.common.JSONFormatter',
                    'fields': {
                        'time': 'asctime',
                        'level': 'levelname',
                        'module': 'module',
                        'method': 'funcName',
                        'line': 'lineno',
                        'path': 'pathname',
                        'message': 'message',
                        'output': 'output',
                        'error': 'error',
                        'environment': 'environment',
                        'callback': 'callback',
                        'exception': 'exc_text',
                        'traceback': 'stack_info'
                    }
                },
                'http_json': {
                    '()': 'navigator_updater.utils.logs.common.JSONFormatter',
                    'fields': {
                        'time': 'asctime',
                        'level': 'levelname',
                        'module': 'module',
                        'method': 'funcName',
                        'line': 'lineno',
                        'path': 'pathname',
                        'message': 'message',
                        'exception': 'exc_text',
                        'traceback': 'stack_info',
                        'request_method': 'request_method',
                        'request_url': 'request_url',
                        'request_body': 'request_body',
                        'request_headers': 'request_headers',
                        'response_code': 'response_code',
                        'response_headers': 'response_headers',
                        'response_body': 'response_body',
                    }
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'root_console',
                    'level': LOGGER_CONFIG.level if LOGGER_CONFIG.level <= logging.DEBUG else max(
                        logging.WARNING, LOGGER_CONFIG.level),
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'root_json',
                    'level': LOGGER_CONFIG.level,
                    'maxBytes': LOGGER_CONFIG.max_log_file_size,
                    'filename': LOGGER_CONFIG.file_path,
                    'backupCount': LOGGER_CONFIG.backup_count
                },
                'conda_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'conda_json',
                    'level': LOGGER_CONFIG.level,
                    'maxBytes': LOGGER_CONFIG.max_log_file_size,
                    'filename': LOGGER_CONFIG.file_path,
                    'backupCount': LOGGER_CONFIG.backup_count
                },
                'http_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'http_json',
                    'level': LOGGER_CONFIG.level,
                    'maxBytes': LOGGER_CONFIG.max_log_file_size,
                    'filename': LOGGER_CONFIG.file_path,
                    'backupCount': LOGGER_CONFIG.backup_count
                }
            },
            'loggers': {
                '': {
                    'level': LOGGER_CONFIG.level,
                    'handlers': ['console', 'file']
                },
                'navigator_updater.conda': {
                    'level': logging.DEBUG,
                    'handlers': ['console', 'conda_file']
                },
                'navigator_updater.http': {
                    'level': logging.DEBUG,
                    'handlers': ['console', 'http_file']
                },
            }
        }


LOGGER_CONFIG = LoggerConfig()
