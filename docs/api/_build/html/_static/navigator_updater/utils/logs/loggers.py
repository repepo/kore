# -*- coding: utf-8 -*-

"""Store logger instances. """

from __future__ import annotations

__all__ = ['logger', 'conda_logger', 'http_logger']

import logging
import typing

if typing.TYPE_CHECKING:
    import requests


class ExtendedLogger(logging.Logger):
    """Custom logger for logging http requests"""

    def http(
            self,
            msg: typing.Optional[str] = None,
            response: typing.Optional['requests.Response'] = None,
            **kwargs: typing.Any,
    ) -> None:
        """Method which extract request and response data and add them into log as extra"""
        if msg is None and response is None:
            raise ValueError('At least `msg` or `response` must be specified!')

        extra = kwargs.get('extra', {})
        if response is not None:
            if msg is None:
                msg = f'[HTTP] "{response.request.method} {response.request.url}" {response.status_code}'

            extra = {
                'request_method': response.request.method,
                'request_url': response.request.url,
                'response_code': response.status_code,
            }

        self.debug(msg, **kwargs, extra=extra)


logging.setLoggerClass(ExtendedLogger)

logger = logging.getLogger()
conda_logger = logging.getLogger('navigator_updater.conda')
http_logger: ExtendedLogger = typing.cast(ExtendedLogger, logging.getLogger('navigator_updater.http'))
