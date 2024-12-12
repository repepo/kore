# -*- coding: utf-8 -*-

# pylint: disable=broad-except,invalid-name

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Worker threads for downloading files."""

from __future__ import annotations

__all__ = ['DownloadAPI']

import collections
import enum
import json
import os
import typing
import urllib3.exceptions

from qtpy.QtCore import QBuffer, QObject, QThread, QTimer, Signal  # pylint: disable=no-name-in-module
import requests

from anaconda_navigator.api.client_api import ClientAPI
from anaconda_navigator.api.conda_api import CondaAPI
from anaconda_navigator.config import CONF
from anaconda_navigator.utils.logs import http_logger, logger
from anaconda_navigator.utils.py3compat import to_text_string
from anaconda_navigator.utils import url_utils
from . import utils as api_utils


# In case verify is False, this prevents spamming the user console with messages
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DownloadWorker(QObject):  # pylint: disable=too-many-instance-attributes
    """Download Worker based on requests."""

    sig_chain_finished = Signal(object, object, object)
    sig_download_finished = Signal(str, str)
    sig_download_progress = Signal(str, str, int, int)
    sig_partial = Signal(object, object, object)
    sig_finished = Signal(object, object, object)

    def __init__(self, method, args, kwargs):
        """Download Worker based on requests."""
        super().__init__()
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self._is_finished = False

    def _handle_partial(self, data):
        self.sig_partial.emit(self, data, None)

    def is_finished(self):
        """Return True if worker status is finished otherwise return False."""
        return self._is_finished

    def start(self):
        """Start process worker for given method args and kwargs."""
        error = None
        output = None

        try:
            output = self.method(*self.args, **self.kwargs)
        except Exception as err:
            print(err)
            error = err

        self.sig_finished.emit(self, output, error)
        self._is_finished = True


class ErrorDetail(enum.IntEnum):
    """Verbose values for :code:`False` validation results."""

    no_internet = enum.auto()
    ssl_error = enum.auto()
    http_error = enum.auto()
    general_error = enum.auto()

    def __bool__(self) -> bool:
        """Retrieve :class:`~bool` equivalent of the instance."""
        return False


class _DownloadAPI(QObject):  # pylint: disable=too-many-instance-attributes
    """Download API based on requests."""

    _sig_download_finished = Signal(str, str)
    _sig_download_progress = Signal(str, str, int, int)
    _sig_partial = Signal(object)

    MAX_THREADS = 20
    DEFAULT_TIMEOUT = 5  # seconds

    def __init__(self):
        """Download API based on requests."""
        super().__init__()
        self._conda_api = CondaAPI()
        self._client_api = ClientAPI()
        self._queue = collections.deque()
        self._queue_workers = collections.deque()
        self._threads = []
        self._workers = []
        self._timer = QTimer()
        self._timer_worker_delete = QTimer()
        self._running_threads = 0
        self._bag_collector = collections.deque()  # Keeps references to old workers

        self._chunk_size = 1024
        self._timer.setInterval(333)
        self._timer.timeout.connect(self._start)
        self._timer_worker_delete.setInterval(5000)
        self._timer_worker_delete.timeout.connect(self._clean_workers)

    def _clean_workers(self):
        """Delete periodically workers in workers bag."""
        while self._bag_collector:
            self._bag_collector.popleft()
        self._timer_worker_delete.stop()

    def _get_verify_ssl(self, verify, set_conda_ssl=True):
        """Get verify ssl."""
        if verify is None:
            verify_value = self._client_api.get_ssl(set_conda_ssl=set_conda_ssl)
        else:
            verify_value = verify
        return verify_value

    @staticmethod
    def _is_internet_available():
        """Check initernet availability."""

        if CONF.get('main', 'offline_mode'):
            connectivity = False
        else:
            connectivity = True  # is_internet_available()

        return connectivity

    def proxy_servers(self):
        """Return the proxy servers available from the conda rc config file."""
        return self._conda_api.load_proxy_config()

    def _start(self):
        """Start threads and check for inactive workers."""
        if self._queue_workers and self._running_threads < self.MAX_THREADS:
            self._running_threads += 1
            thread = QThread()
            worker = self._queue_workers.popleft()
            worker.moveToThread(thread)
            worker.sig_finished.connect(thread.quit)
            thread.started.connect(worker.start)
            thread.start()
            self._threads.append(thread)

        if self._workers:
            for w in self._workers:
                if w.is_finished():
                    self._bag_collector.append(w)
                    self._workers.remove(w)

        if self._threads:
            for t in self._threads:
                if t.isFinished():
                    self._threads.remove(t)
                    self._running_threads -= 1

        if len(self._threads) == 0 and len(self._workers) == 0:
            self._timer.stop()
            self._timer_worker_delete.start()

    def _create_worker(self, method, *args, **kwargs):
        """Create a new worker instance."""
        worker = DownloadWorker(method, args, kwargs)
        self._workers.append(worker)
        self._queue_workers.append(worker)
        self._sig_download_finished.connect(worker.sig_download_finished)
        self._sig_download_progress.connect(worker.sig_download_progress)
        self._sig_partial.connect(worker._handle_partial)  # pylint: disable=protected-access
        self._timer.start()
        return worker

    def _download(  # pylint: disable=too-many-arguments,too-many-branches,too-many-locals
        self,
        url: str,
        path: typing.Optional[str] = None,
        force: bool = False,
        verify: typing.Optional[bool] = None,
        chunked: bool = True,
    ) -> str:
        """Callback for download."""
        verify_value = self._get_verify_ssl(verify, set_conda_ssl=False)

        if path is None:
            path = url_utils.file_name(url)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        if not self._is_internet_available():
            self._sig_download_finished.emit(url, path)
            return path

        # Get headers
        try:
            response = requests.head(
                url,
                proxies=self.proxy_servers(),
                verify=api_utils.normalize_certificate(verify_value),
                timeout=self.DEFAULT_TIMEOUT,
            )
            http_logger.http(response=response)
            response.raise_for_status()
        except Exception as error:
            logger.error(str(error))
            return path

        total_size = int(response.headers.get('Content-Length', 0))

        # Check if file exists
        if os.path.isfile(path) and not force:
            file_size = os.path.getsize(path)
        else:
            file_size = -1

        # Check if existing file matches size of requested file
        if file_size == total_size:
            self._sig_download_finished.emit(url, path)
            return path

        try:
            response = requests.get(
                url,
                stream=chunked,
                proxies=self.proxy_servers(),
                verify=api_utils.normalize_certificate(verify_value),
                timeout=self.DEFAULT_TIMEOUT,
            )
            http_logger.http(response=response)
            response.raise_for_status()
        except Exception as error:
            logger.error(str(error))
            return path

        # File not found or file size did not match. Download file.
        progress_size = 0
        bytes_stream = QBuffer()  # BytesIO was segfaulting for big files
        bytes_stream.open(QBuffer.ReadWrite)

        # For some chunked content the app segfaults (with big files)
        # so now chunked is a kwarg for this method
        if chunked:
            for chunk in response.iter_content(chunk_size=self._chunk_size):
                # print(url, progress_size, total_size)
                if chunk:
                    bytes_stream.write(chunk)
                    progress_size += len(chunk)
                    self._sig_download_progress.emit(
                        url,
                        path,
                        progress_size,
                        total_size,
                    )

                    self._sig_partial.emit(
                        {
                            'url': url,
                            'path': path,
                            'progress_size': progress_size,
                            'total_size': total_size,
                        }
                    )

        else:
            bytes_stream.write(response.content)

        bytes_stream.seek(0)
        data = bytes_stream.data()

        with open(path, 'wb') as f:
            f.write(data)

        bytes_stream.close()

        self._sig_download_finished.emit(url, path)
        return path

    def _is_valid_url(self, url, verify=None):
        """Callback for is_valid_url."""
        verify_value = self._get_verify_ssl(verify)

        if self._is_internet_available():
            try:
                r = requests.head(
                    url,
                    proxies=self.proxy_servers(),
                    verify=api_utils.normalize_certificate(verify_value),
                    timeout=self.DEFAULT_TIMEOUT,
                )
                http_logger.http(response=r)
                value = r.status_code in [200]
            except Exception as error:
                logger.error(str(error))
                value = False

        return value

    def _is_valid_channel(
        self,
        channel,
        conda_url='https://conda.anaconda.org',
        verify=None,
    ):
        """Callback for is_valid_channel."""
        verify_value = self._get_verify_ssl(verify)

        if channel.startswith('https://') or channel.startswith('http://'):
            url = channel
        else:
            url = f'{conda_url}/{channel}'

        if url[-1] == '/':
            url = url[:-1]

        plat = self._conda_api.get_platform()
        repodata_url = f'{url}/{plat}/repodata.json'

        if self._is_internet_available():
            try:
                r = requests.head(
                    repodata_url,
                    proxies=self.proxy_servers(),
                    verify=api_utils.normalize_certificate(verify_value),
                    timeout=self.DEFAULT_TIMEOUT,
                )
                http_logger.http(response=r)
                value = r.status_code in [200]
            except Exception as error:
                logger.error(str(error))
                value = False

        return value

    def _is_valid_api_url(self, url, verify=None, allow_blank=False):
        """Callback for is_valid_api_url."""
        if allow_blank and (not url):
            return True

        if verify is None:
            verify = self._client_api.get_ssl()

        if not self._is_internet_available():
            return ErrorDetail.no_internet

        try:
            response = requests.get(
                url,
                proxies=self.proxy_servers(),
                verify=api_utils.normalize_certificate(verify),
                timeout=self.DEFAULT_TIMEOUT,
            )
            http_logger.http(response=response)
            data = response.json()

        except requests.exceptions.SSLError as error:
            logger.exception(error)
            return ErrorDetail.ssl_error

        except requests.HTTPError as error:
            logger.exception(error)
            return ErrorDetail.http_error

        except Exception as error:
            logger.exception(error)
            return ErrorDetail.general_error

        # data.get('status') == 'ok' is required for Anaconda Server check
        return (data.get('ok', 0) == 1) or (data.get('status', '') == 'ok')

    def _get_api_info(self, url, verify=None):
        """Callback."""
        verify_value = self._get_verify_ssl(verify)
        data = {
            'api_url': url,
            'api_docs_url': 'https://api.anaconda.org/docs',
            'conda_url': 'https://conda.anaconda.org/',
            'main_url': 'https://anaconda.org/',
            'pypi_url': 'https://pypi.anaconda.org/',
            'swagger_url': 'https://api.anaconda.org/swagger.json',
        }
        if self._is_internet_available():
            try:
                r = requests.get(
                    url,
                    proxies=self.proxy_servers(),
                    verify=api_utils.normalize_certificate(verify_value),
                    timeout=self.DEFAULT_TIMEOUT,
                )
                http_logger.http(response=r)
                content = to_text_string(r.content, encoding='utf-8')
                new_data = json.loads(content)
                data['conda_url'] = new_data.get('conda_url', data['conda_url'])
            except Exception as error:
                logger.error(str(error))

        return data

    # --- Public API
    # -------------------------------------------------------------------------
    def download(self, url, path=None, force=False, verify=None, chunked=True):  # pylint: disable=too-many-arguments
        """Download file given by url and save it to path."""
        method = self._download
        return self._create_worker(
            method,
            url,
            path=path,
            force=force,
            verify=verify,
            chunked=chunked,
        )

    def terminate(self):
        """Terminate all workers and threads."""
        for t in self._threads:
            t.quit()
        self._thread = []  # pylint: disable=attribute-defined-outside-init
        self._workers = []

    def is_valid_url(self, url, non_blocking=True):
        """Check if url is valid."""
        if non_blocking:
            method = self._is_valid_url
            return self._create_worker(method, url)
        return self._is_valid_url(url)

    def is_valid_api_url(self, url, non_blocking=True, verify=None, allow_blank=False):
        """Check if anaconda api url is valid."""
        if non_blocking:
            return self._create_worker(self._is_valid_api_url, url=url, verify=verify, allow_blank=allow_blank)
        return self._is_valid_api_url(url=url, verify=verify, allow_blank=allow_blank)

    def is_valid_channel(
        self,
        channel,
        conda_url='https://conda.anaconda.org',
        non_blocking=True,
    ):
        """Check if a conda channel is valid."""
        if non_blocking:
            method = self._is_valid_channel
            return self._create_worker(method, channel, conda_url)
        return self._is_valid_channel(channel, conda_url=conda_url)

    def get_api_info(self, url, non_blocking=True):
        """Query anaconda api info."""
        if non_blocking:
            method = self._get_api_info
            return self._create_worker(method, url)
        return self._get_api_info(url)


DOWNLOAD_API = None


def DownloadAPI():
    """Download API threaded worker based on requests."""
    global DOWNLOAD_API  # pylint: disable=global-statement

    if DOWNLOAD_API is None:
        DOWNLOAD_API = _DownloadAPI()

    return DOWNLOAD_API
