# -*- coding: utf-8 -*-

"""
Manager for downloading resources from the web.

This manager might be used to download regular files, however - it is designed particularly for the resources, that
should be downloaded once per application launch and used continuously.

Manager is optimized to detect repeated downloads and reuse previous download results.
"""

from __future__ import annotations

__all__ = ['Keep', 'Medium', 'Status', 'Download', 'Group', 'Manager', 'manager']

import contextlib
import enum
import os
import shutil
import tempfile
import typing
import uuid

from qtpy import QtCore
import requests

from anaconda_navigator.utils import logs
from anaconda_navigator.utils import workers


T_co = typing.TypeVar('T_co', covariant=True)


class ProcessDownload(typing.Protocol[T_co]):  # pylint: disable=too-few-public-methods
    """Common interface for functions to post-process download results."""

    def __call__(self, download: 'Download') -> T_co:
        """Prepare a `result` of some `download`."""


class ProcessGroup(typing.Protocol[T_co]):  # pylint: disable=too-few-public-methods
    """Common interface for functions to post-process group download results."""

    def __call__(self, group: 'Group') -> T_co:
        """Collect a `result` of some `group`."""


class GroupStats(typing.TypedDict):
    """Tracking stats of a group download."""

    canceled: typing.Set['Download']
    done: typing.Set['Download']
    failed: typing.Set['Download']
    requested: typing.Set['Download']
    started: typing.Set['Download']
    succeeded: typing.Set['Download']


def inject_cancel() -> bool:
    """Top-level function that injects "cancel if required" action into all downloads."""
    from anaconda_navigator import config  # pylint: disable=import-outside-toplevel

    return config.CONF.get('main', 'offline_mode', False)


def inject_kwargs() -> typing.Mapping[str, typing.Any]:
    """Top-level function that injects additional :mod:`requests` arguments into all requests."""
    from anaconda_navigator.api import client_api  # pylint: disable=import-outside-toplevel
    from anaconda_navigator.api import conda_api  # pylint: disable=import-outside-toplevel
    from anaconda_navigator.api import utils  # pylint: disable=import-outside-toplevel

    return {
        'proxies': conda_api.CondaAPI().load_proxy_config(),
        'verify': utils.normalize_certificate(client_api.ClientAPI().get_ssl(set_conda_ssl=False)),
        'timeout': 5,  # seconds
    }


def coalesce_source(sources: typing.Iterable[str]) -> str:
    """Find first file from `sources` that should be available."""
    source: str
    for source in sources:
        if os.path.isfile(source):
            return source
    raise ValueError('no usable source found')


def populate(source: str, destinations: typing.Iterable[str], overwrite: bool = False) -> None:
    """
    Copy content from `source` file into multiple `destinations`.

    By default - if any destination already exists, it would stay untouched. To change this behavior, set `overwrite` to
    :code:`True`.
    """
    destination: str
    for destination in destinations:
        destination = os.path.abspath(destination)
        if destination == source:
            continue
        if overwrite or (not os.path.isfile(destination)):
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)


def write(content: bytes, destinations: typing.Iterable[str], overwrite: bool = False) -> None:
    """
    Write binary `content` into multiple `destinations`.

    By default - if any destination already exists, it would stay untouched. To change this behavior, set `overwrite` to
    :code:`True`.
    """
    destination: str
    for destination in destinations:
        destination = os.path.abspath(destination)
        if overwrite or (not os.path.isfile(destination)):
            stream: typing.BinaryIO
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, 'wb') as stream:
                stream.write(content)


class Keep(enum.IntEnum):
    """
    Options of keeping files being downloaded, if they already exist.

    :code:`NEVER` - would overwrite file in any case, no matter it exists or not. This is the default behavior.

    :code:`FRESH` - file would stay untouched it already exists and of the expected size.

    :code:`ALWAYS` - do not overwrite file in any case.
    """

    NEVER = enum.auto()
    FRESH = enum.auto()
    ALWAYS = enum.auto()


class Medium(str, enum.Enum):
    """
    Intermediate source to store download results in.

    :code:`MEMORY` - download all content directly into the RAM. If no download destination specified - this option
    would be used automatically, regardless of the selected medium.

    :code:`RANDOM` - use random temporary file to store intermediate download results. After download is finished - this
    file is copied into all requested destinations. This is the default behavior.

    :code:`TARGET` - write content directly into the target file. This might be an insecure option if there are clients
    that expect complete file to be available.
    """

    MEMORY = ':memory:'
    RANDOM = ':random:'
    TARGET = ':target:'


class Require(enum.IntEnum):
    """
    Control how many downloads should succeed in a :class:`Group` to treat it as a success.

    :code:`ALL` - all downloads should succeed.

    :code:`ANY` - at least one download should succeed.

    :code:`NONE` - any outcome would be treated as a success.
    """

    ALL = enum.auto()
    ANY = enum.auto()
    NONE = enum.auto()


class Status(enum.IntEnum):
    """
    Possible statuses of a download.

    :code:`PENDING` - download is not yet started.

    :code:`STARTED` - download is already in progress.

    :code:`SUCCEEDED` - download finished successfully (with a download result).

    :code:`FAILED` - download failed. There might be no download result in such case.
    """

    PENDING = enum.auto()
    STARTED = enum.auto()
    SUCCEEDED = enum.auto()
    CANCELED = enum.auto()
    FAILED = enum.auto()


DownloadT = typing.TypeVar('DownloadT', bound='Download')


DownloadIdentity = typing.Tuple[str, str, typing.Any]


class Download(QtCore.QObject):  # pylint: disable=too-many-instance-attributes
    """A single download request."""

    sig_start = QtCore.Signal(object)
    sig_succeeded = QtCore.Signal(object)
    sig_canceled = QtCore.Signal(object)
    sig_failed = QtCore.Signal(object)
    sig_done = QtCore.Signal(object)

    def __init__(self, url: str) -> None:
        """Initialize new :class:`~Download` instance."""
        super().__init__()

        self._content: typing.Optional[bytes] = None
        self._data: typing.Any = None
        self._files: typing.Set[str] = set()
        self._keep_existing: Keep = Keep.NEVER
        self._medium: typing.Union[Medium, str] = Medium.RANDOM
        self._method: str = 'GET'
        self._process: typing.Optional[ProcessDownload] = None
        self._result: typing.Any = None
        self._status: Status = Status.PENDING
        self._status_code: int = 0
        self._url: str = url
        self._worker: workers.TaskWorker = self._run.worker()  # pylint: disable=no-member

        self._worker.signals.sig_start.connect(lambda: self._start())  # pylint: disable=unnecessary-lambda
        self._worker.signals.sig_succeeded.connect(lambda _: self._succeeded())
        self._worker.signals.sig_canceled.connect(lambda _: self._canceled())
        self._worker.signals.sig_failed.connect(lambda _: self._failed())
        self._worker.signals.sig_done.connect(lambda _: self._done())

    @property
    def _identity(self) -> DownloadIdentity:  # noqa: D401
        """
        Identity of the download source.

        This is used to detect requests to the same resources - such requests should have same
        :attr:`~Download._identity`.
        """
        return self._method, self._url, self._data

    @property
    def content(self) -> bytes:  # noqa: D401
        """
        Downloaded content.

        It is unnecessary to store the content in memory in order to use this property. Content might also be stored on
        a disk - it would be fetched from it automatically.

        In case you want to read a content from a disk directly - use :meth:`Download.read` instead.
        """
        if self._content is not None:
            return self._content

        stream: typing.BinaryIO
        with self.open('rb') as stream:
            return stream.read()

    @property
    def files(self) -> typing.Tuple[str, ...]:  # noqa: D401
        """List of files into which content was downloaded."""
        return tuple(self._files)

    @property
    def result(self) -> typing.Any:  # noqa: D401
        """
        Output of the download.

        This would be set only if download is successful, and :meth:`~Download.process` function provided. Otherwise -
        it is :code:`None`.
        """
        return self._result

    @property
    def status(self) -> Status:  # noqa: D401
        """
        Current status of a download.

        Refer to :class:`Status` for available options.
        """
        return self._status

    @property
    def status_code(self) -> int:  # noqa: D401
        """HTTP status code of the resource response."""
        return self._status_code

    @property
    def usable_file(self) -> str:  # noqa: D401
        """Any file from :attr:`~Download.files` that should be available."""
        return coalesce_source(self._files)

    # ------------------------------------------------------------------------------------------------------------------

    def attach(self: DownloadT, method: str, data: typing.Any = None) -> DownloadT:
        """
        Attach data to the resource request.

        It contains of a `method` and a `data` payload. E.g. :code:`.attach('POST', b'{"message": "test"}')`.

        This method returns the self instance, which is useful for chaining calls.
        """
        if self._status != Status.PENDING:
            raise ValueError('can not change properties after download is started')

        self._method = method
        self._data = data
        return self

    def into(self: DownloadT, *files: str) -> DownloadT:
        """
        Add `files` into which resource should be downloaded.

        This method returns the self instance, which is useful for chaining calls.
        """
        if self._status == Status.FAILED:
            raise ValueError('downloaded generated no output')

        if self._status == Status.SUCCEEDED:
            if self._content is not None:
                write(self._content, self._files.difference(files), overwrite=True)
            else:
                populate(coalesce_source(self._files), self._files.difference(files), overwrite=True)

        self._files.update(map(os.path.abspath, files))
        return self

    def keep(self: DownloadT, keep_existing: Keep) -> DownloadT:
        """
        Set how existing files should be treated.

        For available options, refer to :class:`~Keep`.

        This method returns the self instance, which is useful for chaining calls.
        """
        if self._status != Status.PENDING:
            raise ValueError('can not change properties after download is started')

        self._keep_existing = keep_existing
        return self

    def process(self: DownloadT, process: 'typing.Optional[ProcessDownload]') -> DownloadT:
        """
        Set a post-process action.

        It would populate :attr:`~Download.result` if request succeeds.

        This method returns the self instance, which is useful for chaining calls.
        """
        if self._status != Status.PENDING:
            raise ValueError('can not change properties after download is started')

        self._process = process
        return self

    def via(self: DownloadT, medium: typing.Union[Medium, str]) -> DownloadT:
        """
        Set a medium to use to store intermediate download results.

        It might be a path to any file or a particular :class:`Medium` value.

        This method returns the self instance, which is useful for chaining calls.
        """
        if self._status != Status.PENDING:
            raise ValueError('can not change properties after download is started')

        self._medium = medium
        return self

    # ------------------------------------------------------------------------------------------------------------------

    @typing.overload
    def open(
            self,
            mode: typing.Literal['r', 'rt'] = 'rt',
            encoding: typing.Optional[str] = None,
    ) -> typing.TextIO:
        """Open resource for reading."""

    @typing.overload
    def open(
            self,
            mode: typing.Literal['rb'],
    ) -> typing.BinaryIO:
        """Open resource for reading."""

    def open(self, mode='r', encoding=None):
        """Open resource for reading."""
        if 'b' not in mode:
            encoding = encoding or 'utf-8'

        return open(self.usable_file, mode, encoding=encoding)

    # ------------------------------------------------------------------------------------------------------------------

    def _canceled(self) -> None:
        """Process canceled download."""
        logs.logger.debug('download %016d: canceled', id(self))
        self._status = Status.CANCELED
        self.sig_canceled.emit(self)

    def _done(self) -> None:
        """Process completed download."""
        logs.logger.debug('download %016d: done', id(self))
        self.sig_done.emit(self)

    def _failed(self) -> None:
        """Process failed download."""
        logs.logger.debug('download %016d: failed', id(self))
        self._status = Status.FAILED
        self.sig_failed.emit(self)

    def _start(self) -> None:
        """Process started download."""
        logs.logger.debug('download %016d: started', id(self))
        self._status = Status.STARTED
        self.sig_start.emit(self)

    def _succeeded(self) -> None:
        """Process succeeded download."""
        logs.logger.debug('download %016d: succeeded', id(self))
        self._status = Status.SUCCEEDED
        self.sig_succeeded.emit(self)

    # ------------------------------------------------------------------------------------------------------------------

    @workers.Task
    def _run(self) -> None:  # pylint: disable=too-many-branches,too-many-statements
        """Perform download."""
        logs.logger.debug('download %016d: downloading %s', id(self), self._url)

        if inject_cancel():
            logs.logger.debug('download %016d: canceled due to offline mode', id(self))
            raise workers.TaskCanceledError('unable to fetch resources with offline mode enabled')

        source: typing.Optional[str] = None
        with contextlib.suppress(ValueError):
            source = coalesce_source(self._files)

        response: requests.Response
        if (source is not None) and (self._keep_existing is not Keep.NEVER):
            keep: bool = self._keep_existing is Keep.ALWAYS

            if (not keep) and (self._method == 'GET'):
                logs.logger.debug('download %016d: fetching HEAD', id(self))
                try:
                    response = requests.request('HEAD', self._url, **inject_kwargs())
                    logs.http_logger.http(response=response)
                    response.raise_for_status()
                except BaseException:  # pylint: disable=broad-except
                    logs.logger.debug('download %016d: failed to fetch HEAD', id(self), exc_info=True)
                else:
                    try:
                        keep = int(response.headers.get('Content-Length', -1)) == os.path.getsize(source)
                    except (OSError, TypeError, ValueError):
                        pass

            if keep:
                logs.logger.debug('download %016d: keeping existing content', id(self))
                populate(source, self._files, overwrite=False)
                self._finalize()
                return

        logs.logger.debug('download %016d: requesting a content', id(self))
        streaming: bool = bool(self._files) and (self._medium is not Medium.MEMORY)
        response = requests.request(self._method, self._url, data=self._data, stream=streaming, **inject_kwargs())
        logs.http_logger.http(response=response)
        self._status_code = response.status_code
        response.raise_for_status()

        if streaming:
            destination: str
            temporary: bool
            if self._medium is Medium.TARGET:
                destination = self.files[0]
                temporary = False
            elif self._medium is Medium.RANDOM:
                destination = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
                temporary = True
            else:
                destination = str(self._medium)
                temporary = destination not in self._files

            logs.logger.debug('download %016d: streaming content into %s', id(self), destination)

            os.makedirs(os.path.dirname(destination), exist_ok=True)
            response.raw.decode_content = True

            stream: typing.BinaryIO
            with open(destination, 'wb') as stream:
                shutil.copyfileobj(response.raw, stream)

            logs.logger.debug('download %016d: copying into additional files', id(self))

            populate(destination, self._files, overwrite=True)

            if temporary:
                with contextlib.suppress(OSError):
                    os.remove(destination)
        else:
            logs.logger.debug('download %016d: fetching content into memory', id(self))
            self._content = response.content

            logs.logger.debug('download %016d: writing content into files', id(self))
            write(self._content, self._files)

        self._finalize()

    def _finalize(self) -> None:
        """Apply post-processing actions to a download."""
        logs.logger.debug('download %016d: processing output', id(self))
        if self._process is not None:
            self._result = self._process(self)


GroupT = typing.TypeVar('GroupT', bound='Group')


class Group(QtCore.QObject):
    """
    Group of downloads.

    Allows managing multiple downloads simultaneously instead of each download separately.
    """

    sig_start = QtCore.Signal(object)
    sig_succeeded = QtCore.Signal(object)
    sig_canceled = QtCore.Signal(object)
    sig_failed = QtCore.Signal(object)
    sig_done = QtCore.Signal(object)

    def __init__(self, *downloads: Download) -> None:
        """Initialize new :class:`~Group` instance."""
        super().__init__()

        self._downloads: typing.Tuple[Download, ...] = downloads
        self._process: typing.Optional[ProcessGroup] = None
        self._require: Require = Require.ALL
        self._result: typing.Any = None
        self._stats: 'GroupStats' = {
            'done': set(),
            'canceled': set(),
            'failed': set(),
            'requested': set(downloads),
            'started': set(),
            'succeeded': set(),
        }
        self._status: Status = Status.PENDING

        download: Download
        for download in self._downloads:
            # pylint: disable=unnecessary-lambda
            download.sig_start.connect(lambda item: self._start(item))
            download.sig_succeeded.connect(lambda item: self._succeeded(item))
            download.sig_canceled.connect(lambda item: self._canceled(item))
            download.sig_failed.connect(lambda item: self._failed(item))
            download.sig_done.connect(lambda item: self._done(item))

            if download.status == Status.PENDING:
                continue
            self._start(download)
            if download.status == Status.STARTED:
                continue
            if download.status == Status.SUCCEEDED:
                self._succeeded(download)
            elif download.status == Status.CANCELED:
                self._canceled(download)
            elif download.status == Status.FAILED:
                self._failed(download)
            else:
                raise ValueError(f'unexpected download status: {download.status.name}')
            self._done(download)

    @property
    def downloads(self) -> typing.Tuple[Download, ...]:  # noqa: D401
        """Downloads in this group."""
        return self._downloads

    @property
    def result(self) -> typing.Any:  # noqa: D401
        """
        Output of the group.

        This would be set only if the whole group would be successful (see :meth:`~Group.require`), and
        :meth:`~Group.process` is provided. Otherwise - it is :code:`None`.
        """
        return self._result

    @property
    def status(self) -> Status:  # noqa: D401
        """
        Current status of a group download.

        Refer to :class:`Status` for available options.
        """
        return self._status

    # ------------------------------------------------------------------------------------------------------------------

    def require(self: GroupT, require: Require) -> GroupT:
        """
        Set a requirement to treat this group download a successful one.

        Refer to :class:`~Require` for available options.

        This method returns the self instance, which is useful for chaining calls.
        """
        self._require = require
        return self

    def process(self: GroupT, process: 'ProcessGroup') -> GroupT:
        """
        Set a post-process action.

        It would populate :attr:`~Group.result` if request succeeds.

        This method returns the self instance, which is useful for chaining calls.
        """
        self._process = process
        return self

    # ------------------------------------------------------------------------------------------------------------------

    def _canceled(self, download: Download) -> None:
        """Process canceled download."""
        if (not self._stats['canceled']) and (self._require is Require.ALL):
            self._status = Status.CANCELED
            self.sig_canceled.emit(self)
            self.sig_done.emit(self)

        self._stats['canceled'].add(download)

    def _done(self, download: Download) -> None:
        """Process finished download."""
        self._stats['done'].add(download)

        if (self._status is not Status.STARTED) or (self._stats['done'] != self._stats['requested']):
            return

        if (self._require is Require.NONE) or self._stats['succeeded']:
            self._finalize()
            self._status = Status.SUCCEEDED
            self.sig_succeeded.emit(self)
        elif self._stats['canceled']:
            self._status = Status.CANCELED
            self.sig_canceled.emit(self)
        else:
            self._status = Status.FAILED
            self.sig_failed.emit(self)
        self.sig_done.emit(self)

    def _failed(self, download: Download) -> None:
        """Process failed download."""
        if (not self._stats['failed']) and (self._require is Require.ALL):
            self._status = Status.FAILED
            self.sig_failed.emit(self)
            self.sig_done.emit(self)

        self._stats['failed'].add(download)

    def _start(self, download: Download) -> None:
        """Process started download."""
        if not self._stats['started']:
            self._status = Status.STARTED
            self.sig_start.emit(self)

        self._stats['started'].add(download)

    def _succeeded(self, download: Download) -> None:
        """Process succeeded download."""
        self._stats['succeeded'].add(download)

    # ------------------------------------------------------------------------------------------------------------------

    def _finalize(self) -> None:
        """Apply post-processing actions to a group."""
        if self._process is not None:
            self._result = self._process(self)


class ManagerItem(QtCore.QObject):
    """Internal collection item, that allows sharing download results between :class:`~Download` instances."""

    def __init__(self, download: Download) -> None:
        """Initialize new :class:`~ManagerItem` instance."""
        super().__init__()

        self._download: Download = download
        self._children: typing.List[Download] = []

        self._download.sig_done.connect(lambda _: self._done())

    @property
    def identity(self) -> DownloadIdentity:  # noqa: D401
        """
        Identity of the download source.

        Used for detecting downloads with the same sources.
        """
        return self._download._identity  # pylint: disable=protected-access

    def attach(self, download: Download) -> None:
        """Register another `download` which should use results of the current one."""
        logs.logger.debug('download %016d: attaching download to %016d', id(download), id(self._download))
        self._children.append(download)

        if self._download.status in (Status.SUCCEEDED, Status.CANCELED, Status.FAILED):
            self._done()

    def _apply_to(self, download: Download) -> None:
        """Apply results of the internal :class:`~Download` onto a `download`."""
        # pylint: disable=protected-access

        logs.logger.debug('download %016d: copying results from %016d', id(download), id(self._download))
        if download._status != Status.PENDING:  # pylint: disable=protected-access
            raise ValueError('download should not be started to fetch previous results')

        download._start()

        download._status_code = self._download._status_code

        if self._download._status is Status.SUCCEEDED:
            logs.logger.debug('download %016d: applying content', id(download))
            if (download._medium is Medium.MEMORY) or (not download._files):
                download._content = self._download.content

            try:
                self._download.into(*download._files)
            except ValueError:
                download._failed()
            else:
                download._finalize()
                download._succeeded()

        elif self._download._status is Status.CANCELED:
            download._canceled()

        elif self._download._status is Status.FAILED:
            download._failed()

        else:
            raise ValueError(f'unexpected status of the managed download: {self._download._status}')

        download._done()

    def _done(self) -> None:
        """Process the finish of the wrapped :class:`~Download`."""
        while self._children:
            self._apply_to(self._children.pop())


class Manager(QtCore.QObject):  # pylint: disable=too-few-public-methods
    """Central download manager."""

    def __init__(self) -> None:
        """Initialize new :class:`~Manager` instance."""
        super().__init__()

        self.__history: typing.Dict[DownloadIdentity, ManagerItem] = {}

    def execute(self, *items: typing.Union[Download, Group]) -> None:
        """Start downloads."""
        queue: typing.List[Download] = []

        item: typing.Union[Download, Group]
        for item in items:
            if isinstance(item, Download):
                queue.append(item)
            else:
                queue.extend(item.downloads)

        download: Download
        for download in queue:
            identity: DownloadIdentity = download._identity  # pylint: disable=protected-access
            try:
                parent: ManagerItem = self.__history[identity]
            except KeyError:
                self.__history[identity] = ManagerItem(download)
                download._worker.start()  # pylint: disable=protected-access
            else:
                parent.attach(download)


MANAGER: typing.Optional[Manager] = None


def manager() -> Manager:
    """Prepare singleton instance of the :class:`~Manager`."""
    global MANAGER  # pylint: disable=global-statement
    if MANAGER is None:
        MANAGER = Manager()
    return MANAGER
