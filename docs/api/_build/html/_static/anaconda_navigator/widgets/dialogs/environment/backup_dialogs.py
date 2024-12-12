# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Dialogs for backing up environments."""

__all__ = ['BackupSelectorDialog']

import html
import os
import tempfile
import typing
import uuid

import requests
from qtpy import QtWidgets
from qtpy import compat

from anaconda_navigator.api import cloud
from anaconda_navigator.api.cloud import tools as cloud_tools
from anaconda_navigator.utils.logs.loggers import http_logger as logger
from anaconda_navigator.utils import telemetry
from anaconda_navigator.utils import workers
from . import multiaccount_dialogs

if typing.TYPE_CHECKING:
    from anaconda_navigator.api.cloud.tools import error_parsers as cloud_error_parsers
    from anaconda_navigator.widgets import main_window


DEFAULT_ERROR: typing.Final[str] = 'Unable to backup environment due to an unknown error.'


class BackupSelectorDialog(multiaccount_dialogs.PrepopulatedSelectorDialog):  # pylint: disable=too-few-public-methods
    """Dialog for selecting the target to which to export an environment."""

    def __init__(self, parent: 'main_window.MainWindow') -> None:
        """Initialize new :class:`~BackupSelectorDialog` instance."""
        super().__init__(parent=parent)

        self.setWindowTitle('Backup Environment')

    def __init_header__(  # pylint: disable=useless-super-delegation
            self,
            layout: QtWidgets.QVBoxLayout,
            *,
            caption_text: str = 'Select location to backup environment:',
    ) -> None:
        """Initialize header part of the dialog."""
        super().__init_header__(layout, caption_text=caption_text)

    def __init_footer__(
            self,
            layout: QtWidgets.QVBoxLayout,
            *,
            caption_text: str = 'Backup as:',
    ) -> None:
        """Initialize footer part of the dialog."""
        super().__init_footer__(layout, caption_text=caption_text)
        self._controls.account_cloud += self._controls.all[-3:]

    def __init_actions__(  # pylint: disable=useless-super-delegation
            self,
            layout: QtWidgets.QVBoxLayout,
            *,
            accept_text: str = 'Backup',
            reject_text: str = 'Cancel',
    ) -> None:
        """Initialize actions part of the dialog."""
        super().__init_actions__(layout, accept_text=accept_text, reject_text=reject_text)

    def _process_accept(self) -> None:
        """Process clicking on the 'OK' button."""
        self.set_busy(True)
        self.clear_heading_errors()
        self.footer_error = ''  # pylint: disable=attribute-defined-outside-init

        if self.selection == multiaccount_dialogs.SelectorValue.LOCAL:
            self.__process_accept_local()
        elif self.selection == multiaccount_dialogs.SelectorValue.CLOUD:
            self.__process_accept_cloud()
        else:
            raise NotImplementedError()

    def _process_selection(self, value: multiaccount_dialogs.SelectorValue) -> None:
        """Process changing selected value in the dialog."""
        super()._process_selection(value)
        self.__update_acceptable()

    def _process_environment_name(self, value: str) -> None:
        """Process change of the `environment_name` value."""
        super()._process_environment_name(value)
        self.__update_acceptable()

    def __update_acceptable(self) -> None:
        """Update state of the accept button according to current dialog state."""
        if self.selection == multiaccount_dialogs.SelectorValue.LOCAL:
            self.set_acceptable(True)
            return

        if (self.selection == multiaccount_dialogs.SelectorValue.CLOUD) and self.environment_name:
            self.set_acceptable(True)
            return

        self.set_acceptable(False)

    # Local

    def __process_accept_local(self) -> None:
        """Process clicking on the 'OK' button when local account is selected."""
        telemetry.ANALYTICS.instance.event('request-environment-backup', {'to': 'local'})

        filter_yaml: typing.Final[str] = 'Conda environment files (*.yaml *.yml)'
        filter_any: typing.Final[str] = 'All files (*)'

        path: str
        selected_filter: str
        path, selected_filter = compat.getsavefilename(
            parent=self,
            caption='Backup Environment',
            basedir=os.path.expanduser('~'),
            filters=';;'.join([filter_yaml, filter_any]),
        )
        if not path:
            self.set_busy(False)
            return
        if (not os.path.splitext(path)[1]) and (selected_filter == filter_yaml):
            path += '.yaml'

        worker = self.parent().api.export_environment(
            prefix=self.parent().current_prefix,
            file=os.path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.yml'),
        )
        worker.requested_file = path
        worker.sig_finished.connect(self.__check_local_export)

    def __check_local_export(
            self, worker: typing.Any, output: typing.Any, error: typing.Any,  # pylint: disable=unused-argument
    ) -> None:
        """Check result of :code:`conda env export` command."""
        if not self.isVisible():
            try:
                os.remove(worker.file)
            except OSError:
                pass
            return

        if not os.path.isfile(worker.file):
            self.set_busy(False)
            self.add_heading_error(multiaccount_dialogs.HEADING_ERROR_TEMPLATE.format(
                content='Unable to create requested file.',
            ))
            return

        new_worker: workers.TaskWorker = cloud_tools.clear_environment.worker(
            source_file=worker.file,
            target_file=worker.requested_file,
            name=worker.name,
        )
        self.finished.connect(new_worker.cancel)
        new_worker.signals.sig_done.connect(lambda result: self.finished.disconnect(new_worker.cancel))
        new_worker.signals.sig_done.connect(self.__check_local_inject)
        new_worker.start()

    def __check_local_inject(self, result: workers.TaskResult) -> None:
        """"""
        try:
            os.remove(result.call.kwargs['source_file'])
        except OSError:
            pass

        if result.status == workers.TaskStatus.SUCCEEDED:
            telemetry.ANALYTICS.instance.event('backup-environment', {'to': 'local'})
            super()._process_accept()
            return

        try:
            os.remove(result.call.kwargs['target_file'])
        except OSError:
            pass

        if result.status == workers.TaskStatus.FAILED:
            self.add_heading_error(multiaccount_dialogs.HEADING_ERROR_TEMPLATE.format(
                content='Unable to create requested file.',
            ))

        self.set_busy(False)

    # Cloud

    def __process_accept_cloud(self) -> None:
        """Process clicking on the 'OK' button when Cloud account is selected."""
        telemetry.ANALYTICS.instance.event('request-environment-backup', {'to': 'cloud'})
        worker = self.parent().api.export_environment(
            prefix=self.parent().current_prefix,
            file=os.path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.yml'),
        )
        worker.sig_finished.connect(self.__check_cloud_export)

    def __check_cloud_export(
            self, worker: typing.Any, output: typing.Any, error: typing.Any,  # pylint: disable=unused-argument
    ) -> None:
        """Check result of :code:`conda env export` command."""
        if not self.isVisible():
            try:
                os.remove(worker.file)
            except OSError:
                pass
            return

        if not os.path.isfile(worker.file):
            self.set_busy(False)
            self.add_heading_error(multiaccount_dialogs.HEADING_ERROR_TEMPLATE.format(
                content='System error creating a backup.',
            ))
            return

        new_worker: workers.TaskWorker = cloud_tools.clear_environment.worker(
            source_file=worker.file,
            target_file=os.path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.yml'),
            name=worker.name,
        )
        self.finished.connect(new_worker.cancel)
        new_worker.signals.sig_done.connect(lambda result: self.finished.disconnect(new_worker.cancel))
        new_worker.signals.sig_done.connect(self.__check_cloud_inject)
        new_worker.start()

    def __check_cloud_inject(self, result: workers.TaskResult) -> None:
        """"""
        try:
            os.remove(result.call.kwargs['source_file'])
        except OSError:
            pass

        if result.status == workers.TaskStatus.SUCCEEDED:
            new_worker: workers.TaskWorker
            new_worker = cloud.CloudAPI().create_environment.worker(  # pylint: disable=no-member
                name=self.environment_name,
                path=result.call.kwargs['target_file'],
            )
            self.finished.connect(new_worker.cancel)
            new_worker.signals.sig_done.connect(lambda result: self.finished.disconnect(new_worker.cancel))
            new_worker.signals.sig_done.connect(self.__check_cloud_create)
            new_worker.start()
            return

        try:
            os.remove(result.call.kwargs['target_file'])
        except OSError:
            pass

        if result.status == workers.TaskStatus.FAILED:
            self.add_heading_error(multiaccount_dialogs.HEADING_ERROR_TEMPLATE.format(
                content='Unable to create requested file.',
            ))

        self.set_busy(False)

    def __check_cloud_create(self, result: workers.TaskResult) -> None:
        """Check Cloud response after creating a new environment."""
        if self.isVisible() and (result.status == workers.TaskStatus.FAILED):
            handlers: cloud_tools.HttpErrorHandlers = cloud_tools.HttpErrorHandlers()
            handlers.register_handler(BaseException, self._handle_header_error(DEFAULT_ERROR))
            handlers.register_http_handler(
                self._handle_environment_already_exists(path=result.call.kwargs['path']),
                409,
                'environment_already_exists',
            )
            handlers.register_http_handler(self._handle_unprocessable_entry(), 422)

            if handlers.handle(exception=typing.cast(BaseException, result.exception)):
                return

            self.set_busy(False)

        try:
            os.remove(result.call.kwargs['path'])
        except OSError:
            pass

        if self.isVisible() and (result.status == workers.TaskStatus.SUCCEEDED):
            telemetry.ANALYTICS.instance.event('backup-environment', {'to': 'cloud'})
            super()._process_accept()

    def __check_cloud_update(self, result: workers.TaskResult) -> None:
        """Check Cloud response after updating existing environment."""
        try:
            os.remove(result.call.kwargs['path'])
        except OSError:
            pass

        if not self.isVisible():
            return

        if result.status == workers.TaskStatus.SUCCEEDED:
            super()._process_accept()

        elif result.status == workers.TaskStatus.FAILED:
            handlers: cloud_tools.HttpErrorHandlers = cloud_tools.HttpErrorHandlers()
            handlers.register_handler(BaseException, self._handle_header_error(DEFAULT_ERROR))
            handlers.handle(exception=typing.cast(BaseException, result.exception))

        self.set_busy(False)

    # handlers

    def _handle_environment_already_exists(
            self,
            path: str,
    ) -> 'cloud_error_parsers.Handler[requests.RequestException]':
        """
        Handle conflict on creating a new environment in Cloud.

        If `environment_override` is set - may launch update of existing Cloud environment.
        """
        def result(exception: BaseException) -> bool:  # pylint: disable=unused-argument
            if self.environment_override:
                worker: workers.TaskWorker  # pylint: disable=no-member
                worker = cloud.CloudAPI().update_environment.worker(  # pylint: disable=no-member
                    name=self.environment_name,
                    path=path,
                )
                self.finished.connect(worker.cancel)
                worker.signals.sig_done.connect(lambda result: self.finished.disconnect(worker.cancel))
                worker.signals.sig_done.connect(self.__check_cloud_update)
                worker.start()
                return True

            template: str = multiaccount_dialogs.FOOTER_ERROR_TEMPLATE
            self.footer_error = template.format(  # pylint: disable=attribute-defined-outside-init
                content='Please rename environment to continue.',
            )
            return False

        return result

    def _handle_unprocessable_entry(self) -> 'cloud_error_parsers.Handler[requests.RequestException]':
        """Handle 422 HTTP error."""
        def result(exception: requests.RequestException) -> bool:
            items: typing.List[typing.Any] = []
            if exception.response is not None:
                try:
                    logger.http(response=exception.response)
                    items.extend(exception.response.json()['detail'])
                except (ValueError, TypeError, KeyError):
                    pass

            pending: bool = True
            for item in items:
                try:
                    if item['loc'] == ['body', 'name']:
                        template: str = multiaccount_dialogs.FOOTER_ERROR_TEMPLATE
                        self.footer_error = template.format(  # pylint: disable=attribute-defined-outside-init
                            content=html.escape(item['msg']),
                        )
                        pending = False
                except (TypeError, KeyError):
                    pass

            if pending:
                self.add_heading_error(multiaccount_dialogs.HEADING_ERROR_TEMPLATE.format(content=DEFAULT_ERROR))
            return False

        return result
