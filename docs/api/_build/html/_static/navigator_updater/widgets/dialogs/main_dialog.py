# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright 2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Anaconda Navigator Updater main dialog."""

# Standard library imports
import os

# Third party imports
from qtpy.QtCore import QSize, Qt, QTimer, Signal
from qtpy.QtSvg import QSvgWidget
from qtpy.QtWidgets import QHBoxLayout, QProgressBar, QVBoxLayout

# Local imports
from navigator_updater.api.conda_api import CondaAPI
from navigator_updater.api import utils as api_utils
from navigator_updater.config import MAC, NAVIGATOR_LOCKFILE, WIN
from navigator_updater.external import filelock
from navigator_updater.static import images
from navigator_updater.utils import launch as launch_utils
from navigator_updater.utils.misc import set_windows_appusermodelid
from navigator_updater.utils.styles import load_style_sheet
from navigator_updater.utils import version_utils
from navigator_updater.widgets import ButtonNormal, ButtonPrimary, LabelBase, SpacerHorizontal, SpacerVertical
from navigator_updater.widgets.dialogs import DialogBase, MessageBoxQuestion

# yapf: enable


class MainDialog(DialogBase):  # pylint: disable=too-many-instance-attributes
    """Main dialog for the anaconda navgator updater."""
    # Signals
    sig_application_updated = Signal()
    sig_ready = Signal()

    # Class variables
    PACKAGE = 'anaconda-navigator'  # pylint: disable=invalid-name
    WIDTH = 450  # pylint: disable=invalid-name
    HEIGHT = 200  # pylint: disable=invalid-name

    def __init__(self, latest_version=None, prefix=None):  # pylint: disable=too-many-statements
        """Main dialog for the anaconda navgator updater."""
        super().__init__()

        # Variables
        self.api = CondaAPI()
        self.prefix = prefix or os.environ.get(
            'CONDA_PREFIX', self.api.ROOT_PREFIX
        )
        self.is_root_writable = False
        self.info = {}
        self.first_run = True
        self.setup_ready = False
        self.busy = False
        self.up_to_date = False
        self.error = False
        self.success = False
        self.status = ''
        self.current_version = None
        self.latest_version = latest_version
        self.style_sheet = load_style_sheet()
        self.timer = QTimer()
        self.timer_2 = QTimer()
        self._windows_appusermodelid = None

        # Widgets
        self.message_box = None  # For testing
        self.label_icon = QSvgWidget()
        self.label_message = LabelBase(
            'There`s a new version of Anaconda Navigator available. We strongly recommend you to update.'
        )

        self.label_status = LabelBase('')
        self.progress_bar = QProgressBar()
        self.button_cancel = ButtonNormal('Dismiss')
        self.button_update = ButtonPrimary('Update now')
        self.button_launch = ButtonPrimary('Launch Navigator')

        # Widgets setup
        if WIN:
            self._windows_appusermodelid = set_windows_appusermodelid()

        self.setMinimumSize(self.WIDTH, self.HEIGHT)
        self.label_message.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label_message.setWordWrap(True)
        self.label_status.setWordWrap(True)
        self.button_update.setAutoDefault(True)
        self.button_launch.setAutoDefault(True)
        self.button_cancel.setFocusPolicy(Qt.NoFocus)
        self.timer.setInterval(1000)
        self.timer_2.setInterval(5000)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(self.style_sheet)
        self.label_icon.load(images.ANACONDA_LOGO)
        self.label_icon.setMaximumSize(QSize(64, 64))
        self.label_icon.setMinimumSize(QSize(64, 64))
        self.setWindowTitle('Anaconda Navigator Updater')
        self.progress_bar.setMaximumWidth(self.WIDTH // 3)
        self.setMinimumWidth(self.WIDTH)
        self.setMaximumWidth(self.WIDTH)
        self.setMinimumHeight(self.HEIGHT)

        # Layouts
        layout_status = QHBoxLayout()
        layout_status.addWidget(self.label_status)
        layout_status.addWidget(SpacerHorizontal())
        layout_status.addWidget(self.progress_bar)

        layout_text = QVBoxLayout()
        layout_text.addWidget(self.label_message)
        layout_text.addStretch()
        layout_text.addWidget(SpacerVertical())
        layout_text.addLayout(layout_status)

        layout_icon = QVBoxLayout()
        layout_icon.addWidget(self.label_icon)
        layout_icon.addStretch()

        layout_top = QHBoxLayout()
        layout_top.addLayout(layout_icon)
        layout_top.addWidget(SpacerHorizontal())
        layout_top.addLayout(layout_text)

        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_cancel)
        layout_buttons.addWidget(SpacerHorizontal())
        layout_buttons.addWidget(self.button_update)
        layout_buttons.addWidget(self.button_launch)

        layout = QVBoxLayout()
        layout.addLayout(layout_top)
        layout.addWidget(SpacerVertical())
        layout.addWidget(SpacerVertical())
        layout.addStretch()
        layout.addLayout(layout_buttons)

        self.setLayout(layout)

        # Signals
        self.button_update.clicked.connect(self.install_update)
        self.button_cancel.clicked.connect(self.reject)
        self.button_launch.clicked.connect(self.launch)
        self.timer.timeout.connect(self.refresh)
        self.timer_2.timeout.connect(self.check_conditions)

        # Setup
        self.timer.start()
        self.timer_2.start()
        self.check_conditions()
        self.refresh()

    def closeEvent(self, event):  # pylint: disable=invalid-name,unused-argument
        """Catch close event."""
        self.timer.stop()
        self.timer_2.stop()

    def check_conditions(self):
        """Check every 5 seconds installed packages in case conda was used."""
        packages = self.api.linked(prefix=self.prefix)
        package = [package for package in packages if self.PACKAGE in package]
        if package:
            _, current_version, _ = api_utils.split_canonical_name(package[0])
            self.current_version = current_version
        else:
            self.current_version = None

        if self.latest_version is None:
            worker_search = self.api.search(
                self.PACKAGE, platform=self.api.get_platform()
            )
            worker_search.sig_finished.connect(self._search_callback)
        else:
            worker = self.api.info()
            worker.sig_finished.connect(self.setup)
            self.check_versions()

    def check_versions(self):
        """Check if navigator is up-to-date."""
        self.up_to_date = (not self.latest_version) or bool(
            self.current_version and (version_utils.compare(self.current_version, self.latest_version) >= 0)
        )

    def _search_callback(self, worker, output, error):
        """Setup the widget."""
        if isinstance(output, dict):
            packages = output.get(self.PACKAGE, [])
            versions = [package.get('version') for package in packages]
            unique_versions = []
            for version in versions:
                if version not in unique_versions:
                    unique_versions.append(version)
            if unique_versions:
                self.latest_version = unique_versions[-1]

        self.check_versions()
        worker = self.api.info()
        worker.sig_finished.connect(self.setup)

        self.refresh()

    def setup(self, worker, info, error):  # pylint: disable=unused-argument
        """Setup the widget."""
        self.info = info
        self.is_root_writable = info.get('root_writable', False)
        self.setup_ready = True
        self.sig_ready.emit()
        self.refresh()

        if self.button_update.isVisible():
            self.button_update.setFocus()

        if self.button_launch.isVisible():
            self.button_launch.setFocus()

    def update_style_sheet(self, style_sheet=None):
        """Update custom CSS style sheet."""
        self.style_sheet = load_style_sheet()
        self.setStyleSheet(self.style_sheet)

    def refresh(self):  # pylint: disable=too-many-branches,too-many-statements
        """Refresh enabled/disabled status of widgets."""
        current_version = 'Not installed'
        if self.current_version:
            current_version = self.current_version

        latest_version = '-'
        if self.latest_version:
            latest_version = self.latest_version

        main_message = (
            f'Current version:    &nbsp;&nbsp;&nbsp;&nbsp;<i>{current_version}</i><br>'
            f'Available version: &nbsp;&nbsp;<b>{latest_version}</b><br>'
        )

        message = self.status
        running = self.check_running()
        self.button_launch.setVisible(False)

        if not self.setup_ready:
            self.button_update.setDisabled(True)
            self.progress_bar.setVisible(True)
            message = 'Updating index...'
            self.update_status(message)
        elif self.busy:
            self.button_update.setDisabled(True)
            self.progress_bar.setVisible(True)
        else:
            self.progress_bar.setVisible(False)

            if running:
                message = 'Please close Anaconda Navigator before updating.'
                self.button_update.setDisabled(running)
            elif not running:
                self.button_update.setDisabled(False)
                if self.success and self.current_version:
                    message = 'Anaconda Navigator was updated successfully.'
                    self.button_update.setVisible(False)
                    self.button_launch.setVisible(True)
                elif self.up_to_date:
                    message = 'Anaconda Navigator is already up to date.'
                    self.button_update.setVisible(False)
                    self.button_launch.setVisible(True)
                elif not self.error:
                    self.button_update.setVisible(True)
                    if self.current_version:
                        message = 'An update for Anaconda Navigator is now available.'
                        self.button_update.setText('Update now')
                    else:
                        message = (
                            'Anaconda Navigator is available for install.'
                        )
                        self.button_update.setText('Install now')

                if not self.is_root_writable and WIN:
                    self.button_update.setDisabled(True)
                    message = 'Need to run with elevated privileges'

            if self.error:
                self.button_update.setDisabled(False)
                message = 'Cannot update Anaconda Navigator, <b>{0}</b>'
                message = message.format(self.error)

        self.label_status.setText(message)
        self.label_message.setText(main_message)

    def update_status(self, status='', value=-1, max_val=-1):
        """Update progress bar and message status."""
        if status:
            self.status = status
            self.label_status.setText(status)
            if value < 0 and max_val < 0:
                self.progress_bar.setRange(0, 0)
            else:
                self.progress_bar.setMinimum(0)
                self.progress_bar.setMaximum(max_val)
                self.progress_bar.setValue(int(value))

    def check_running(self):
        """Check if Anaconda Navigator is running."""
        # Create file lock
        lock = filelock.FileLock(NAVIGATOR_LOCKFILE)
        try:
            running = False
            with lock.acquire(timeout=0.01):
                pass
        except filelock.Timeout:
            running = True
        return running

    # --- Conda actions and helpers
    # -------------------------------------------------------------------------
    def partial_output_ready(self, worker, output, error):  # pylint: disable=unused-argument
        """Handle conda partial output ready."""
        self.busy = True
        # print(type(output))
        # print(output)

        # Get errors and data from ouput if it exists
        fetch = None
        if output and isinstance(output, dict):
            fetch = output.get('fetch')
            max_val = output.get('maxval', -1)
            value = output.get('progress', -1)

        if fetch:
            status = f'Fetching <b>{fetch}</b>...'
            self.update_status(status=status, max_val=max_val, value=value)

    def output_ready(self, worker, output, error):  # pylint: disable=unused-argument
        """Handle conda output ready."""
        self.check_conditions()

        # Get errors and data from ouput if it exists
        error_text = output.get('error', '')
        exception_type = output.get('exception_type', '')
        exception_name = output.get('exception_name', '')
        success = output.get('success')
        actions = output.get('actions', {})
        # op_order = output.get('op_order', [])
        # action_check_fetch = actions.get('CHECK_FETCH', [])
        # action_rm_fetch = actions.get('RM_FETCHED', [])
        # action_fetch = actions.get('FETCH', [])
        # action_check_extract = actions.get('CHECK_EXTRACT', [])
        # action_rm_extract = actions.get('RM_EXTRACTED', [])
        # action_extract = actions.get('EXTRACT', [])
        # action_unlink = actions.get('UNLINK', [])
        action_link = actions.get('LINK', [])
        # action_symlink_conda = actions.get('SYMLINK_CONDA', [])

        self.busy = False

        # Get errors from json output
        if error_text or exception_type or exception_name or (not success):
            self.error = exception_name
            self.success = False
            self.up_to_date = False
        elif success and action_link:
            self.sig_application_updated.emit()
            self.error = None
            self.success = True
            self.up_to_date = False
        elif success:
            self.success = False
            self.error = None
            self.up_to_date = True

        worker.lock.release()
        self.refresh()

    def install_update(self):
        """Install the specified version or latest version of navigator."""
        self.busy = True
        self.refresh()
        # conda_prefix = self.info.et('conda_prefix')
        # root_prefix = self.info.et('root_prefix')
        navigator_prefixes = [
            # os.path.join(self.api.ROOT_PREFIX, 'envs', '_navigator_'),
            # os.path.join(self.api.ROOT_PREFIX, 'envs', '_conda_'),
            self.prefix,
        ]
        for prefix in navigator_prefixes:
            if self.api.environment_exists(prefix=prefix):
                break

        if self.latest_version:
            pkgs = [f'{self.PACKAGE}=={self.latest_version}']
        else:
            pkgs = [self.PACKAGE.format(self.latest_version)]

        # Lock Navigator
        lock = filelock.FileLock(NAVIGATOR_LOCKFILE)
        lock.acquire()
        worker = self.api.install(prefix=prefix, pkgs=pkgs)
        worker.lock = lock
        worker.sig_partial.connect(self.partial_output_ready)
        worker.sig_finished.connect(self.output_ready)
        self.refresh()

        if self.prefix == self.api.ROOT_PREFIX:
            name = 'root'
        else:
            name = os.path.basename(self.prefix)

        self.button_launch.setFocus()
        if self.current_version:
            msg = f'Updating package on <b>{name}</b>...'
        else:
            msg = f'Installing package on <b>{name}</b>...'
        self.update_status(msg)

    def launch(self):
        """Launch Anaconda Navigator."""
        leave_path_alone = True
        prefix = self.prefix
        command = 'anaconda-navigator'

        # Use the app bundle on OSX
        if MAC:
            command = f'open \'{os.path.join(prefix, "Anaconda-Navigator.app")}\''

        launch_utils.launch(
            root_prefix=self.api.ROOT_PREFIX,
            prefix=prefix,
            command=command,
            package_name='anaconda-navigator-app',
            leave_path_alone=leave_path_alone,
        )
        self.close()

    # --- Qt Overrides
    # -------------------------------------------------------------------------
    def reject(self):
        """Override Qt method."""
        if self.busy:
            msg_box = MessageBoxQuestion(
                title='Quit Navigator Updater?',
                text='Anaconda Navigator is being '
                'updated. <br><br>'
                'Are you sure you want to quit?'
            )

            if msg_box.exec_():
                super().reject()
        else:
            super().reject()
