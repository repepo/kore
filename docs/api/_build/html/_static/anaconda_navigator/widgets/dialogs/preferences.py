# -*- coding: utf-8 -*-

# pylint: disable=invalid-name,missing-function-docstring,no-name-in-module,unused-argument

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Preferences dialog."""

from configparser import ConfigParser
from copy import deepcopy
import io
import json
import os
import sys
import typing
from qtpy import QtCore
from qtpy.QtCore import QPoint, Qt, Signal
from qtpy.QtGui import QCursor, QPixmap
from qtpy.QtWidgets import (
    QCheckBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QScrollArea, QTextEdit, QVBoxLayout, QWidget,
)
import yaml
from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.api.conda_api import CondaAPI
from anaconda_navigator.api import download_api
from anaconda_navigator.api.external_apps import get_applications
from anaconda_navigator.config import CONF, WIN, WIN7
from anaconda_navigator.static.images import INFO_ICON, WARNING_ICON
from anaconda_navigator.utils import telemetry
from anaconda_navigator.utils import url_utils
from anaconda_navigator.widgets import ButtonNormal, ButtonPrimary, ComboBoxBase, SpacerHorizontal, SpacerVertical
from anaconda_navigator.widgets.dialogs import DialogBase, MessageBoxError
from anaconda_navigator.widgets.dialogs.offline import DialogOfflineMode
from anaconda_navigator.widgets.dialogs import login as login_dialogs


CONDARC_DEFAULT = {
    'always_yes': True,
    'channels': ['defaults'],
    'ssl_verify': True
}


class SettingsDialog(DialogBase):  # pylint: disable=missing-class-docstring
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.text_edit = QTextEdit()
        self.text_edit.setFixedSize(600, 600)

        self.button_reset = ButtonNormal('Reset to defaults')
        self.button_reset.clicked.connect(self.reset_to_defaults)

        self.button_cancel = ButtonNormal('Cancel')
        self.button_cancel.clicked.connect(self._cancel)

        self.button_save = ButtonPrimary('Save and Restart')

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.button_reset)
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.button_cancel)
        self.buttons_layout.addWidget(SpacerHorizontal())
        self.buttons_layout.addWidget(self.button_save)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.text_edit)
        self.main_layout.addWidget(SpacerVertical())
        self.main_layout.addLayout(self.buttons_layout)

        self.setLayout(self.main_layout)

    def reset_to_defaults(self):
        raise NotImplementedError()

    def _cancel(self, e):
        self.close()

    @staticmethod
    def _restart_application():
        QtCore.QCoreApplication.quit()

        if WIN:
            QtCore.QProcess.startDetached(f'"{sys.argv[0]}"')
        else:
            QtCore.QProcess.startDetached(sys.executable, sys.argv)


class NavigatorSettingsDialog(SettingsDialog):  # pylint: disable=missing-class-docstring
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('Navigator settings (anaconda-navigator.ini)')
        self.file_path = None
        self.config = config

        self.button_save.clicked.connect(self._save)

    def reset_to_defaults(self):
        # This hack done to have the ability to get all values
        # and update them appropriately but not touch the original
        # config, because on some widgets there are timers to write
        # data to config in some interval.
        config = deepcopy(self.config)

        self.text_edit.setText(config.get_defaults())

    def setup(self, file_path):
        if os.path.exists(file_path):
            self.file_path = file_path
            with open(file_path, 'r') as file:  # pylint: disable=unspecified-encoding
                self.text_edit.setText(file.read())

    def _validate_config(self):
        """
        Validates the config data to not miss required fields
        in the config (anaconda-navigator.ini) file.
        Returns the dictionary with missing fields and sections.

        :return dict:
        """
        config_text = self.text_edit.toPlainText()

        buffer = io.StringIO(config_text)
        new_config = ConfigParser()
        new_config.read_file(buffer)

        missing_data = {}

        for section in CONF.sections():
            if not new_config.has_section(section):
                missing_data[section] = []

            for option in CONF.options(section):
                if not new_config.has_option(section, option):
                    missing_data.setdefault(section, []).append(option)

        return missing_data

    def _save(self, e):
        """
        Saves the data to the config (anaconda-navigator.ini) file if
        no missing data. Otherwise popup with missing data will arise.
        """
        missing_data = self._validate_config()

        if missing_data:
            text = (
                'The saved data missed some sections or attributes and could '  # pylint: disable=implicit-str-concat
                'cause the issues with working Navigator! Please fix.'
            )
            msg_box = MessageBoxError(
                title='Navigator Settings Save Error',
                text=text,
                error=json.dumps(missing_data, indent=4),
                report=False,
                json=True
            )
            msg_box.exec_()
        else:
            if self.file_path and os.path.exists(self.file_path):
                with open(self.file_path, 'w') as file:  # pylint: disable=unspecified-encoding
                    file.write(self.text_edit.toPlainText())
                    self._restart_application()


class CondaSettingsDialog(SettingsDialog):  # pylint: disable=missing-class-docstring
    YAML_DOCS_URL = 'https://en.wikipedia.org/wiki/YAML'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conda_api = CondaAPI()

        self.setWindowTitle('Conda settings (.condarc)')
        self.button_save.clicked.connect(self._save)

    def reset_to_defaults(self):
        self.text_edit.setText(yaml.dump(CONDARC_DEFAULT))

    def setup(self):
        self.text_edit.setText(self.conda_api.load_rc_plain())

    def _save(self, e):
        """
        Saves the data to config (.condarc) file if the data
        is a valid YAML format data. Otherwise the popup with
        error message will arise.
        """
        text = self.text_edit.toPlainText()

        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exception:
            msg_box = MessageBoxError(
                title='Conda Settings Save Error',
                text='The saved data is not a valid yaml config! Please fix.',
                error=exception,
                report=False,
                learn_more=self.YAML_DOCS_URL
            )
            msg_box.exec_()
        else:
            self.conda_api.save_rc_plain(text)
            self._restart_application()


class PreferencesDialog(DialogBase):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Application preferences dialog."""

    sig_urls_updated = Signal(str, str)
    sig_check_ready = Signal()
    sig_reset_ready = Signal()

    def __init__(self, config=CONF, environments=None, **kwargs):  # pylint: disable=too-many-statements
        """Application preferences dialog."""
        super().__init__(**kwargs)

        self.api = AnacondaAPI()
        self.widgets_changed = set()
        self.widgets = []
        self.widgets_dic = {}
        self.config = config
        self.environments = environments

        # Widgets
        self.button_ok = ButtonPrimary('Apply')
        self.button_cancel = ButtonNormal('Cancel')
        self.button_reset = ButtonNormal('Reset to defaults')
        self.button_nav_settings = ButtonPrimary('Configure Navigator')
        self.button_conda_settings = ButtonPrimary('Configure Conda')
        self.row = 0

        self.setFixedWidth(615)
        self.setFixedHeight(600)

        # Widget setup
        self.setWindowTitle('Preferences')

        # Layouts
        self.grid_layout = QGridLayout()

        settings_buttons_layout = QHBoxLayout()
        settings_buttons_layout.addWidget(self.button_nav_settings)
        settings_buttons_layout.addWidget(SpacerHorizontal())
        settings_buttons_layout.addWidget(self.button_conda_settings)
        settings_buttons_layout.addStretch()

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.button_reset)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.button_cancel)
        buttons_layout.addWidget(SpacerHorizontal())
        buttons_layout.addWidget(self.button_ok)
        buttons_layout.setContentsMargins(0, 10, 0, 0)

        self.nav_settings = NavigatorSettingsDialog(config)
        self.conda_settings = CondaSettingsDialog()

        main_layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        scroll.setWidget(scroll_widget)

        content_layout = QVBoxLayout(scroll_widget)
        content_layout.addLayout(self.grid_layout)

        main_layout.addWidget(scroll)
        main_layout.addWidget(SpacerVertical())
        main_layout.addLayout(settings_buttons_layout)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        # Signals
        self.button_ok.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)
        self.button_reset.clicked.connect(self.reset_to_defaults)
        self.button_reset.clicked.connect(lambda: self.button_ok.setEnabled(True))
        self.button_nav_settings.clicked.connect(self.show_navigator_settings_dialog)
        self.button_conda_settings.clicked.connect(self.show_conda_settings_dialog)

        # Setup
        self.grid_layout.setSpacing(0)
        self.setup()
        self.button_ok.setDisabled(True)
        self.widgets[0].setFocus()
        self.button_ok.setDefault(True)
        self.button_ok.setAutoDefault(True)

    def show_navigator_settings_dialog(self, e):
        self.nav_settings.setup(self.config.filename())
        self.nav_settings.exec_()

    def show_conda_settings_dialog(self, e):
        self.conda_settings.setup()
        self.conda_settings.exec_()

    # --- Helpers
    # -------------------------------------------------------------------------
    def get_option(self, option, section='main'):
        """Get configuration option from `main` section."""
        return self.config.get(section, option, None)

    def set_option(self, option, value, section='main'):
        """Set configuration option in `main` section."""
        self.config.set(section, option, value)

    def get_option_default(self, option, section='main'):
        """Get configuration option default value in `main` section."""
        return self.config.get_default(section, option)

    def set_option_default(self, option, section='main'):
        """Set configuration option default value in `main` section."""
        self.set_option(option, self.get_option_default(option), section)

    def create_widget(  # pylint: disable=too-many-arguments
            self,
            widget=None,
            label=None,
            option=None,
            hint=None,
            check=None,
            info=None,
            section='main',
    ):
        """Create preference option widget and add to layout."""
        config_value = self.get_option(option, section=section)
        widget._text = label  # pylint: disable=protected-access
        widget.label = QLabel(label)
        widget.option = option
        widget.section = section
        widget.set_value(config_value)
        widget.label_information = QLabel()
        widget.label_information.setMinimumWidth(16)
        widget.label_information.setMaximumWidth(16)
        widget.label_information.setMinimumHeight(16)
        widget.label_information.setMaximumHeight(16)

        form_widget = QWidget()
        h_layout = QHBoxLayout()
        h_layout.addSpacing(4)
        h_layout.addWidget(widget.label_information, 0, Qt.AlignRight)
        h_layout.addWidget(widget, 0, Qt.AlignLeft)
        h_layout.addWidget(QLabel(hint or ''), 0, Qt.AlignLeft)
        form_widget.setLayout(h_layout)

        if check:
            widget.check_value = check
        else:
            widget.check_value = lambda value: (True, '')

        if info:
            label = widget.label_information
            label = PreferencesDialog.update_icon(label, INFO_ICON)
            label.setToolTip(info)

        self.widgets.append(widget)
        self.widgets_dic[option] = widget
        self.grid_layout.addWidget(widget.label, self.row, 0, Qt.AlignRight | Qt.AlignCenter)
        self.grid_layout.addWidget(form_widget, self.row, 1, Qt.AlignLeft | Qt.AlignCenter)
        self.row += 1

    def create_textbox(  # pylint: disable=too-many-arguments
            self, label, option, hint=None, check=None, info=None, placeholder=None, section='main',
    ):
        """Create textbox (QLineEdit) preference option."""
        widget = QLineEdit()
        widget.setAttribute(Qt.WA_MacShowFocusRect, False)
        widget.setMinimumWidth(250)

        if placeholder:
            widget.setPlaceholderText(placeholder)

        widget.get_value = lambda w=widget: w.text()
        widget.set_value = lambda value, w=widget: w.setText(value)
        widget.set_warning = lambda w=widget: w.setSelection(0, 1000)
        widget.textChanged.connect(lambda v=None, w=widget: self.options_changed(widget=w))

        self.create_widget(
            widget=widget,
            option=option,
            label=label,
            hint=hint,
            check=check,
            info=info,
            section=section,
        )

    def create_checkbox(self, label, option, check=None, hint=None, info=None):  # pylint: disable=too-many-arguments
        """Create checkbox preference option."""
        widget = QCheckBox()
        widget.get_value = lambda w=widget: bool(w.checkState())
        widget.set_value = lambda value, w=widget: bool(w.setCheckState(Qt.Checked if value else Qt.Unchecked))

        api_widget = self.widgets_dic['anaconda_api_url']
        widget.set_warning = lambda w=widget: api_widget
        widget.stateChanged.connect(lambda v=None, w=widget: self.options_changed(widget=w))
        self.create_widget(
            widget=widget,
            option=option,
            label=label,
            hint=hint,
            check=check,
            info=info,
        )

    def create_combobox(self, label, option, check=None, hint=None, info=None):  # pylint: disable=too-many-arguments
        widget = ComboBoxBase()
        widget.set_value = lambda *args, **kwargs: None
        widget.get_value = lambda w=widget: widget.currentData()

        default_env, selected_idx = self.get_option(option), 0

        for i, (prefix, name) in enumerate(self.environments.items()):
            if default_env == prefix:
                selected_idx = i

            widget.addItem(name, prefix)
            widget.setItemData(i, prefix, Qt.ToolTipRole)

        widget.setCurrentIndex(selected_idx)
        widget.currentIndexChanged.connect(lambda v=None, w=widget: self.options_changed(widget=w))

        self.create_widget(widget=widget, option=option, label=label, hint=hint, check=check, info=info)

    def options_changed(self, value=None, widget=None):
        """Callback helper triggered on preference value change."""
        config_value = self.get_option(widget.option, widget.section)

        if config_value != widget.get_value():
            self.widgets_changed.add(widget)
        else:
            if widget in self.widgets_changed:
                self.widgets_changed.remove(widget)

        self.button_ok.setDisabled(not self.widgets_changed)

    def widget_for_option(self, option):
        """Return the widget for the given option."""
        return self.widgets_dic[option]

    # --- API
    # -------------------------------------------------------------------------
    def set_initial_values(self):
        """
        Set configuration values found in other config files.

        Some options of configuration are found in condarc or in
        anaconda-client configuration.
        """
        # This method would also update Navigator's preference, unless user is logged into account on a trusted server
        self.api.client_get_ssl(set_conda_ssl=True)

    def setup(self):
        """Set up the preferences dialog."""
        def api_url_checker(value, allow_blank=False):
            """
            Custom checker to use selected ssl option instead of stored one.

            This allows to set an unsafe api url directly on the preferences dialog. Without this, one would have to
            first disable, click accept, then open preferences again and change api url for it to work.
            """
            # Ssl widget
            ssl_widget = self.widgets_dic.get('ssl_verification')
            verify = ssl_widget.get_value() if ssl_widget else True

            # Certificate path
            ssl_cert_widget = self.widgets_dic.get('ssl_certificate')
            if ssl_cert_widget:
                verify = ssl_cert_widget.get_value()

            # Offline mode
            offline_widget = self.widgets_dic.get('offline_mode')
            if ssl_widget or ssl_cert_widget:
                offline_mode = offline_widget.get_value()
            else:
                offline_mode = False

            if offline_mode:
                basic_check = (False, 'API Domain cannot be modified when working in <b>offline mode</b>.<br>')
            else:
                basic_check = self.is_valid_api(value, verify=verify, allow_blank=allow_blank)

            return basic_check

        def ssl_checker(value):
            """Counterpart to api_url_checker."""
            api_url_widget = self.widgets_dic.get('anaconda_api_url')
            api_url = api_url_widget.get_value()
            return self.is_valid_api(api_url, verify=value)

        def ssl_certificate_checker(value):
            """Check if certificate path is valid/exists."""
            ssl_widget = self.widgets_dic.get('ssl_verification')
            verify = ssl_widget.get_value() if ssl_widget else True
            ssl_cert_widget = self.widgets_dic.get('ssl_certificate')
            path = ssl_cert_widget.get_value()
            return self.is_valid_cert_file(path, verify)

        self.set_initial_values()
        self.create_textbox(
            'Anaconda.org API domain',
            'anaconda_api_url',
            check=api_url_checker,
        )
        self.create_textbox(
            'Anaconda Server API domain',
            'anaconda_server_api_url',
            placeholder='http(s)://example.com',
            check=lambda base_url: api_url_checker(
                url_utils.join(base_url, 'api/system') if base_url else None,
                allow_blank=True,
            ),
        )
        self.create_textbox(
            'Enterprise 4 Repository API domain',
            'enterprise_4_repo_api_url',
            placeholder='http(s)://example.com',
            check=lambda base_url: api_url_checker(base_url, allow_blank=True),
        )
        self.create_checkbox(
            'Enable SSL verification',
            'ssl_verification',
            check=ssl_checker,
            hint=('<i>Disabling this option is not <br>'
                  'recommended for security reasons</i>'),
        )
        self.create_textbox(
            'SSL certificate path (Optional)',
            'ssl_certificate',
            check=ssl_certificate_checker,
        )

        self.create_combobox(
            'Default conda environment',
            'default_env',
        )
        info_message = '''To help us improve Anaconda Navigator, fix bugs,
and make it even easier for everyone to use Python,
we gather anonymized usage information, just like
most web browsers and mobile apps.'''
        self.create_checkbox(
            'Quality improvement reporting',
            'provide_analytics',
            info=info_message,
        )
        info_offline = DialogOfflineMode.MESSAGE_PREFERENCES
        extra = '<br><br>' if WIN7 else ''
        self.create_checkbox(
            'Enable offline mode',
            'offline_mode',
            info=info_offline + extra,
        )
        self.create_checkbox('Hide offline mode dialog', 'hide_offline_dialog')
        self.create_checkbox('Hide quit dialog', 'hide_quit_dialog')
        self.create_checkbox('Hide update dialog on startup', 'hide_update_dialog')
        self.create_checkbox('Hide running applications dialog', 'hide_running_apps_dialog')
        self.create_checkbox('Enable high DPI scaling', 'enable_high_dpi_scaling')
        self.create_checkbox('Show application startup error messages', 'show_application_launch_errors')
        self.create_checkbox('Show hidden Anaconda Server channels', 'anaconda_server_show_hidden_channels')

        ssl_ver_widget = self.widgets_dic.get('ssl_verification')
        ssl_ver_widget.stateChanged.connect(self.enable_disable_cert)
        ssl_cert_widget = self.widgets_dic.get('ssl_certificate')
        ssl_cert_widget.setPlaceholderText('Certificate to verify SSL connections')

        info_message = (
            'Directory path where {0} is installed on your machine. \n'  # pylint: disable=implicit-str-concat
            'To see {0} on the home tab, you will also need to click \n'
            'the Refresh button on the home tab.'
        )

        for app_spec in get_applications(cached=True).installable_apps.values():
            if not app_spec.is_available:
                continue

            self.create_textbox(
                f'{app_spec.display_name} path',
                f'{app_spec.app_name}_path',
                info=info_message.format(app_spec.display_name),
                section='applications',
            )

        # Refresh enabled/disabled status of certificate textbox
        self.enable_disable_cert()

    def enable_disable_cert(self, value=None):
        """Refresh enabled/disabled status of certificate textbox."""
        ssl_cert_widget = self.widgets_dic.get('ssl_certificate')
        if value:
            value = bool(value)
        else:
            ssl_ver_widget = self.widgets_dic.get('ssl_verification')
            value = bool(ssl_ver_widget.checkState())
        ssl_cert_widget.setEnabled(value)

    @staticmethod
    def update_icon(label, icon):
        """Update icon for information or warning."""
        pixmap = QPixmap(icon)
        label.setScaledContents(True)
        label.setPixmap(pixmap.scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return label

    @staticmethod
    def warn(widget, text=None):
        """Display warning for widget in preferences."""
        label = widget.label_information
        if text:
            label = PreferencesDialog.update_icon(label, WARNING_ICON)
            label.setToolTip(str(text))
            w = widget.label_information.width() // 2
            h = widget.label_information.height() // 2
            position = widget.label_information.mapToGlobal(QPoint(w, h))
            QCursor.setPos(position)
        else:
            label.setPixmap(QPixmap())
            label.setToolTip('')

    # --- Checkers
    # -------------------------------------------------------------------------
    def is_valid_url(self, url):
        """Check if a given URL returns a 200 code."""
        output = self.api.download_is_valid_url(url, non_blocking=False)
        error = ''
        if not output:
            error = 'Invalid api url.'
        return output, error

    @staticmethod
    def is_valid_cert_file(path, verify):
        """"Check if ssl certificate file in given path exists."""
        output = True
        error = ''

        # Only validate if it is not empty and if ssl_verification is checked
        if path.strip() and verify:
            output = os.path.isfile(path)
            if not output:
                error = 'File not found.'
        return output, error

    def is_valid_api(self, url, verify=True, allow_blank=False):
        """Check if a given URL is a valid anaconda api endpoint."""
        if (verify is not False) and (url_utils.netloc(url) in self.config.get('ssl', 'trusted_servers', [])):
            verify = False

        output = self.api.download_is_valid_api_url(url, non_blocking=False, verify=verify, allow_blank=allow_blank)
        if output:
            return output, ''

        if (output is download_api.ErrorDetail.ssl_error) and login_dialogs.TrustServerDialog(url, parent=self).exec_():
            output = self.api.download_is_valid_api_url(url, non_blocking=False, verify=False, allow_blank=allow_blank)
            if output:
                return output, ''

        error: str
        if ('/api' not in url) and self.is_valid_url(url)[0]:
            url_api_1 = url.replace('https://', 'https://api.').replace('http://', 'http://api.')
            url_api_2 = url.rstrip('/') + '/api'

            error = f'Invalid API url. <br><br><br>Try using:<br><b>{url_api_1}</b> or <br><b>{url_api_2}</b>'
        else:
            error = 'Invalid API url. <br><br>Check the url is valid and corresponds to the api endpoint.'

        return output, error

    def run_checks(self):
        """
        Run all check functions on configuration options.

        This method checks and warns but it does not change/set values.
        """
        checks = []
        for widget in self.widgets_changed:
            value = widget.get_value()
            check, error = widget.check_value(value)
            checks.append(check)

            if check:
                self.warn(widget)
            else:
                self.button_ok.setDisabled(True)
                widget.set_warning()
                self.warn(widget, error)
                break

        # Emit checks ready
        self.sig_check_ready.emit()
        return checks

    def reset_to_defaults(self):
        """Reset the preferences to the default values."""
        for widget in self.widgets:
            default = self.get_option_default(widget.option, widget.section)
            widget.set_value(default)

            # Flag all values as updated
            self.options_changed(widget=widget, value=default)
        self.sig_reset_ready.emit()

    def accept(self):
        """Override Qt method."""
        sig_updated = False
        anaconda_api_url = None
        checks = self.run_checks()

        # Update values
        if checks and all(checks):
            for widget in self.widgets_changed:
                value = widget.get_value()
                self.set_option(widget.option, value, widget.section)

                if widget.option == 'default_env':
                    telemetry.ANALYTICS.instance.event('change-preference', {'preference': 'default_env'})

                # Settings not stored on Navigator config, but taken from anaconda-client config
                if widget.option == 'anaconda_api_url':
                    anaconda_api_url = value  # Store it to be emitted
                    self.api.client_set_api_url(value)
                    sig_updated = True

                # ssl_verify/verify_ssl handles True/False/<Path to cert>
                # On navi it is split in 2 options for clarity
                if widget.option in ['ssl_certificate', 'ssl_verification']:
                    ssl_veri = self.widgets_dic.get('ssl_verification')
                    ssl_cert = self.widgets_dic.get('ssl_certificate')
                    verify = ssl_veri.get_value()
                    path = ssl_cert.get_value()

                    if path.strip() and verify:
                        value = path
                    else:
                        value = verify

                    logged_api_url: typing.Optional[str] = self.config.get('main', 'logged_api_url', None)
                    trusted_servers: typing.List[str] = self.config.get('ssl', 'trusted_servers', [])
                    if url_utils.netloc(logged_api_url or '') in trusted_servers:
                        self.api.client_set_ssl(False)
                    else:
                        self.api.client_set_ssl(value)

            if sig_updated and anaconda_api_url:

                def _api_info(worker, output, error):
                    conda_url = output.get('conda_url')
                    try:
                        self.sig_urls_updated.emit(anaconda_api_url, conda_url)
                        super(PreferencesDialog, self).accept()  # pylint: disable=super-with-arguments
                    except RuntimeError:
                        # Some tests on appveyor/circleci fail
                        pass

                worker = self.api.api_urls()
                worker.sig_chain_finished.connect(_api_info)

            super().accept()
