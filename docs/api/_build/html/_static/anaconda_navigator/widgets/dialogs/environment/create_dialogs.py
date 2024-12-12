# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Dialogs for creating new environments."""

__all__ = ['CreateDialog']

import collections
import re
import sys
import typing
from qtpy import QtCore
from qtpy import QtWidgets
from anaconda_navigator.api import conda_api
from anaconda_navigator import utils
from anaconda_navigator import widgets
from . import common


class PackageGroupOption(typing.NamedTuple):
    """
    Details about single option in package group (e.g. single version of `python` or `r` package).

    :param title: Title of the option, which is displayed in the combobox (e.g. '3.8.11').
    :param packages: List of packages, that should be installed when this option is selected
                     (e.g. ['python=3.8', 'extra-python-package'])

    Three values are used to sort available options in combobox: (`sort_prefix`, `sort_version`, `sort_postfix`).
    `sort_prefix` and `sort_postfix` are sorted lexicographically, while `sort_version` - by version order.
    """

    title: str
    packages: typing.Tuple[str, ...] = ()
    sort_prefix: str = ''
    sort_version: str = ''
    sort_postfix: str = ''


def compare_options(first: str, second: str) -> typing.Tuple[int, int]:
    """
    Get the difference between `first` and `second` lines.

    :returns: Tuple of:

              - Length of the common part of both strings (from the beginning)
              - Total length of mismatched tails of both strings (with a negative sign)
    """
    result: int = 0

    left: str
    right: str
    for left, right in zip(first, second):
        if left == right:
            result += 1
        else:
            break

    if result == 0:
        return 0, 0

    return result, 2 * result - len(first) - len(second)


class PackageGroup(QtCore.QObject):
    """
    Controls for single package group (`python`, `r`).

    :param title: Name of the group to show in interface.
    :param enabled: Enable this group (check the checkbox) as soon, as any option will be pushed to the group.
    :param default_option: Default option, that should be selected.

                           This value might be an "approximate" one - option that will be closest to it will be
                           selected. E.g. `default_option`="3.8" may be used to select "3.8.11" and not "3.7.11" or
                           "3.9.5".

                           If no value will be provided - last option will be selected (corresponds to "latest").
    """

    sig_refreshed = QtCore.Signal()

    def __init__(
            self,
            title: str,
            enabled: bool = False,
            default_option: typing.Optional[str] = None,
    ) -> None:
        """Initialize new :class:`~PackageGroup` instance."""
        super().__init__()

        self.enabled = enabled
        self.default_option = default_option

        self.checkbox = widgets.CheckBoxBase(title)
        self.checkbox.setChecked(False)
        self.checkbox.setDisabled(True)
        self.checkbox.stateChanged.connect(self.refresh)

        self.combo = widgets.ComboBoxBase()
        self.combo.setDisabled(True)

    @property
    def checked(self) -> bool:
        """Checkbox is checked for this group."""
        return self.checkbox.isChecked()

    @property
    def current(self) -> typing.Sequence[str]:
        """List of packages, that should be installed."""
        if not self.checkbox.isChecked():
            return ()

        data: typing.Optional[PackageGroupOption] = self.combo.currentData()
        if data is None:
            return ()
        return data.packages

    def push(self, options: typing.Iterable[PackageGroupOption]) -> None:
        """
        Add new `options` to the group.

        These options will be merged with the existing ones.
        """
        total: typing.Set[PackageGroupOption] = set()

        # Add existing options
        index: int = 0
        count: int = self.combo.count()
        while index < count:
            total.add(self.combo.itemData(index))
            index += 1

        # Merge with new options
        total.update(options)

        # Sort all options
        versions_order: typing.Mapping[str, int] = {
            item: index
            for index, item in enumerate(utils.sort_versions([item.sort_version for item in total]))
        }
        ordered: typing.Sequence[PackageGroupOption] = sorted(
            total,
            key=lambda item: (item.sort_prefix, versions_order[item.sort_version], item.sort_postfix),
        )

        # Push options
        ideal: str = self.combo.currentText() or self.default_option or ''

        best_value: str = ''
        best_score: typing.Tuple[int, int] = (0, 0)

        seen: typing.Set[str] = set()

        option: PackageGroupOption
        self.combo.clear()
        for index, option in enumerate(ordered):
            if option.title in seen:
                continue
            self.combo.addItem(option.title, option)
            seen.add(option.title)

            # Find the best matching option from the available ones
            # This part supports "approximate" default_options (e.g. default_option="3.8" may match "3.8.11")
            current_score: typing.Tuple[int, int] = compare_options(ideal, option.title)
            if current_score > best_score:
                best_value = option.title
                best_score = current_score

        if best_value:
            self.combo.setCurrentText(best_value)
        elif self.combo.count() > 0:
            self.combo.setCurrentIndex(self.combo.count() - 1)

        # Triggers
        self.refresh()

    def refresh(self) -> None:
        """Update state of the controls of the group."""
        if self.combo.count() == 0:
            self.checkbox.setEnabled(False)
            self.checkbox.setChecked(False)
        elif not self.checkbox.isEnabled():
            self.checkbox.setEnabled(True)
            self.checkbox.setChecked(self.enabled)

        self.combo.setEnabled(self.checkbox.isChecked())

        self.sig_refreshed.emit()


class CreateDialog(common.EnvironmentActionsDialog):  # pylint: disable=too-many-instance-attributes
    """Create new environment dialog."""

    sig_environments_updated = QtCore.Signal(object, object)
    sig_setup_ready = QtCore.Signal()
    sig_check_ready = QtCore.Signal()

    def __init__(self, parent=None, api=None):  # pylint: disable=too-many-statements
        """Create new environment dialog."""
        super().__init__(parent=parent, api=api)

        self.setMinimumWidth(self.BASE_DIALOG_WIDTH)
        self.setWindowTitle('Create new environment')

        self.label_name = widgets.LabelBase('Name:')
        self.label_name.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.text_name = common.LineEditEnvironment()
        self.text_name.setPlaceholderText('New environment name')
        self.text_name.textChanged.connect(self.refresh_name)

        self.label_location = widgets.LabelBase('Location:')
        self.label_location.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.label_prefix = widgets.LabelBase('')
        self.label_prefix.setObjectName('environment-location')

        self.label_packages = widgets.LabelBase('Packages:')
        self.label_packages.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.button_ok = widgets.ButtonPrimary('Create')
        self.button_ok.clicked.connect(self.accept)
        self.button_ok.setDisabled(True)

        self.button_cancel = widgets.ButtonNormal('Cancel')
        self.button_cancel.clicked.connect(self.reject)

        package_group: PackageGroup
        self.package_groups: typing.Mapping[str, PackageGroup] = collections.OrderedDict((
            (
                'python',
                PackageGroup(
                    title='Python',
                    enabled=True,
                    default_option=f'{sys.version_info.major}.{sys.version_info.minor}',
                ),
            ),
            ('r', PackageGroup('R')),
        ))
        for package_group in self.package_groups.values():
            package_group.sig_refreshed.connect(self.refresh)

        index: int
        layout_packages = QtWidgets.QGridLayout()
        for index, package_group in enumerate(self.package_groups.values()):
            if index:
                layout_packages.addWidget(widgets.SpacerVertical(), 2 * index - 1, 0)
            layout_packages.addWidget(package_group.checkbox, 2 * index, 0)
            layout_packages.addWidget(widgets.SpacerHorizontal(), 2 * index, 1)
            layout_packages.addWidget(package_group.combo, 2 * index, 2)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.label_name, 0, 0, 1, 1)
        grid.addWidget(widgets.SpacerHorizontal(), 0, 1, 1, 1)
        grid.addWidget(self.text_name, 0, 2, 1, 4)
        grid.addWidget(widgets.SpacerVertical(), 1, 0, 1, 1)
        grid.addWidget(self.label_location, 2, 0, 1, 1)
        grid.addWidget(widgets.SpacerHorizontal(), 2, 1, 1, 1)
        grid.addWidget(self.label_prefix, 2, 2, 1, 4)
        grid.addWidget(widgets.SpacerVertical(), 3, 0, 1, 1)
        grid.addWidget(self.label_packages, 4, 0, 1, 1)
        grid.addWidget(widgets.SpacerHorizontal(), 4, 1, 1, 1)
        grid.addLayout(layout_packages, 4, 2, 3, 1)

        layout_buttons = QtWidgets.QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_cancel)
        layout_buttons.addWidget(widgets.SpacerHorizontal())
        layout_buttons.addWidget(self.button_ok)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(grid)
        main_layout.addWidget(widgets.SpacerVertical())
        main_layout.addWidget(widgets.SpacerVertical())
        main_layout.addLayout(layout_buttons)

        self.setLayout(main_layout)
        self.text_name.setFocus()

        self.sig_setup_ready.connect(self._request_python)
        self.sig_setup_ready.connect(self._request_r)

    def _request_package(
            self,
            name: str,
            handler: typing.Callable[[typing.Any, typing.Any, typing.Any], None],
    ) -> None:
        """
        Request search of available package distributions.

        It tries to search in local cache first.
        If nothing found in cache - conda search might be used to find more in current channels.

        :param name: Name of the package to request versions for.
        :param handler: Handler to process search results with.
        """
        cached: typing.List[str] = self._packages.get(name, {}).get('versions', [])
        if cached:
            handler(None, {name: [{'version': item} for item in cached]}, '')
        else:
            conda_api.CondaAPI().search(name, offline=self.api.is_offline()).sig_finished.connect(handler)

    def _request_python(self) -> None:
        """Request list of available python versions from Conda."""
        self._request_package('python', self._fetch_python)

    def _fetch_python(
            self,
            _worker: typing.Optional[conda_api.ProcessWorker],
            output: typing.Mapping[str, typing.Any],
            error: str,
    ) -> None:
        """Parse available python versions from Conda."""
        if error:
            return

        pattern: typing.Pattern[str] = re.compile(r'^\d+\.\d+(?!\d)')

        version: str
        versions: typing.Dict[str, str] = {}
        raw_versions: typing.Sequence[str] = utils.sort_versions(
            versions={item['version'] for item in output.get('python', [])},
        )
        for version in raw_versions:
            reg_match: typing.Optional[typing.Match[str]] = pattern.match(version)
            if reg_match:
                versions[reg_match.group(0)] = version

        self.package_groups['python'].push([
            PackageGroupOption(long, (f'python={short}',), sort_prefix='python', sort_version=long)
            for short, long in versions.items()
        ])

    def _request_r(self) -> None:
        """Request list of available r versions from Conda."""
        self._request_package('r-base', self._fetch_r)

    def _fetch_r(
            self,
            _worker: typing.Optional[conda_api.ProcessWorker],
            output: typing.Mapping[str, typing.Any],
            error: str,
    ) -> None:
        """Parse available r versions from Conda."""
        if error:
            return

        versions: typing.Set[str] = {item['version'] for item in output.get('r-base', [])}

        self.package_groups['r'].push([
            PackageGroupOption(item, (f'r-base={item}', 'r-essentials'), sort_prefix='r-base', sort_version=item)
            for item in versions
        ])

    def refresh_name(self) -> None:
        """Update information, that is calculated from the name."""
        self.update_location()
        self.refresh()

    def refresh(self) -> None:
        """Update status of buttons based on data."""
        valid_environments: bool = bool(self.environments)
        valid_name: bool = self.is_valid_env_name(self.name)
        valid_package: bool = any(item.checked for item in self.package_groups.values())
        self.button_ok.setEnabled(valid_environments and valid_name and valid_package)

    @property
    def packages(self) -> typing.Sequence[str]:
        """List of packages, that must be installed in the new environment."""
        result: typing.Set[str] = set()

        package_group: PackageGroup
        for package_group in self.package_groups.values():
            result.update(package_group.current)

        return list(result)
