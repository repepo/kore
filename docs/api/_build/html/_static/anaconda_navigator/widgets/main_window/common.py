# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Common components for main window."""

__all__ = ['Component']

import typing
from qtpy import QtCore

if typing.TYPE_CHECKING:
    from anaconda_navigator.widgets import main_window


class Component(QtCore.QObject):
    """Abstract component to use in the main window."""

    __alias__: typing.ClassVar[str]

    def __init__(self, parent: 'main_window.MainWindow') -> None:
        """Initialize new :class:`~Component` instance."""
        super().__init__(parent=parent)
        self.__main_window: typing.Final[main_window.MainWindow] = parent

    @property
    def main_window(self) -> 'main_window.MainWindow':  # noqa: D401
        """Parent :class:`~anaconda_navigator.widgets.main_window.MainWindow` instance."""
        return self.__main_window

    # Virtual endpoints

    def setup(self, worker: typing.Any, output: typing.Any, error: str, initial: bool) -> None:
        """Perform component configuration from `conda_data`."""

    def update_style_sheet(self) -> None:
        """Update style sheet of the tab."""

    def start_timers(self) -> None:
        """Start component timers."""

    def stop_timers(self) -> None:
        """Stop component timers."""
