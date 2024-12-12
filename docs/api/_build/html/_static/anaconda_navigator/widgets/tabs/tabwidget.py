# -*- coding: utf-8 -*-

# pylint: disable=invalid-name,no-name-in-module

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Custom tab widget with custom tabbar."""

import typing

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from anaconda_navigator.utils import attribution
from anaconda_navigator.widgets import (
    ButtonLink, ButtonToolBase, FrameBase, FrameTabBar, FrameTabBody, LabelBase, StackBody
)


class LabelTabHeader(LabelBase):
    """Label used in CSS styling."""


class FrameTabBarBottom(FrameBase):
    """Frame used in CSS styling."""


class FrameTabBarLink(FrameBase):
    """Frame used in CSS styling."""


class FrameTabBarSocial(FrameBase):
    """Frame used in CSS styling."""


class ButtonTab(ButtonToolBase):
    """Button used in custom tab bar for CSS styling."""

    def __init__(self, *args, **kwargs):
        """Button used in custom tab bar for CSS styling."""
        super().__init__(*args, **kwargs)
        self.setCheckable(True)
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def keyPressEvent(self, event):
        """Override Qt method."""
        key = event.key()
        if key in [Qt.Key_Enter, Qt.Key_Return]:
            self.animateClick()
        super().keyPressEvent(event)


class TabBar(QWidget):  # pylint: disable=too-many-instance-attributes
    """Custom QTabBar that includes centered icons and text bellow the icon."""

    sig_index_changed = Signal(int)
    sig_url_clicked = Signal(object)

    def __init__(self, *args, **kwargs):
        """Custom QTabBar."""
        super().__init__(*args, **kwargs)
        self.buttons = []
        self.links = []
        self.links_social = []
        self.frame_bottom = FrameTabBarBottom()
        self.frame_social = FrameTabBarSocial()
        self.frame_link = FrameTabBarLink()
        self.current_index = None

        # Layouts
        self.layout_top = QVBoxLayout()
        self.layout_link = QVBoxLayout()
        self.layout_advertisement = QVBoxLayout()
        self.banners_stack = QStackedWidget()
        self.layout_social = QHBoxLayout()
        self._label_links_header = LabelTabHeader('')

        layout = QVBoxLayout()
        layout.addLayout(self.layout_top)
        layout.addStretch()

        self.layout_link.addLayout(self.layout_advertisement)
        self.frame_link.setLayout(self.layout_link)
        self.frame_social.setLayout(self.layout_social)

        layout_bottom = QVBoxLayout()
        layout_bottom.addWidget(self.frame_link)
        layout_bottom.addWidget(self.frame_social)
        self.frame_bottom.setLayout(layout_bottom)

        layout.addWidget(self.frame_bottom)
        #        self.layout_bottom.addWidget(self._label_links_header, 0,
        #                                     Qt.AlignLeft)
        self.setLayout(layout)

    def set_links_header(self, text):
        """Add links header to the bottom of the custom tab bar."""
        self._label_links_header.setText(text)

    def add_social(self, text, url=None):
        """Add social link on bottom of side bar."""
        button = ButtonLink()
        button.setText(' ')
        button.setObjectName(text.lower())
        button.setFocusPolicy(Qt.StrongFocus)
        button.clicked.connect(lambda v=None, url=url: self.sig_url_clicked.emit(url))
        self.layout_social.addWidget(button, 0, Qt.AlignCenter)
        self.links_social.append(button)

    def add_link(self, text: str, url: typing.Optional[str] = None, utm_medium: typing.Optional[str] = None) -> None:
        """Add link on bottom of side bar."""
        button = ButtonLink()
        button.setText(text)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button.setFocusPolicy(Qt.StrongFocus)

        def on_click(_checked: bool = False) -> None:
            if url is None:
                return

            new_url: str = url
            if utm_medium is not None:
                new_url = attribution.POOL.settings.inject_url_parameters(new_url, utm_medium=utm_medium)

            self.sig_url_clicked.emit(new_url)

        button.clicked.connect(on_click)
        self.layout_link.addWidget(button)
        self.links.append(button)

    def add_advertisement(self):
        """Put advertisement in the layout."""
        for item in reversed(range(self.layout_advertisement.count())):
            self.layout_advertisement.removeItem(self.layout_advertisement.itemAt(item))
        widget: typing.Optional[QWidget] = attribution.POOL[attribution.PartnerWidgetPlacement.BOTTOM_LEFT_CORNER]
        if widget is not None:
            self.layout_advertisement.addWidget(widget, alignment=Qt.AlignCenter)

    def add_tab(self, text, icon=None):
        """Create the widget that replaces the normal tab content."""
        button = ButtonTab()
        button.setObjectName(text.lower())
        button.setText(text)
        button.setFocusPolicy(Qt.StrongFocus)

        if icon:
            button.setIcon(icon)

        self.layout_top.addWidget(button)
        self.buttons.append(button)
        index = self.buttons.index(button)
        button.clicked.connect(lambda b=button, i=index: self.refresh(button, index))

    def refresh(self, button=None, index=None):
        """Refresh pressed status of buttons."""
        widths = []
        for b in self.buttons:
            b.setChecked(False)
            b.setProperty('checked', False)
            widths.append(b.width())

        max_width = max(widths)
        for b in self.buttons:
            b.setMinimumWidth(max_width)

        if button:
            button.setChecked(True)
            button.setProperty('checked', True)

        if index is not None:
            self.sig_index_changed.emit(index)
            self.current_index = index


class TabWidget(QWidget):
    """Curstom Tab Widget that includes a more customizable `tabbar`."""

    sig_current_changed = Signal(int)
    sig_url_clicked = Signal(object)

    def __init__(self, *args, **kwargs):
        """Custom Tab Widget that includes a more customizable `tabbar`."""
        super().__init__(*args, **kwargs)
        self.frame_sidebar = FrameTabBar()
        self.frame_tab_content = FrameTabBody()
        self.stack = StackBody()
        self.tabbar = TabBar()

        layout_sidebar = QVBoxLayout()
        layout_sidebar.addWidget(self.tabbar)
        self.frame_sidebar.setLayout(layout_sidebar)

        layout_content = QHBoxLayout()
        layout_content.addWidget(self.stack)
        self.frame_tab_content.setLayout(layout_content)

        layout = QHBoxLayout()
        layout.addWidget(self.frame_sidebar)
        layout.addWidget(self.frame_tab_content)

        self.setLayout(layout)
        self.tabbar.sig_index_changed.connect(self.setCurrentIndex)

        self.tabbar.sig_url_clicked.connect(self.sig_url_clicked)

    def count(self):
        """Override Qt method."""
        return self.stack.count()

    def widget(self, index):
        """Override Qt method."""
        return self.stack.widget(index)

    def currentWidget(self):
        """Override Qt method."""
        return self.stack.currentWidget()

    def currentIndex(self):
        """Override Qt method."""
        return self.tabbar.current_index

    def setCurrentIndex(self, index):
        """Override Qt method."""
        if self.currentIndex() != index:
            self.tabbar.current_index = index
            self.tabbar.buttons[index].setChecked(True)
            self.tabbar.buttons[index].setFocus()
            self.stack.setCurrentIndex(index)
            self.sig_current_changed.emit(index)

    def currentText(self):
        """Override Qt method."""
        index = self.currentIndex()
        text = ''
        if index:
            button = self.tabbar.buttons[self.currentIndex()]
            if button:
                text = button.text()
        return text

    def addTab(self, widget, icon=None, text=''):
        """Override Qt method."""
        if not widget:
            raise TypeError('tab widget cant be None')

        self.tabbar.add_tab(text, icon)
        self.stack.addWidget(widget)
        self.setCurrentIndex(0)

    def add_link(self, text: str, url: typing.Optional[str] = None, utm_medium: typing.Optional[str] = None) -> None:
        """Add links to the bottom area of the custom tab bar."""
        self.tabbar.add_link(text, url, utm_medium)

    def add_advertisement(self):  # pylint: disable=missing-function-docstring
        self.tabbar.add_advertisement()

    def add_social(self, text, url=None):
        """Add social link on bottom of side bar."""
        self.tabbar.add_social(text, url)

    def set_links_header(self, text):
        """Add links header to the bottom of the custom tab bar."""
        self.tabbar.set_links_header(text)

    def refresh(self):
        """Refresh size of buttons."""
        self.tabbar.refresh()
