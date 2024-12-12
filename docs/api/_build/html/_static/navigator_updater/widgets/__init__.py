# -*- coding: utf-8 -*-

# pylint: disable=invalid-name,no-name-in-module,too-many-lines,unused-argument

# -----------------------------------------------------------------------------
# Copyright 2016 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------
"""Wigets module."""

# yapf: disable

# Third party imports
from qtpy.QtCore import Qt, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (QCheckBox, QComboBox, QFrame, QLabel, QLineEdit,
                            QMenu, QPushButton, QSizePolicy, QStackedWidget,
                            QToolButton, QWidget)

# Local imports
from navigator_updater.utils.qthelpers import (add_actions, create_action,
                                               update_pointer)

# yapf: enable


# --- Base widgets
# -----------------------------------------------------------------------------
class WidgetBase(QWidget):
    """Widget base implementation."""

    sig_hovered = Signal(bool)
    sig_focused = Signal(bool)

    def _fix_style(self):
        # Mac related issues
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)
        self.setFocusPolicy(Qt.StrongFocus)

    def _fix_layout(self, layout):
        if layout:
            layout.setSpacing(0)
            layout.setContentsMargins(0, 0, 0, 0)

            items = (layout.itemAt(i).widget() for i in range(layout.count()))
            for w in items:
                if w:
                    new_layout = w.layout()
                    self._fix_layout(new_layout)

    def focusInEvent(self, event):
        """Override Qt method."""
        QWidget.focusInEvent(self, event)
        self.setProperty('focused', True)

    def focusOutEvent(self, event):
        """Override Qt method."""
        QWidget.focusOutEvent(self, event)
        self.setProperty('focused', False)

    def enterEvent(self, event):
        """Override Qt method."""
        QWidget.enterEvent(self, event)
        self.setProperty('hovered', True)

    def leaveEvent(self, event):
        """Override Qt method."""
        QWidget.leaveEvent(self, event)
        self.setProperty('hovered', False)

    def setDisabled(self, value):
        """Override Qt method."""
        QWidget.setDisabled(self, value)
        self.setProperty('disabled', value)
        self.setProperty('enabled', not value)

    def setEnabled(self, value):
        """Override Qt method."""
        QWidget.setEnabled(self, value)
        self.setProperty('enabled', value)
        self.setProperty('disabled', not value)

    def setProperty(self, name, value):
        """Override Qt method."""
        QWidget.setProperty(self, name, value)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def setLayout(self, layout):
        """Override Qt method."""
        self._fix_layout(layout)
        QWidget.setLayout(self, layout)


class ButtonBase(QPushButton, WidgetBase):
    """Base button used in CSS styling."""

    def __init__(self, *args, **kwargs):
        """Base button used in CSS styling."""
        super().__init__(*args, **kwargs)
        self.setAutoDefault(False)
        self.setDefault(False)
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)
        self.setFocusPolicy(Qt.StrongFocus)

    def mousePressEvent(self, event):
        """Override Qt method."""
        super().mousePressEvent(event)
        self.setProperty('pressed', True)

    def mouseReleaseEvent(self, event):
        """Override Qt method."""
        super().mouseReleaseEvent(event)
        self.setProperty('pressed', False)


class ButtonToolBase(QToolButton, WidgetBase):
    """Base button used in CSS styling."""

    def __init__(self, parent=None, text=''):
        """Base button used in CSS styling."""
        super().__init__(parent=parent)
        self.setCheckable(False)
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setText(text)
        self.setFocusPolicy(Qt.StrongFocus)  # Needed on OSX
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)  # Needed on OSX
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class CheckBoxBase(QCheckBox, WidgetBase):
    """Checkbox used in CSS styling."""

    def __init__(self, *args, **kwargs):
        """Checkbox used in CSS styling."""
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)  # Needed on OSX
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)  # Needed on OSX


class ComboBoxBase(QComboBox):  # pylint: disable=too-few-public-methods
    """Combobox used in CSS styling."""

    def __init__(self, *args, **kwargs):
        """Combobox used in CSS styling."""
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)  # Needed on OSX
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)  # Needed on OSX

    def showPopup(self):
        """Override Qt method."""
        index = self.currentIndex()
        menu = QMenu(self)
        actions = []

        for i in range(self.count()):
            text = self.itemText(i)
            action = create_action(
                self,
                text,
                toggled=lambda v=None, i=i: self.setCurrentIndex(i)
            )

            actions.append(action)

            if i == index:
                action.setChecked(True)

        add_actions(menu, actions)
        menu.setFixedWidth(self.width())
        bottom_left = self.contentsRect().bottomLeft()
        menu.popup(self.mapToGlobal(bottom_left))


class FrameBase(QFrame, WidgetBase):
    """Button used in CSS styling."""

    def __init__(self, *args, **kwargs):
        """Button used in CSS styling."""
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.NoFocus)  # Needed on OSX
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)  # Needed on OSX

    def setLayout(self, layout):
        """Override Qt method."""
        self._fix_layout(layout)
        super().setLayout(layout)


class LabelBase(QLabel, WidgetBase):
    """Label used in CSS styling."""

    def __init__(self, *args, **kwargs):
        """Label used in CSS styling."""
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.NoFocus)  # Needed on OSX
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)  # Needed on OSX


class StackBody(QStackedWidget, WidgetBase):
    """Stacked widget used in CSS styling of main custom bar stack."""

    def __init__(self, *args, **kwargs):
        """Stacked widget used in CSS styling of main custom bar stack."""
        super().__init__(*args, **kwargs)
        self.setContentsMargins(0, 0, 0, 0)
        self.setFrameStyle(QFrame.NoFrame)
        self.setFocusPolicy(Qt.StrongFocus)  # Needed on OSX
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)  # Needed on OSX

    def setLayout(self, layout):
        """Override Qt method."""
        self._fix_layout(layout)
        super().setLayout(layout)


class LineEditBase(QLineEdit):
    """Line edit used in CSS styling."""

    def __init__(self, *args, **kwargs):
        """Line edit used in CSS styling."""
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)  # Needed on OSX
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)  # Needed on OSX

    def mousePressEvent(self, event):
        """Override Qt method."""
        super().mousePressEvent(event)
        self.setProperty('pressed', True)

    def mouseReleaseEvent(self, event):
        """Override Qt method."""
        super().mouseReleaseEvent(event)
        self.setProperty('pressed', False)


# --- Buttons
# -----------------------------------------------------------------------------
class ButtonToolNormal(ButtonToolBase):
    """Button used in CSS styling."""


class ButtonNormal(ButtonBase):
    """Button used in CSS styling."""


class ButtonPrimary(ButtonBase):
    """Button used in CSS styling."""


class ButtonDanger(ButtonBase):
    """Button used in CSS styling."""


class ButtonLink(QPushButton, WidgetBase):
    """
    Button use to represent a clickable (and keyboard focusable) web link.

    It is styled to be used as a label.
    """

    sig_hovered = Signal(bool)

    def __init__(self, *args, **kwargs):
        """
        Button use to represent a clickable (and keyboard focusable) web link.

        It is styled to be used as a label.
        """
        super().__init__(*args, **kwargs)
        self.setAutoDefault(False)
        self.setDefault(False)
        self.setFocusPolicy(Qt.StrongFocus)  # Needed on OSX
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)  # Needed on OSX

    def enterEvent(self, event):
        """Override Qt method."""
        super().enterEvent(event)
        update_pointer(Qt.PointingHandCursor)
        self.sig_hovered.emit(True)

    def leaveEvent(self, event):
        """Override Qt method."""
        super().leaveEvent(event)
        update_pointer()
        self.sig_hovered.emit(False)


class ButtonLabel(QPushButton):  # pylint: disable=too-few-public-methods
    """
    A button that is used next to ButtonLink to avoid missalignments.

    It looks and acts like a label.
    """

    def __init__(self, *args, **kwargs):
        """A button that is used next to ButtonLink to avoid missalignments."""
        super().__init__(*args, **kwargs)
        self.setDisabled(True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)  # Needed on mac


# --- Frames
# -----------------------------------------------------------------------------
class FrameTabBar(FrameBase):
    """Frame used in CSS styling."""


class FrameTabBody(FrameBase):
    """Frame used in CSS styling."""


class FrameTabHeader(FrameBase):
    """Frame used in CSS styling."""


class FrameTabContent(FrameBase):
    """Frame used in CSS styling."""


class FrameTabFooter(FrameBase):
    """Frame used in CSS styling."""


# --- Labels
# -----------------------------------------------------------------------------
class LabelLinks(LabelBase):
    """Label link used as url link."""

    LINKS_STYLE = """<style>
    a {
        color:green;
        text-decoration: underline;
    }
    </style>
    """

    def __init__(self, *args, **kwargs):
        """Label link used as url link."""
        super().__init__(*args, **kwargs)
        self.setOpenExternalLinks(False)
        self.linkActivated.connect(self._link_activated)
        self._original_text = self.text()
        self._add_style()

    def _add_style(self):
        text = self._original_text
        if self.LINKS_STYLE not in text:
            self.setText(self.LINKS_STYLE + text)

    @staticmethod
    def _link_activated(url):
        QDesktopServices.openUrl(QUrl(url))
        # tracker = GATracker()
        # tracker.track_event('content', 'link', url)

    def setText(self, text):
        """Override Qt method."""
        self._original_text = text
        super().setText(text)
        self._add_style()


# --- Spacers
# -----------------------------------------------------------------------------
class SpacerHorizontal(LabelBase):
    """Label used in CSS styling."""


class SpacerVertical(LabelBase):
    """Label used in CSS styling."""


# --- Other Buttons
# -----------------------------------------------------------------------------
class ButtonPrimaryAction(QPushButton):  # pylint: disable=too-few-public-methods
    """Button used in CSS styling."""


class ButtonCancel(QPushButton):  # pylint: disable=too-few-public-methods
    """Button used in CSS styling."""


class ButtonSecondaryTextual(QPushButton):  # pylint: disable=too-few-public-methods
    """Button used in CSS styling."""


class ButtonSecondaryIcon(QPushButton):  # pylint: disable=too-few-public-methods
    """Button used in CSS styling."""


class ButtonEnvironmentCancel(QPushButton):  # pylint: disable=too-few-public-methods
    """Button used in CSS styling."""


class ButtonEnvironmentPrimary(QPushButton):  # pylint: disable=too-few-public-methods
    """Button used in CSS styling."""


# --- Buttons that are used as labels.
class ButtonEnvironmentOptions(QPushButton):  # pylint: disable=too-few-public-methods
    """Button used in CSS styling."""


class FrameEnvironments(QFrame):  # pylint: disable=too-few-public-methods
    """Frame used in CSS styling."""


class FrameEnvironmentsList(QFrame):  # pylint: disable=too-few-public-methods
    """Frame used in CSS styling."""


class FrameEnvironmentsListButtons(QFrame):  # pylint: disable=too-few-public-methods
    """Frame used in CSS styling."""


class FrameEnvironmentsPackages(QFrame):  # pylint: disable=too-few-public-methods
    """Frame used in CSS styling."""
