# -*- coding: utf-8 -*-

# pylint: disable=invalid-name,no-name-in-module,too-many-lines,unused-argument

# -----------------------------------------------------------------------------
# Copyright 2016 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------
"""Custom widgets used for dialog definition and styling."""

# yapf: disable

# Third party imports
from qtpy.QtCore import Qt, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                            QListWidget, QListWidgetItem, QTextEdit,
                            QVBoxLayout)

# Local imports
# from navigator_updater.utils.py3compat import parse
from navigator_updater.utils.py3compat import to_text_string
from navigator_updater.utils.styles import load_style_sheet
from navigator_updater.widgets import (ButtonBase, ButtonDanger, ButtonNormal,
                                       ButtonPrimary, FrameBase, LabelBase,
                                       LineEditBase, SpacerHorizontal,
                                       SpacerVertical)

# yapf: enable


class FrameDialog(FrameBase):
    """Frame widget used for CSS styling of the body dialogs."""


class FrameDialogBody(FrameBase):
    """Frame widget used for CSS styling of the body dialogs."""


class FrameDialogTitleBar(FrameBase):
    """Frame widget used for CSS styling of the title bar of dialogs."""

    # pos, old_pos
    sig_moved = Signal(object, object)

    def __init__(self, parent=None):
        """Frame widget used for CSS styling of the title bar of dialogs."""
        super().__init__(parent=parent)
        self._mouse_pressed = False
        self.setMouseTracking(True)
        self._old_pos = None

    def mousePressEvent(self, event):
        """Override Qt method."""
        self._mouse_pressed = True
        self._old_pos = event.globalPos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Override Qt method."""
        if self._mouse_pressed:
            self.sig_moved.emit(event.globalPos(), self._old_pos)
            self._old_pos = event.globalPos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Override Qt method."""
        self._mouse_pressed = False
        super().mouseReleaseEvent(event)


class LabelDialogTitleBar(LabelBase):
    """Label used for CSS styling of the title of dialogs."""


class ButtonDialogClose(ButtonBase):
    """Button used for CSS styling ot the close dialog button."""


class DialogBase(QDialog):
    """Base dialog widget."""

    def __init__(self, *args, **kwargs):
        """Base dialog widget."""
        super().__init__(*args, **kwargs)

        # Widgets
        self.frame_dialog = FrameDialog(self)
        self.frame_title_bar = FrameDialogTitleBar(self)
        self.frame_body = FrameDialogBody(self)
        self.button_close_dialog = ButtonDialogClose('X')
        self.label_title_bar = LabelDialogTitleBar('Title')
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.MSWindowsFixedSizeDialogHint)
        self.setSizeGripEnabled(False)
        self.style_sheet = None

        # Widget setup
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.button_close_dialog.setFocusPolicy(Qt.NoFocus)

        # Signals
        self.button_close_dialog.clicked.connect(self.reject)
        self.frame_title_bar.sig_moved.connect(self._move_dialog)

        self.update_style_sheet()

    def setWindowTitle(self, title):
        """Qt override."""
        self.label_title_bar.setText(title)

    def setLayout(self, body_layout):
        """Qt override."""
        title_layout = QHBoxLayout()
        title_layout.addWidget(self.label_title_bar)
        title_layout.addStretch(100000000)
        title_layout.addWidget(self.button_close_dialog)
        title_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_title_bar.setLayout(title_layout)
        self.frame_body.setLayout(body_layout)

        layout_dialog = QVBoxLayout()
        layout_dialog.addWidget(self.frame_title_bar)
        layout_dialog.addWidget(self.frame_body)
        layout_dialog.setContentsMargins(0, 0, 0, 0)
        layout_dialog.setSpacing(0)
        self.frame_dialog.setLayout(layout_dialog)

        layout = QVBoxLayout()
        layout.addWidget(self.frame_dialog)
        self._fix_layout(layout)
        self._fix_layout(title_layout)
        super().setLayout(layout)

        # self.frame_title_bar.setMidLineWidth(self.frame_body.width())

    def _move_dialog(self, pos, old_pos):
        """Postion dialog callback to emulate title bar grab."""
        dx = old_pos.x() - pos.x()
        dy = old_pos.y() - pos.y()
        pos = self.pos()
        self.move(pos.x() - dx, pos.y() - dy)

    def _fix_layout(self, layout):
        if layout:
            layout.setSpacing(0)
            layout.setContentsMargins(0, 0, 0, 0)
            # layout.setSizeConstraint(QLayout.SetFixedSize)

            items = (layout.itemAt(i).widget() for i in range(layout.count()))
            for w in items:
                if w:
                    new_layout = w.layout()
                    self._fix_layout(new_layout)
        return layout

    def update_style_sheet(self, style_sheet=None):
        """Update custom css stylesheet."""
        if style_sheet is None:
            self.style_sheet = load_style_sheet()
        self.setStyleSheet(self.style_sheet)


class ListWidgetActionPackages(QListWidget):  # pylint: disable=too-few-public-methods
    """Custom widget for the actions to apply on package install/remove."""


class ActionsDialog(DialogBase):
    """Accept actions for pacakge manager."""

    def __init__(self, text, packages=(), parent=None):
        """Accept actions for pacakge manager."""
        super().__init__(parent=parent)

        self.packages = packages

        self.label = QLabel(text)
        self.list = ListWidgetActionPackages(self)
        self.button_cancel = ButtonDanger('Cancel')
        self.button_accept = ButtonPrimary('Ok')

        self.setWindowTitle('Proceed with the following actions?')

        for item in packages:
            item = QListWidgetItem(item)
            self.list.addItem(item)

        # Layout
        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_cancel)
        layout_buttons.addWidget(SpacerHorizontal())
        layout_buttons.addWidget(self.button_accept)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(SpacerVertical())
        layout.addWidget(self.list)
        layout.addWidget(SpacerVertical())
        layout.addWidget(SpacerVertical())
        layout.addLayout(layout_buttons)
        self.setLayout(layout)

        self.button_accept.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)


class InputDialog(DialogBase):
    """Input dialog."""

    def __init__(self, title='', text='', value=None, value_type=None):
        """Base message box dialog."""
        super().__init__()

        # Widgets
        self.label = LabelBase(text)
        self.text = LineEditBase()
        self.button_ok = ButtonPrimary('Ok')
        self.button_cancel = ButtonNormal('Cancel')

        # Widget setup
        self.setWindowTitle(to_text_string(title))
        if value:
            self.text.setText(str(value))

        # Layouts
        layout = QVBoxLayout()

        layout_text = QHBoxLayout()
        layout_text.addWidget(self.label)
        layout_text.addWidget(SpacerHorizontal())
        layout_text.addWidget(self.text)

        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_cancel)
        layout_buttons.addWidget(SpacerHorizontal())
        layout_buttons.addWidget(self.button_ok)

        layout.addLayout(layout_text)
        layout.addWidget(SpacerVertical())
        layout.addWidget(SpacerVertical())
        layout.addLayout(layout_buttons)

        self.setLayout(layout)

        # Signals
        self.button_ok.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)


class MessageBox(DialogBase):  # pylint: disable=too-many-instance-attributes
    """Base message box dialog."""

    QUESTION_BOX = 100   # pylint: disable=invalid-name
    INFORMATION_BOX = 101   # pylint: disable=invalid-name
    ERROR_BOX = 102   # pylint: disable=invalid-name
    REMOVE_BOX = 103   # pylint: disable=invalid-name

    sig_url_clicked = Signal(object)

    def __init__(  # pylint: disable=too-many-arguments,too-many-statements
            self, type_, error='', title='', text='', learn_more=None
    ):
        """Base message box dialog."""
        super().__init__()
        #        self.tracker = GATracker()
        self.label_text = QLabel(to_text_string(text))
        self.textbox_error = QTextEdit()
        self.button_ok = ButtonPrimary('Ok')
        self.button_yes = ButtonPrimary('Yes')
        self.button_no = ButtonNormal('No')
        self.button_copy = ButtonNormal('Copy text')
        self.button_learn = ButtonNormal('Learn more')
        self.button_remove = ButtonDanger('Remove')
        self.button_cancel = ButtonNormal('Cancel')
        self.bbox = QDialogButtonBox(Qt.Horizontal)

        self.label_text.setOpenExternalLinks(False)
        self.label_text.linkActivated.connect(self.url_clicked)
        self.textbox_error.setReadOnly(True)
        self.textbox_error.setFrameStyle(QTextEdit.Plain)
        self.textbox_error.setFrameShape(QTextEdit.NoFrame)
        self.setMinimumWidth(260)
        self.textbox_error.verticalScrollBar().show()
        self.setWindowTitle(to_text_string(title))

        error = to_text_string(error).split('\n')
        error = '<br>'.join(error)
        self.textbox_error.setText(error)

        # Layouts
        layout = QVBoxLayout()
        layout.addWidget(self.label_text)
        layout.addWidget(SpacerVertical())
        if error:
            layout.addWidget(self.textbox_error)
            layout.addWidget(SpacerVertical())
            layout.addWidget(self.button_copy)
            layout.addWidget(SpacerVertical())
        layout.addWidget(SpacerVertical())

        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()

        layout.addLayout(layout_buttons)

        self.layout = layout
        self.setLayout(layout)

        # Signals
        self.button_copy.clicked.connect(self.copy_text)
        self.button_ok.clicked.connect(self.accept)
        self.button_yes.clicked.connect(self.accept)
        self.button_no.clicked.connect(self.reject)
        self.button_remove.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

        # Setup
        self.button_learn.setVisible(bool(learn_more))
        if bool(learn_more):
            layout_buttons.addWidget(self.button_learn)
            layout_buttons.addWidget(SpacerHorizontal())
            self.button_learn.clicked.connect(
                lambda: self.show_url(learn_more)
            )

        if type_ == self.ERROR_BOX:
            layout_buttons.addWidget(self.button_ok)
            self.button_yes.setVisible(False)
            self.button_no.setVisible(False)
            self.button_remove.setVisible(False)
            self.button_cancel.setVisible(False)
        elif type_ == self.INFORMATION_BOX:
            layout_buttons.addWidget(self.button_ok)
            self.button_yes.setVisible(False)
            self.button_no.setVisible(False)
            self.textbox_error.setVisible(False)
            self.button_copy.setVisible(False)
            self.button_remove.setVisible(False)
            self.button_cancel.setVisible(False)
        elif type_ == self.QUESTION_BOX:
            layout_buttons.addStretch()
            layout_buttons.addWidget(self.button_no)
            layout_buttons.addWidget(SpacerHorizontal())
            layout_buttons.addWidget(self.button_yes)
            layout_buttons.addWidget(SpacerHorizontal())
            self.textbox_error.setVisible(False)
            self.button_ok.setVisible(False)
            self.button_copy.setVisible(False)
            self.button_remove.setVisible(False)
            self.button_cancel.setVisible(False)
        elif type_ == self.REMOVE_BOX:
            layout_buttons.addStretch()
            layout_buttons.addWidget(self.button_cancel)
            layout_buttons.addWidget(SpacerHorizontal())
            layout_buttons.addWidget(self.button_remove)
            layout_buttons.addWidget(SpacerHorizontal())
            self.textbox_error.setVisible(False)
            self.button_ok.setVisible(False)
            self.button_copy.setVisible(False)
            self.button_yes.setVisible(False)
            self.button_no.setVisible(False)

    def url_clicked(self, url):
        """Emit url interaction."""
        self.sig_url_clicked.emit(url)

    def copy_text(self):
        """Copy all the content of the displayed error message."""
        self.textbox_error.selectAll()
        self.textbox_error.copy()

    def show_url(self, url=None):
        """Open url in default browser."""
        if url:
            qurl = QUrl(url)
            QDesktopServices.openUrl(qurl)
            self.tracker.track_event('help', 'documentation', url)


class MessageBoxQuestion(MessageBox):
    """Question message box."""

    def __init__(self, text='', title=''):
        """Question message box."""
        super().__init__(
            text=text,
            title=title,
            type_=self.QUESTION_BOX,
        )


class MessageBoxRemove(MessageBox):
    """Question message box."""

    def __init__(self, text='', title=''):
        """Question message box."""
        super().__init__(
            text=text,
            title=title,
            type_=self.REMOVE_BOX,
        )


class MessageBoxInformation(MessageBox):
    """Information message box."""

    def __init__(self, text='', error='', title=''):
        """Information message box."""
        super().__init__(
            text=text,
            title=title,
            type_=self.INFORMATION_BOX,
        )


class MessageBoxError(MessageBox):
    """Error message box dialog with ability to send error reprots."""

    def __init__(  # pylint: disable=too-many-arguments
        self, text='', error='', title='', report=True, learn_more=None
    ):
        """Error message box dialog with ability to send error reprots."""
        super().__init__(
            text=text,
            title=title,
            error=error,
            type_=self.ERROR_BOX,
            learn_more=learn_more,
        )
        self.text = text
        self.error = error
        if report:
            self.button_send = ButtonNormal('Report Issue', parent=self)
            self.bbox.addButton(self.button_send, QDialogButtonBox.ActionRole)
            self.button_send.clicked.connect(self.send)

    def send(self):
        """Send error report to github and create an issue with a template."""


#        import webbrowser
#        base = "https://github.com/ContinuumIO/anaconda-issues/issues/new?{0}"
#        query = parse.urlencode(
#            {
#                'title': "Navigator Error",
#                'labels': "Navigator",
#                'body': "{}\n\n{}\n\n{}".format(
#                    self.text, self.error, GATracker().info
#                )
#            }
#        )
#        url = base.format(query)
#        webbrowser.open_new_tab(url)


# --- Local testing
# -----------------------------------------------------------------------------
def test():  # pragma : no cover
    """Run local tests."""
    from navigator_updater.utils.qthelpers import qapplication   # pylint: disable=import-outside-toplevel

    app = qapplication(test_time=3)
    widget_information = MessageBoxInformation(text='SomeRandomText')
    widget_information.update_style_sheet()
    widget_information.show()

    widget_error = MessageBoxError(text='SomeRandomText', error='Some error')
    widget_error.update_style_sheet()
    widget_error.show()

    actions_widget = ActionsDialog('Testing', packages=['a', 'b', 'c'])
    actions_widget.update_style_sheet()
    actions_widget.show()
    app.exec_()


if __name__ == '__main__':  # pragma : no cover
    test()
