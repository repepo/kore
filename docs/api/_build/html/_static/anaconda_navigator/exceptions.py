# -*- coding: utf-8 -*-

"""Navigator Exceptions and Exception handling module."""

from __future__ import annotations

__all__ = ['CustomMessageException', 'exception_handler']

import html
import os
import tempfile
import traceback
import typing
import urllib.request
import webbrowser


class CustomMessageException(Exception):
    """
    Exception with custom error message.

    :param message: Message to be displayed.

                    Should be a pre-formatted HTML content.
    """

    def __init__(self, message: str) -> None:
        """Initialize new :class:`~CustomMessageException` instance."""
        super().__init__(message)
        self.__message: typing.Final[str] = message

    @property
    def message(self) -> str:
        """Content to display on qt error box or html page"""
        return self.__message


def form_default_html_message(error: Exception, trace: str) -> str:
    """Form message when we do not get it in CustomMessageException instance."""
    return f'''
        <h1>Navigator Error</h1>
        <p>An unexpected error occurred on Navigator start-up</p>
        <h2>Report</h2>
        <p>
          Please report this issue in the anaconda
          <a href="https://github.com/ContinuumIO/anaconda-issues/issues">issue tracker</a>
        </p>
        <h2>Main Error</h2>
        <p><pre>{html.escape(str(error))}</pre></p>
        <h2>Traceback</h2>
        <p><pre>{html.escape(str(trace))}</pre></p>'''


def display_qt_error_box(message):
    """Display a Qt styled error message box."""
    # pylint: disable=import-outside-toplevel,cyclic-import
    from anaconda_navigator.app import start
    from anaconda_navigator.widgets.dialogs import MessageBoxError

    if not hasattr(start, 'app'):
        raise AttributeError('Qt application is not initialized')

    msg_box = MessageBoxError(
        title='Navigator Start Error',
        text=message,
        report=False,  # Disable reporting on github
        learn_more=None,
    )
    msg_box.setFixedWidth(600)
    return msg_box.exec_()


def display_browser_error_box(message: str) -> None:
    """
    Display a new browser tab with an error description.

    :param message: HTML content to show on web page.
    """
    content: str = f'''<!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Navigator Error</title>
          </head>
          <body>
            {message}
          </body>
        </html>'''

    file_descriptor: int
    file_path: str
    file_descriptor, file_path = tempfile.mkstemp(suffix='.html')

    file_stream: typing.TextIO
    with os.fdopen(file_descriptor, 'wt', encoding='utf-8') as file_stream:
        file_stream.write(content)

    webbrowser.open_new_tab(f'file://{urllib.request.pathname2url(os.path.abspath(file_path))}')


def exception_handler(
        func: typing.Callable[..., typing.Any],
        *args: typing.Any,
        **kwargs: typing.Any,
) -> typing.Optional[int]:
    """Handle global application exceptions and display information."""
    try:
        return_value = func(*args, **kwargs)
        if isinstance(return_value, int):
            return return_value
    except Exception as exception:  # pylint: disable=broad-except
        handle_exception(exception)
    return None


def handle_exception(error: Exception) -> None:
    """This will provide a dialog for the user with the error found."""
    # pylint: disable=import-outside-toplevel,cyclic-import
    from anaconda_navigator.utils.logs import logger

    logger.critical(error, exc_info=True)

    message: str
    if isinstance(error, CustomMessageException):
        message = error.message
    else:
        message = form_default_html_message(error, traceback.format_exc())

    try:
        display_qt_error_box(message)
    except Exception as exception:  # pylint: disable=broad-except
        logger.exception(exception)
        display_browser_error_box(message)
