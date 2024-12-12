# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Utils for login dialogs."""

import typing
from qtpy import QtCore
from qtpy import QtGui
from anaconda_navigator.utils import attribution


EMAIL_RE = QtCore.QRegExp(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
EMAIL_RE_VALIDATOR = QtGui.QRegExpValidator(EMAIL_RE)

USER_RE = QtCore.QRegExp(r'^[A-Za-z0-9_\.][A-Za-z0-9_\.\-]+$')
USER_RE_VALIDATOR = QtGui.QRegExpValidator(USER_RE)

FORGOT_LOGIN_URL = 'account/forgot_username'
FORGOT_PASSWORD_URL = 'account/forgot_password'  # nosec


class Span:  # pylint: disable=too-few-public-methods
    """Block of a plain text value."""

    __slots__ = ('__value',)

    def __init__(self, value: str) -> None:
        """Initialize new :class:`~Span` instance."""
        self.__value: typing.Final[str] = value

    @property
    def value(self) -> str:  # noqa: D401
        """Text value of a :class:`~Span`."""
        return self.__value

    def __str__(self) -> str:
        """Prepare string representation of an instance."""
        return self.__value


class UrlSpan(Span):  # pylint: disable=too-few-public-methods
    """Block of a URL with additional query properties."""

    __slots__ = ('__query',)

    def __init__(self, value: str, *, utm_medium: str, utm_content: str, **query: str) -> None:
        """Initialize new :class:`~UrlSpan` instance."""
        super().__init__(value)

        self.__query: typing.Final[typing.Mapping[str, str]] = {
            'utm_medium': utm_medium,
            'utm_content': utm_content,
            **query,
        }

    @property
    def query(self) -> typing.Mapping[str, str]:  # noqa: D401
        """Additional parameters to add to query part of a URL."""
        return self.__query

    def __str__(self) -> str:
        """Prepare string representation of an instance."""
        return attribution.POOL.settings.inject_url_parameters(self.value, force=False, **self.query)


class TextContainer(typing.NamedTuple):
    """Common details for multiple login dialogs."""

    title: typing.Optional[str] = None
    header_frame_logo_path: typing.Optional[str] = None
    info_frame_text: typing.Sequence[Span] = ()
    info_frame_logo_path: typing.Optional[str] = None
    form_forgot_links_msg: typing.Optional[str] = None
    form_primary_text: typing.Optional[str] = None
    form_secondary_text: typing.Optional[str] = None
    form_input_label_text: typing.Optional[str] = None
    form_submit_button_text: typing.Optional[str] = None
    message_box_error_text: typing.Optional[str] = None
