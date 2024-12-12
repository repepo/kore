# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Classes used in CSS styling of login dialogs."""

__all__ = [
    'LabelMainLoginTitle', 'LabelMainLoginText', 'LabelMainLoginSubTitle', 'WidgetLoginInfoFrame',
    'WidgetLoginFormFrame', 'WidgetLoginCardsFrame', 'WidgetLoginCardsFrame', 'WidgetLoginPageContent',
    'SecondaryButton', 'WidgetLoginCard', 'WidgetNoticeFrame', 'LabelLoginLogo', 'WidgetHeaderFrame',
]

from anaconda_navigator.widgets import ButtonNormal, FrameBase
from anaconda_navigator.widgets.dialogs import LabelBase


class LabelMainLoginTitle(LabelBase):
    """Label used in CSS styling."""


class LabelMainLoginText(LabelBase):
    """Label used in CSS styling."""


class LabelMainLoginSubTitle(LabelBase):
    """Label used in CSS styling."""


class WidgetLoginInfoFrame(FrameBase):
    """Widget used in CSS styling."""


class WidgetLoginFormFrame(FrameBase):
    """Widget used in CSS styling."""


class WidgetLoginCardsFrame(FrameBase):
    """Widget used in CSS styling."""


class WidgetLoginPageContent(FrameBase):
    """Widget used in CSS styling."""


class SecondaryButton(ButtonNormal):
    """Label used in CSS styling."""


class WidgetLoginCard(FrameBase):
    """Widget used in CSS styling."""


class WidgetNoticeFrame(FrameBase):
    """Widget used in CSS styling."""


class LabelLoginLogo(LabelBase):
    """Label used in CSS styling."""


class WidgetHeaderFrame(FrameBase):
    """Widget used in CSS styling."""
