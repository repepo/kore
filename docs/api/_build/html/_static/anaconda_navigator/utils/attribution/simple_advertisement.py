# -*- coding: utf-8 -*-

"""Basic implementation of an advertisement."""

from __future__ import annotations

__all__ = ['CompositeAdvertisementWidget']

import typing
import warnings

from qtpy import QtCore
from qtpy import QtWidgets

from anaconda_navigator.config import preferences
from anaconda_navigator.utils import url_utils
from anaconda_navigator import widgets
from . import core
from . import resources


class SimpleAdvertisementDict(typing.TypedDict, total=False):
    """Definition of a single simple advertisement."""

    text: str
    redirect_url: str
    image_url: str


class CompositeAdvertisementDict(core.WidgetDict):  # pylint: disable=too-few-public-methods
    """Collection of a simple advertisements, that might be rotated in a single location."""

    advertisements: typing.MutableSequence[SimpleAdvertisementDict]


class CustomStackedWidget(QtWidgets.QStackedWidget):  # pylint: disable=too-few-public-methods
    """Widget for carousel of advertisements."""

    def __init__(self, parent: CompositeAdvertisementWidget) -> None:
        """Initialize new :class:`~CustomStackedWidget` instance."""
        super().__init__()

        advertisement: 'SimpleAdvertisementDict'
        for advertisement in parent.content:
            url: typing.Optional[str] = advertisement.get('redirect_url', None)
            if not url:
                warnings.warn('no `redirect_url` provided for an advertisement')
                continue

            image: typing.Optional[bytes] = None
            image_url: typing.Optional[str] = advertisement.get('image_url', None)
            if image_url:
                image = resources.load_resource(image_url)

            text: typing.Optional[str] = advertisement.get('text', None)
            if (not text) and (not image):
                warnings.warn('`text` or valid `image_url` should be provided for an advertisement')
                continue

            widget: widgets.LabelImageLinkVertical = widgets.LabelImageLinkVertical(
                text=text,
                url=parent.settings.inject_url_parameters(
                    url,
                    utm_medium='ad',
                    utm_content=url_utils.file_name(image_url),
                ),
                image=image,
                width=150,  # NOTE: width-height-margin should be calculated from placement
                height=150,
                margin=10,
            )
            widget.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.addWidget(widget)

        self.__timer: typing.Final[QtCore.QTimer] = QtCore.QTimer(self)
        self.__timer.setInterval(preferences.AD_SLIDESHOW_TIMEOUT * 1000)
        self.__timer.timeout.connect(self.next)
        if self.count() > 1:
            self.__timer.start()

    def next(self) -> None:
        """Show next advertisement."""
        current: int = self.currentIndex()
        if current < 0:
            return
        current += 1
        if current >= self.count():
            current = 0
        self.setCurrentIndex(current)


CONTENT = typing.Sequence['SimpleAdvertisementDict']


class CompositeAdvertisementWidget(core.PartnerWidget[CustomStackedWidget]):
    """Widget with a simple image advertisement banners."""

    __slots__ = ('__content',)

    def __init__(self, definition: 'CompositeAdvertisementDict', settings: core.PartnerSettings) -> None:
        """Initialize new :class:`~PartnerWidget` instance."""
        super().__init__(
            settings=settings,
            placement=definition['placement'],
        )

        self.__content: typing.Final[CONTENT] = definition['advertisements']

    @property
    def content(self) -> CONTENT:  # noqa: D401
        """Description of the advertisements to show in widget."""
        return self.__content

    def widget(self) -> CustomStackedWidget:
        """Generate new widget to use in Qt application."""
        return CustomStackedWidget(parent=self)
