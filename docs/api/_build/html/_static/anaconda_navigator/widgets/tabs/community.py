# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Community Tab."""

from __future__ import annotations

__all__ = ['CommunityTab']

import collections
import contextlib
import json
import os
import re
import typing

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from anaconda_navigator.config import CONF, CONTENT_PATH, IMAGE_DATA_PATH
from anaconda_navigator.static.content import LINKS_INFO_PATH
from anaconda_navigator.static.images import LOGO_PATH, VIDEO_ICON_PATH
from anaconda_navigator.utils import download_manager
from anaconda_navigator.utils import url_utils
from anaconda_navigator.utils.logs import logger
from anaconda_navigator.widgets import ButtonBase, FrameTabContent, FrameTabHeader, SpacerHorizontal, WidgetBase
from anaconda_navigator.widgets.helperwidgets import LineEditSearch
from anaconda_navigator.widgets.lists.content import ListItemContent, ListWidgetContent


def parse_content(download: download_manager.Download) -> typing.Tuple[bool, typing.List[typing.Any]]:
    """
    Parse data fetched from https://anaconda.com/api/content.

    Returns tuple of primary content flag and the content itself. Primary content flag is used as a sign that primary
    content is loaded and doesn't need to be fetched from the file.
    """
    content: typing.List[typing.Any] = []
    try:
        stream: typing.TextIO
        with download.open('rt', encoding='utf-8') as stream:
            content = json.load(stream)
    except (OSError, TypeError, ValueError):
        pass
    return bool(content), content


def parse_events(download: download_manager.Download) -> typing.Tuple[bool, typing.List[typing.Any]]:
    """
    Parse data fetched from https://anaconda.com/api/events.

    Output is similar to :func:`~parse_content`.
    """
    content: typing.List[typing.Any] = []
    try:
        stream: typing.TextIO
        with download.open('rt', encoding='utf-8') as stream:
            content = json.load(stream)
    except (OSError, TypeError, ValueError):
        pass

    item: typing.Any
    for item in content:
        item['tags'] = ['event']
        item['uri'] = item.pop('url', '')

    return False, content[:25]


def parse_videos(download: download_manager.Download) -> typing.Tuple[bool, typing.List[typing.Any]]:
    """
    Parse data fetched from https://anaconda.com/api/videos.

    Output is similar to :func:`~parse_content`.
    """
    content: typing.List[typing.Any] = []
    try:
        stream: typing.TextIO
        with download.open('rt', encoding='utf-8') as stream:
            content = json.load(stream)
    except (OSError, TypeError, ValueError):
        pass

    item: typing.Any
    for item in content:
        item['tags'] = ['video']
        item['uri'] = item.pop('video', '')
        if item['uri']:
            item['banner'] = item.pop('thumbnail', '')
        item['date'] = item.pop('date_start', '')

    return False, content[:25]


def parse_webinars(download: download_manager.Download) -> typing.Tuple[bool, typing.List[typing.Any]]:
    """
    Parse data fetched from https://anaconda.com/api/webinars.

    Output is similar to :func:`~parse_content`.
    """
    content: typing.List[typing.Any] = []
    try:
        stream: typing.TextIO
        with download.open('rt', encoding='utf-8') as stream:
            content = json.load(stream)
    except (OSError, TypeError, ValueError):
        pass

    item: typing.Any
    for item in content:
        item['tags'] = ['webinar']
        item['uri'] = item.pop('url', '')

        image: typing.Any = item.pop('image', '')
        if image and isinstance(image, dict):
            item['banner'] = image.get('src', '')

    return False, content[:25]


def combine(group: download_manager.Group) -> typing.Tuple[bool, typing.Any]:
    """
    Combine outputs from all data sources into a single content

    Output is similar to :func:`~parse_content`.
    """
    total_flag: bool = False
    total_content: typing.Any = []

    download: download_manager.Download
    for download in group.downloads:
        if download.result is None:
            continue

        current_flag: bool
        current_content: typing.Any
        try:
            current_flag, current_content = download.result
        except (TypeError, ValueError):
            continue

        total_flag = total_flag or current_flag
        total_content.extend(current_content)

    return total_flag, total_content


# --- Widgets used in CSS styling
# -----------------------------------------------------------------------------

class ButtonToggle(ButtonBase):
    """Toggle button used in CSS styling."""

    def __init__(self, *args, **kwargs):
        """Toggle button used in CSS styling."""
        super().__init__(*args, **kwargs)
        self.setCheckable(True)
        self.clicked.connect(lambda v=None: self._fix_check)

    def _fix_check(self):
        self.setProperty('checked', self.isChecked())
        self.setProperty('unchecked', not self.isChecked())


# --- Main widgets
# -----------------------------------------------------------------------------

class CommunityTab(WidgetBase):  # pylint: disable=too-many-instance-attributes
    """Tab with a community content."""

    sig_ready = QtCore.Signal()

    def __init__(  # pylint: disable=too-many-arguments
            self,
            parent: typing.Optional[QtCore.QObject] = None,
            tags: typing.Optional[typing.Sequence[str]] = None,
            utm_medium: str = '',
    ) -> None:
        """Initialize new :class:`~CommunityTab` content."""
        super().__init__(parent=parent)

        self.__content: typing.List[typing.Any] = []
        self.__content_offset: int = 0
        self.__content_step: int = 1
        self.__tags: typing.Optional[typing.Sequence[str]] = tags

        self.__pixmaps: typing.Dict[str, QtGui.QPixmap] = {}
        self.__filter_widgets: typing.List[ButtonToggle] = []
        self.__default_pixmap = QtGui.QPixmap(VIDEO_ICON_PATH).scaled(
            100,
            60,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation,
        )

        # Widgets
        self.setObjectName('Tab')

        self.__timer_load: QtCore.QTimer = QtCore.QTimer()
        self.__timer_load.setInterval(333)

        self.__filters_layout = QtWidgets.QHBoxLayout()

        self.__text_filter: LineEditSearch = LineEditSearch()
        self.__text_filter.setPlaceholderText('Search')
        self.__text_filter.setAttribute(QtCore.Qt.WA_MacShowFocusRect, False)
        font_metrics: QtGui.QFontMetrics = self.__text_filter.fontMetrics()
        self.__text_filter.setMaximumWidth(font_metrics.width('M' * 23))

        layout_header: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        layout_header.addLayout(self.__filters_layout)
        layout_header.addStretch()
        layout_header.addWidget(self.__text_filter)

        frame_header: FrameTabHeader = FrameTabHeader()
        frame_header.setLayout(layout_header)

        self.__list: ListWidgetContent = ListWidgetContent(utm_medium=utm_medium)
        self.__list.setAttribute(QtCore.Qt.WA_MacShowFocusRect, False)
        self.__list.setMinimumHeight(200)

        layout_content: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        layout_content.addWidget(self.__list)

        frame_content: FrameTabContent = FrameTabContent()
        frame_content.setLayout(layout_content)

        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        layout.addWidget(frame_header)
        layout.addWidget(frame_content)
        self.setLayout(layout)

        # Signals
        self.__timer_load.timeout.connect(self.__put_content)
        self.__text_filter.textChanged.connect(self.__filter_content)

    def setup(self) -> None:
        """Set up tab content."""
        group = download_manager.Group(
            download_manager.Download('https://anaconda.com/api/events').into(
                os.path.join(CONTENT_PATH, 'events.json'),
            ).process(
                parse_events,
            ),
            download_manager.Download('https://anaconda.com/api/videos').into(
                os.path.join(CONTENT_PATH, 'videos.json'),
            ).process(
                parse_videos,
            ),
            download_manager.Download('https://anaconda.com/api/webinars').into(
                os.path.join(CONTENT_PATH, 'webinars.json'),
            ).process(
                parse_webinars,
            ),
            download_manager.Download('https://anaconda.com/api/content').into(
                os.path.join(CONTENT_PATH, 'content.json'),
            ).process(
                parse_content,
            ),
        ).require(
            download_manager.Require.NONE,
        ).process(
            combine,
        )

        group.sig_succeeded.connect(lambda _: self.__process_content(group=group))

        download_manager.manager().execute(group)

    def __process_content(self, group: download_manager.Group) -> None:
        """Apply content to the tab after it is loaded from the web."""
        flag: bool
        content: typing.List[typing.Any]
        flag, content = group.result

        stream: typing.TextIO
        if not flag:
            try:
                with open(LINKS_INFO_PATH, 'rt', encoding='utf-8') as stream:
                    content.extend(json.load(stream))
            except (OSError, ValueError):
                logger.exception('failed to load bundled community content')

        item: typing.Any
        for item in content:
            item['uri'] = item['uri'].replace('<p>', '').replace('</p>', '').replace(' ', '%20')

            banner: str = item.get('banner', '')
            if banner:
                item['image_file'] = url_utils.safe_file_name(banner)

            item['image_file_path'] = os.path.join(IMAGE_DATA_PATH, item.get('image_file', '...'))

        content.sort(key=lambda tile: (tile.get('sticky', '') not in (True, 'True', 'true', '1'), tile.get('tags', [])))
        self.__content = content

        # with open(CONTENT_JSON_PATH, 'wt', encoding='utf-8') as stream:
        #     json.dump(content, stream)

        self.__make_tag_filters()
        self.__timer_load.start(31)

    def __make_tag_filters(self) -> None:
        """Create tag filtering checkboxes based on available content tags."""
        counts: typing.Counter[str] = collections.Counter(
            tag
            for item in self.__content
            for tag in item.get('tags', [])
        )

        if self.__tags is None:
            self.__tags = sorted(counts)

        for tag in sorted(self.__tags):
            count: int = counts[tag]
            if not count:
                continue

            toggle = ButtonToggle(f'{tag.capitalize()} ({count})'.strip())
            toggle.setObjectName(tag.lower())
            toggle.setChecked(CONF.get('checkboxes', tag.lower(), True))
            toggle.clicked.connect(self.__filter_content)

            self.__filter_widgets.append(toggle)
            self.__filters_layout.addWidget(toggle)
            self.__filters_layout.addWidget(SpacerHorizontal())

    def __put_content(self) -> None:
        """
        Add items to the list.

        This method adds only a part of the content. Each next call adds next part to the interface.
        """
        index: int
        for index in range(self.__content_offset, self.__content_offset + self.__content_step):
            if index >= len(self.__content):
                self.__timer_load.stop()
                self.sig_ready.emit()
                break

            item: typing.Any = self.__content[index]

            banner: str = item.get('banner', None) or ''
            path: str = item.get('image_file_path', None) or ''
            content_item: ListItemContent = ListItemContent(
                title=item.get('title', None) or '',
                subtitle=item.get('subtitle', None) or '',
                uri=item.get('uri', None) or '',
                date=item.get('date', None) or '',
                summary=item.get('summary', None) or '',
                tags=item.get('tags', []),
                banner=banner,
                path=path,
                pixmap=self.__default_pixmap,
            )
            self.__list.addItem(content_item)

            # This allows the content to look for the pixmap
            content_item.pixmaps = self.__pixmaps

            # Use images shipped with Navigator, if no image try the download
            local_image: str = os.path.join(LOGO_PATH, item.get('image_file', '...'))
            if os.path.isfile(local_image):
                self.__process_thumbnail(content_item, local_image)
            elif banner and path:
                self.__download_thumbnail(content_item, banner, path)

        self.__content_offset += self.__content_step
        self.__filter_content()

    def __download_thumbnail(self, item: ListItemContent, url: str, path: str) -> None:
        """Fetch thumbnail for a card from the web."""
        download: download_manager.Download
        download = download_manager.Download(url).into(path).keep(download_manager.Keep.FRESH)
        download.sig_succeeded.connect(lambda: self.__process_thumbnail(item, path))
        download_manager.manager().execute(download)

    def __process_thumbnail(self, item: ListItemContent, path: str) -> None:
        """Process thumbnail and apply it to a card."""
        if path not in self.__pixmaps:
            try:
                if not os.path.isfile(path):
                    self.__pixmaps[path] = QtGui.QPixmap()
                else:
                    extension: str = os.path.splitext(path)[1][1:].upper()
                    if extension in ('PNG', 'JPG', 'JPEG'):
                        self.__pixmaps[path] = QtGui.QPixmap(path, format=extension)
                    else:
                        self.__pixmaps[path] = QtGui.QPixmap(path)
            except OSError:
                logger.exception('failed to initialize community card thumbnail')

        with contextlib.suppress(RuntimeError):
            item.update_thumbnail(self.__pixmaps[path])

    def __filter_content(self):
        """Filter content by a search string on all the fields of the item."""
        tokens: typing.List[str] = list(
            filter(bool, map(str.strip, re.split(r'\W', self.__text_filter.text().lower()))),
        )

        toggle: ButtonToggle
        selected_tags: typing.List[str] = []
        for toggle in self.__filter_widgets:
            tag: str = toggle.objectName()
            checked: bool = toggle.isChecked()

            CONF.set('checkboxes', tag, checked)
            if checked:
                selected_tags.append(tag)

        index: int
        for index in range(self.__list.count()):
            item = self.__list.item(index)

            all_checks: typing.List[bool] = []

            token: str
            for token in tokens:
                all_checks.append(
                    token in item.title.lower() or
                    token in item.venue.lower() or
                    token in ' '.join(item.authors).lower() or
                    token in item.summary.lower()
                )

            all_checks.append(any(
                tag.lower() in selected_tags
                for tag in item.tags
            ))

            item.setHidden(not all(all_checks))

    def ordered_widgets(
            self,
            next_widget: typing.Optional[QtWidgets.QWidget] = None,  # pylint: disable=unused-argument
    ) -> typing.List[QtWidgets.QWidget]:
        """Fix tab order of UI widgets."""
        return [
            *self.__filter_widgets,
            self.__text_filter,
            *self.__list.ordered_widgets(),
        ]

    def update_style_sheet(self) -> None:
        """Update custom CSS stylesheet."""
        self.__list.update_style_sheet()
