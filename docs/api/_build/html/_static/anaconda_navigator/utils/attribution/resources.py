# -*- coding: utf-8 -*-

"""Utilities to manage resources."""

from __future__ import annotations

__all__ = ['load_resource']

import json
import os
import typing
import uuid
import requests
from anaconda_navigator.utils.logs import http_logger, logger
from anaconda_navigator import config as navigator_config


def load_resource(url: str) -> typing.Optional[bytes]:
    """Fetch external resource."""
    try:
        response: requests.Response = requests.get(url, timeout=30)
        http_logger.http(response=response)
        response.raise_for_status()
        CACHE.store(url, response.content)
        return response.content
    except requests.RequestException:
        logger.exception('Unable to fetch resource: %s', url)
        return CACHE.get(url)


class Cache:
    """Class manipulate with cache data"""

    __slots__ = ('__content', '__root')

    def __init__(self, root: str) -> None:
        """Initialize new :class:`~Cache` instance."""
        self.__root: typing.Final[str] = root
        self.__content: typing.Dict[str, str] = {}

        os.makedirs(self.__root, exist_ok=True)

        self._load()

    @property
    def __index_path(self) -> str:  # noqa: D401
        """Path to the cache index file."""
        return os.path.join(self.__root, 'cache_mapping.json')

    def __record_path(self, key: str) -> str:
        """Prepare path to the cache record file."""
        return os.path.join(self.__root, f'cache_data_{key}')

    def _load(self) -> None:
        """Loading mapping data from mapping.json into content variable"""
        try:
            stream: typing.TextIO
            with open(self.__index_path, 'rt',  encoding='utf-8') as stream:
                self.__content = json.load(stream)
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError):
            logger.debug('cache index is broken')

    def _save(self) -> None:
        """Save mapping data from self.__context into json mapping file"""
        try:
            stream: typing.TextIO
            with open(self.__index_path, 'wt', encoding='utf-8') as stream:
                json.dump(self.__content, stream, ensure_ascii=False)
        except OSError:
            logger.debug('unable to update cache index')

    def store(self, key: str, data: bytes) -> None:
        """Save data into json data file by uuid key"""
        record_key: str
        try:
            record_key = self.__content[key]
        except KeyError:
            record_key = self.__content[key] = uuid.uuid4().hex
            self._save()

        try:
            stream: typing.BinaryIO
            with open(self.__record_path(record_key), 'wb') as stream:
                stream.write(data)
        except OSError:
            pass

    def get(self, key: str) -> typing.Optional[bytes]:
        """Get cache data by key"""
        try:
            stream: typing.BinaryIO
            with open(self.__record_path(self.__content[key]), 'rb') as stream:
                return stream.read()
        except (KeyError, OSError):
            return None


CACHE: typing.Final[Cache] = Cache(root=navigator_config.AD_CACHE)
