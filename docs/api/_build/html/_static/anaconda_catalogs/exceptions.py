"""Exceptions specifically related to the Anaconda catalogs service."""

from requests import HTTPError

# TODO: Add more refined exceptions


class AnacondaCatalogsError(HTTPError):
    pass


class ClientRequiresUpdate(HTTPError):
    pass
