# -*- coding: utf-8 -*-

"""Structure to store API endpoint URLs."""

__all__ = ['EndpointCollection']

import typing
from urllib import parse


class EndpointCollection(typing.Mapping[str, typing.Any]):
    """
    Collection of API endpoint URLs.

    :param base_url: Root URL, parent for all API endpoints.
    :param endpoints: Collection of links, that are relative to the `base_url`.

                      Hierarchies are also supported - collection of links might be used besides relative urls.
    """

    __slots__ = ('__base_url', '__endpoints')

    def __init__(self, base_url: str, endpoints: typing.Mapping[str, typing.Any]) -> None:
        """Initialize new :class:`~ApiCollection` instance."""
        self.__base_url: typing.Final[str] = base_url

        key: str
        value: typing.Any
        data: typing.Dict[str, typing.Union[str, EndpointCollection]] = {}
        for key, value in endpoints.items():
            if isinstance(value, str):
                data[key] = parse.urljoin(self.__base_url, value)
            elif isinstance(value, typing.Mapping):
                data[key] = EndpointCollection(base_url=base_url, endpoints=value)
            else:
                raise TypeError(f'endpoint must be str or Mapping, not {type(value).__name__}')

        self.__endpoints: typing.Final[typing.Mapping[str, typing.Union[str, EndpointCollection]]] = data

    @property
    def base_url(self) -> str:  # noqa: D401
        """Root URL, parent for all API endpoints."""
        return self.__base_url

    def __getattr__(self, key: str) -> typing.Any:
        """Retrieve endpoint URL as an attribute."""
        try:
            return self.__endpoints[key]
        except KeyError:
            raise AttributeError(f'{type(self).__name__!r} object has no attribute {key!r}') from None

    def __getitem__(self, key: str) -> typing.Any:
        """Retrieve endpoint URL as an item."""
        return self.__endpoints[key]

    def __len__(self) -> int:
        """Retrieve total number of registered endpoints."""
        return len(self.__endpoints)

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate through endpoint names."""
        return iter(self.__endpoints)
