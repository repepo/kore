from __future__ import annotations

import sys
from typing import Any
from typing import Optional

import requests
from intake.catalog import Catalog
from intake.catalog.local import CatalogParser

from . import __version__
from .exceptions import AnacondaCatalogsError
from .exceptions import ClientRequiresUpdate

DEFAULT_BASE_URI = "https://anaconda.cloud/api"
API_VERSION = "2023.02.20"
USER_AGENT = f"anaconda-catalogs/{__version__}"


class AnacondaCatalog(Catalog):
    name = "anaconda_catalog"

    def __init__(self, name: str, base_uri: Optional[str] = None, **kwargs: Any):
        self.name = name
        self.base_uri = base_uri or DEFAULT_BASE_URI
        super().__init__(name=name, **kwargs)

    def _get_from_server(self) -> dict[str, Any]:
        """Load the catalog spec from the API."""
        # Note: This is split out as a separate method to enable easier mocking
        url = slash_join(self.base_uri, "catalogs", self.name)
        response = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "Api-Version": API_VERSION,
                "User-Agent": USER_AGENT,
            },
        )
        # Even before trying the standard response handling
        # See if the backend server explicitly wants us to perform behavior
        # such as printing to the user or displaying a custom response
        data: dict[str, Any] = safe_get_json(response) or {}
        self._handle_response_display_message(data.get("display_message", {}) or {})

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise AnacondaCatalogsError(str(e))
        return response.json()

    def _get_catalog_spec(self) -> dict[str, Any]:
        """Load the Intake-parsable spec from the service, handling nested catalogs."""
        response_data = self._get_from_server()
        catalogs = response_data.get("catalogs")
        if catalogs is None:
            # It is a single catalog
            return response_data["spec"]

        # Otherwise, it's a catalog of catalogs
        sources = {}
        for catalog in catalogs:
            cid = catalog["id"]
            name = catalog["spec"].get("name") or cid
            normalized_name = "_".join(name.lower().split())
            sources[normalized_name] = {
                "driver": "anaconda_catalog",
                "args": {"name": cid},
            }
        return {"sources": sources}

    def _load(self) -> None:
        """Populate the catalog by loading the spec from the catalogs service."""
        spec = self._get_catalog_spec()

        # Parse the catalog spec. This is the same way Intake parses YAML catalogs internally.
        context = {"root": self.name}
        result = CatalogParser(spec, context=context)

        self._entries = {}

        cfg: dict[str, Any] = result.data or {}
        for entry in cfg["data_sources"]:
            entry._catalog = self
            self._entries[entry.name] = entry

        self.metadata.update(cfg.get("metadata") or {})
        self.description: str = self.description or cfg.get("description", "")

    def _handle_response_display_message(self, display_message: dict[str, Any]) -> None:
        if message := display_message.get("message_of_the_day"):
            print(message, file=sys.stderr)
        error_code = display_message.get("error_code", None)
        error_message = display_message.get("error_message", "")
        if error_code == "client_requires_update":
            raise ClientRequiresUpdate(
                error_message
                or f"anaconda-catalogs {__version__} is out-of-date. Please upgrade to the latest version",
            )
        elif error_code == "unknown_error":
            raise AnacondaCatalogsError(
                error_message or "An unknown error has occurred"
            )


# It really shouldn't be so complicated to safely create a URL from components...
# https://codereview.stackexchange.com/a/175423
def slash_join(*args: str) -> str:
    return "/".join(arg.strip("/") for arg in args)


def safe_get_json(response: requests.Response) -> Optional[dict[str, Any]]:
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return None
