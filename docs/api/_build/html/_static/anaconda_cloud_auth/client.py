import warnings
from typing import Any
from typing import Optional
from typing import Union
from urllib.parse import urljoin

import requests
from requests import PreparedRequest
from requests import Response
from requests.auth import AuthBase
from semver import VersionInfo

from anaconda_cloud_auth import __version__ as version
from anaconda_cloud_auth.config import APIConfig
from anaconda_cloud_auth.config import AuthConfig
from anaconda_cloud_auth.exceptions import LoginRequiredError
from anaconda_cloud_auth.exceptions import TokenNotFoundError
from anaconda_cloud_auth.token import TokenInfo


class BearerAuth(AuthBase):
    def __init__(
        self, domain: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        self.api_key = api_key
        if domain is None:
            domain = AuthConfig().domain

        self._token_info = TokenInfo(domain=domain)

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        if not self.api_key:
            try:
                r.headers[
                    "Authorization"
                ] = f"Bearer {self._token_info.get_access_token()}"
            except TokenNotFoundError:
                pass
        else:
            r.headers["Authorization"] = f"Bearer {self.api_key}"
        return r


class BaseClient(requests.Session):
    _user_agent: str = f"anaconda-cloud-auth/{version}"
    _api_version: Optional[str] = None

    def __init__(
        self,
        base_uri: Optional[str] = None,
        domain: Optional[str] = None,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        super().__init__()

        if base_uri and domain:
            raise ValueError("Can only specify one of `domain` or `base_uri` argument")

        kwargs = {}
        if domain is not None:
            kwargs["domain"] = domain
        if api_key is not None:
            kwargs["key"] = api_key

        self.config = APIConfig(**kwargs)
        # base_url overrides domain
        self._base_uri = base_uri or f"https://{self.config.domain}"
        self.headers["User-Agent"] = user_agent or self._user_agent
        self.api_version = api_version or self._api_version
        if self.api_version:
            self.headers["Api-Version"] = self.api_version
        self.auth = BearerAuth(api_key=self.config.key)

    def request(
        self,
        method: Union[str, bytes],
        url: Union[str, bytes],
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        joined_url = urljoin(self._base_uri, str(url))
        response = super().request(method, joined_url, *args, **kwargs)
        if response.status_code == 401 or response.status_code == 403:
            if response.request.headers.get("Authorization") is None:
                raise LoginRequiredError(
                    f"{response.reason}: You must login before using this API endpoint using\n"
                    f"  anaconda login"
                )

        self._validate_api_version(response.headers.get("Min-Api-Version"))

        return response

    def _validate_api_version(self, min_api_version_string: Optional[str]) -> None:
        """Validate that the client API version against the min API version from the service."""
        if min_api_version_string is None or self.api_version is None:
            return None

        # Convert to optional Version objects
        api_version = _parse_semver_string(self.api_version)
        min_api_version = _parse_semver_string(min_api_version_string)

        if api_version is None or min_api_version is None:
            return None

        if api_version < min_api_version:
            warnings.warn(
                f"Client API version is {self.api_version}, minimum supported API version is {min_api_version_string}. "
                "You may need to update your client.",
                DeprecationWarning,
            )


def _parse_semver_string(version: str) -> Optional[VersionInfo]:
    """Parse a version string into a semver Version object, stripping off any leading zeros from the components.

    If the version string is invalid, returns None.

    """
    norm_version = ".".join(s.lstrip("0") for s in version.split("."))
    try:
        return VersionInfo.parse(norm_version)
    except ValueError:
        return None


def client_factory(
    user_agent: Optional[str], api_version: Optional[str] = None
) -> BaseClient:
    return BaseClient(user_agent=user_agent, api_version=api_version)
