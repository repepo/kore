from functools import cached_property
from typing import List
from typing import Optional
from typing import Union

import requests
from pydantic import BaseModel
from pydantic import BaseSettings

from anaconda_cloud_auth import __version__ as version

OIDC_REQUEST_HEADERS = {"User-Agent": f"anaconda-cloud-auth/{version}"}


class APIConfig(BaseSettings):
    class Config:
        env_prefix = "ANACONDA_CLOUD_API_"
        env_file = ".env"
        keep_untouched = (
            cached_property,  # this allows pydantic models to have a cached_property
        )

    domain: str = "anaconda.cloud"
    key: Optional[str] = None

    @cached_property
    def aau_token(self) -> Union[str, None]:
        # The token is cached in anaconda_anon_usage, so we can also cache here
        try:
            from anaconda_anon_usage.tokens import token_string
        except ImportError:
            return None

        try:
            return token_string()
        except Exception:
            # We don't want this to block user login in any case,
            # so let any Exceptions pass silently.
            return None


class AuthConfig(BaseSettings):
    class Config:
        env_prefix = "ANACONDA_CLOUD_AUTH_"
        env_file = ".env"

    domain: str = "id.anaconda.cloud"
    client_id: str = "b4ad7f1d-c784-46b5-a9fe-106e50441f5a"
    redirect_uri: str = "http://127.0.0.1:8000/auth/oidc"
    openid_config_path: str = ".well-known/openid-configuration"

    @property
    def well_known_url(self: "AuthConfig") -> str:
        """The URL from which to load the OpenID configuration."""
        return f"https://{self.domain}/{self.openid_config_path}"

    @property
    def oidc(self) -> "OpenIDConfiguration":
        """The OIDC configuration, cached as a regular instance attribute."""
        res = requests.get(self.well_known_url, headers=OIDC_REQUEST_HEADERS)
        res.raise_for_status()
        oidc_config = OpenIDConfiguration(**res.json())
        return self.__dict__.setdefault("_oidc", oidc_config)


class OpenIDConfiguration(BaseModel):
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str

    id_token_signing_alg_values_supported: List[str] = []
