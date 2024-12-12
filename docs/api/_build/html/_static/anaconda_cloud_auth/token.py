import base64
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Type
from typing import Union
from urllib.error import HTTPError

import jwt
import keyring
from keyring.backend import KeyringBackend
from keyring.backend import properties
from keyring.errors import PasswordDeleteError
from keyring.errors import PasswordSetError
from pydantic import BaseModel

from anaconda_cloud_auth.config import AuthConfig
from anaconda_cloud_auth.exceptions import TokenExpiredError
from anaconda_cloud_auth.exceptions import TokenNotFoundError

# Note: we can remove this if we pin keyring>=23.9.0
try:
    classproperty = properties.classproperty
except AttributeError:
    _KeyringClassMethod = Callable[[Type[KeyringBackend]], Any]

    def classproperty(method: _KeyringClassMethod) -> _KeyringClassMethod:
        return properties.ClassProperty(classmethod(method))


logger = logging.getLogger(__name__)

KEYRING_NAME = "Anaconda Cloud"


LocalKeyringData = Dict[str, Dict[str, str]]


def _as_base64_string(payload: str) -> str:
    """Encode a string to a base64 string"""
    return base64.b64encode(payload.encode("utf-8")).decode("utf-8")


class NavigatorFallback(KeyringBackend):
    priority = 0.1  # type: ignore

    @classproperty
    def viable(cls) -> bool:
        try:
            import anaconda_navigator  # noqa: F401

            return True
        except ModuleNotFoundError:
            return False

    def set_password(self, service: str, username: str, password: str) -> None:
        raise PasswordSetError("This keyring cannot set passwords")

    def _get_auth_domain(self) -> str:
        from anaconda_navigator.config import CONF as navigator_config

        known_mapping = {"https://anaconda.cloud": "id.anaconda.cloud"}

        cloud_base_url: str = navigator_config.get(
            "main", "cloud_base_url", "https://anaconda.cloud"
        ).strip("/")
        return known_mapping[cloud_base_url]

    def get_password(self, service: str, username: str) -> Union[str, None]:
        try:
            from anaconda_navigator.api.nucleus.token import NucleusToken

            auth_domain = self._get_auth_domain()
        except ImportError:
            return None

        if service != KEYRING_NAME and username != auth_domain:
            return None

        token = NucleusToken.from_file()
        if token is not None:
            from anaconda_cloud_auth.actions import _get_api_key
            from anaconda_cloud_auth.actions import refresh_access_token

            auth_config = AuthConfig(domain=auth_domain)
            if not token.valid:
                try:
                    access_token = refresh_access_token(
                        token.refresh_token, auth_config=auth_config
                    )
                except HTTPError:
                    return None
            else:
                access_token = token.access_token

            api_key = _get_api_key(access_token)
            token_info = {
                "username": token.username,
                "api_key": api_key,
                "domain": auth_config.domain,
            }
            payload = json.dumps(token_info)
            encoded = _as_base64_string(payload)
            keyring.set_password(KEYRING_NAME, auth_domain, encoded)

            return encoded
        return None

    def delete_password(self, service: str, username: str) -> None:
        auth_domain = self._get_auth_domain()
        try:
            from anaconda_navigator.api.nucleus.token import (
                TOKEN_FILE as navigator_token_file,
            )

        except ImportError:
            return None

        if service != KEYRING_NAME and username != auth_domain:
            return None
        else:
            try:
                os.remove(navigator_token_file)
            except FileNotFoundError:
                return None


class AnacondaKeyring(KeyringBackend):
    keyring_path = Path("~/.anaconda/keyring").expanduser()
    priority = 9  # type: ignore

    @classproperty
    def viable(cls) -> bool:
        try:
            cls.keyring_path.parent.mkdir(exist_ok=True, parents=True)
            with cls.keyring_path.open("a") as f:
                writable = f.writable()
            return writable
        except OSError:
            return False

    def _read(self) -> LocalKeyringData:
        if not self.keyring_path.exists():
            return {}

        try:
            with self.keyring_path.open("r") as fp:
                data = json.load(fp)
            return data
        except json.JSONDecodeError:
            return {}

    def _save(self, data: LocalKeyringData) -> None:
        self.keyring_path.parent.mkdir(exist_ok=True, parents=True)

        with self.keyring_path.open("w") as fp:
            json.dump(data, fp)

    def set_password(self, service: str, username: str, password: str) -> None:
        data = self._read()

        if service not in data:
            data[service] = {}

        data[service][username] = password

        self._save(data)

    def get_password(self, service: str, username: str) -> Union[str, None]:
        data = self._read()
        return data.get(service, {}).get(username, None)

    def delete_password(self, service: str, username: str) -> None:
        data = self._read()
        try:
            data.get(service, {}).pop(username)
            self._save(data)
        except KeyError:
            raise PasswordDeleteError


class TokenInfo(BaseModel):
    api_key: Union[str, None] = None
    username: Union[str, None] = None
    domain: str

    @classmethod
    def load(cls, domain: str) -> "TokenInfo":
        """Load the token information from the system keyring."""
        keyring_data = keyring.get_password(KEYRING_NAME, domain)
        if keyring_data is None:
            raise TokenNotFoundError

        decoded_bytes = base64.b64decode(keyring_data)
        decoded_dict = json.loads(decoded_bytes)
        logger.debug("🔓 Token has been successfully retrieved from system keychain 🎉")
        return TokenInfo(**decoded_dict)

    def save(self) -> None:
        """Write the token information to the system keyring."""
        payload = self.json()
        encoded = _as_base64_string(payload)
        keyring.set_password(KEYRING_NAME, self.domain, encoded)
        logger.debug("🔒 Token has been safely stored in system keychain 🎉")

    def delete(self) -> None:
        """Delete the token information from the system keyring."""
        try:
            keyring.delete_password(KEYRING_NAME, self.domain)
            if NavigatorFallback.viable:
                NavigatorFallback().delete_password(KEYRING_NAME, self.domain)
        except PasswordDeleteError:
            raise TokenNotFoundError

    @property
    def expired(self) -> bool:
        if self.api_key is None:
            return True

        decoded = jwt.decode(
            self.api_key, algorithms=["RS256"], options={"verify_signature": False}
        )
        expiry = dt.datetime.fromtimestamp(decoded["exp"]).replace(
            tzinfo=dt.timezone.utc
        )
        return expiry < dt.datetime.now(tz=dt.timezone.utc)

    def get_access_token(self) -> str:
        """Get the access token, ensuring login and refresh if necessary."""
        if self.api_key is None:
            try:
                new_token_info = TokenInfo.load(self.domain)
            except TokenNotFoundError:
                message = "No token found, please login with `anaconda login`"
                raise TokenNotFoundError(message)

            # Store the new token information for later retrieval
            self.username = new_token_info.username
            self.api_key = new_token_info.api_key

        assert self.api_key is not None

        if self.expired:
            raise TokenExpiredError(
                "Your login token as expired. Please login again using\n"
                "  anaconda login --force"
            )

        return self.api_key
