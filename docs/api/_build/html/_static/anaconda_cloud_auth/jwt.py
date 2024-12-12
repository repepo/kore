from typing import Any
from typing import Dict
from typing import Union

import requests
from jwt import PyJWKClient
from jwt import PyJWKClientError
from jwt import PyJWKSet


class JWKClient(PyJWKClient):
    def fetch_data(self) -> Dict[str, Any]:
        # This method fails in the original class due to using urlopen.
        # The jwks URI likely blocks the user-agent used by urlopen
        jwk_set: Union[Dict[str, Any], None] = None
        try:
            jwk_set = requests.get(self.uri).json()
        except requests.exceptions.RequestException as e:
            raise PyJWKClientError(f'Fail to fetch data from the url, err: "{e}"')
        else:
            assert jwk_set is not None
            return jwk_set
        finally:
            if self.jwk_set_cache is not None and jwk_set is not None:
                self.jwk_set_cache.put(PyJWKSet(jwk_set["keys"]))
