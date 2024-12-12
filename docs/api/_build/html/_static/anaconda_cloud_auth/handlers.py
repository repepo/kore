import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from socket import socket
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Union
from urllib.parse import parse_qs
from urllib.parse import urlparse

import requests
from pydantic.main import BaseModel

from anaconda_cloud_auth.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class Result(BaseModel):
    """This class is needed to capture the auth code redirect data"""

    auth_code: str = ""
    state: str = ""
    scopes: List[str] = []


TRequest = Union[socket, Tuple[bytes, socket]]


class AuthCodeRedirectServer(HTTPServer):
    """A simple http server to handle the incoming auth code redirect from Ory"""

    _open_servers: Set["AuthCodeRedirectServer"] = set()

    def __init__(self, oidc_path: str, server_address: Tuple[str, int]):
        super().__init__(server_address, AuthCodeRedirectRequestHandler)
        self.result: Union[Result, None] = None
        self.host_name = str(self.server_address[0])
        self.oidc_path = oidc_path

    def __enter__(self) -> "AuthCodeRedirectServer":
        self._open_servers.add(self)
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            self._open_servers.remove(self)
        except KeyError:
            # In case another thread removed it already, just ignore
            pass
        return super().__exit__(exc_type, exc_val, exc_tb)

    def finish_request(self, request: TRequest, client_address: str) -> None:
        """Finish one request by instantiating RequestHandlerClass."""
        AuthCodeRedirectRequestHandler(
            self.oidc_path,
            self.host_name,
            request,
            client_address,
            server=self,
        )


class AuthCodeRedirectRequestHandler(BaseHTTPRequestHandler):
    """Request handler to get the auth code from the redirect from Ory"""

    server: AuthCodeRedirectServer

    def __init__(
        self,
        oidc_path: str,
        host_name: str,
        *args: Any,
        **kwargs: Any,
    ):
        # these are set before __init__ because __init__ calls the do_GET method
        self.oidc_path = oidc_path
        self.host_name = host_name

        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Override base method to suppress log message."""

    def _handle_auth(self, query_params: Dict[str, List[str]]) -> None:
        if "code" in query_params and "state" in query_params:
            location = "https://anaconda.cloud/local-login-success"
            self.server.result = Result(
                auth_code=query_params["code"][0],
                state=query_params["state"][0],
                scopes=query_params.get("scope", []),
            )
        else:
            location = "https://anaconda.cloud/local-login-error"

        self.send_response(HTTPStatus.TEMPORARY_REDIRECT)
        self.send_header("Location", location)
        self.end_headers()

    def _not_found(self) -> None:
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urlparse(f"http://{self.host_name}{self.path}")
        query_params = parse_qs(parsed_url.query)

        # Only accept requests to self.oidc_path
        if parsed_url.path != self.oidc_path:
            self._not_found()
        else:
            self._handle_auth(query_params)


def capture_auth_code(redirect_uri: str, state: str) -> str:
    parsed_url = urlparse(redirect_uri)

    host_name, port = parsed_url.netloc.split(":")
    server_port = int(port or "80")
    oidc_path = parsed_url.path

    logger.debug(f"Listening on: {redirect_uri}")

    with AuthCodeRedirectServer(oidc_path, (host_name, server_port)) as web_server:
        web_server.handle_request()

    result = web_server.result

    if result is None:
        raise AuthenticationError("Could not complete authentication")

    if result.auth_code is None:
        raise AuthenticationError("No authorization code")

    if result.state != state:
        raise AuthenticationError("State does not match")

    return result.auth_code


def shutdown_all_servers() -> None:
    """Cancel all open AuthCodeRedirectServer instances, which may be blocking their thread waiting
    for an auth code.

    This function should be called from a separate thread than the servers.

    """
    for server in list(AuthCodeRedirectServer._open_servers):
        # Here, we just make a single request to force the `server.handle_request()` method to stop blocking.
        # There is probably a better way by calling some method, but :shrug:
        requests.get(f"http://{server.host_name}:{server.server_port}/cancel")
