import logging
import socket
import sys
import threading
from collections import namedtuple
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import requests

logger = logging.getLogger("repo_cli")


Config = namedtuple(
    "Config",
    [
        "client_id",
        "port",
        "authorize_url",
        "token_url",
        "redirect_uri",
        "server",
        "verify_ssl",
    ],
)

HTML_SUCCESS_RESPONSE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anaconda Server - Token received</title>
    <style>
        body { text-align: center; padding: 150px; }
        h1 { font-size: 50px; }
        body { font: 20px Helvetica, sans-serif; color: #333; }
        article { display: block; text-align: left; width: 650px; margin: 0 auto; }
        a { color: #dc8100; text-decoration: none; }
        a:hover { color: #333; text-decoration: none; }
    </style>
</head>
<body>
    <article>
        <h1>Token received</h1>
        <div>
            <p>We successfully received a token.<br>You can close the browser window and continue to use the CLI</p>
            <p>&mdash; Anaconda</p>
        </div>
    </article>
</body>
</html>
"""

HTML_404_RESPONSE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anaconda Server - Not Found</title>
    <style>
        body { text-align: center; padding: 150px; }
        h1 { font-size: 50px; }
        body { font: 20px Helvetica, sans-serif; color: #333; }
        article { display: block; text-align: left; width: 650px; margin: 0 auto; }
        a { color: #dc8100; text-decoration: none; }
        a:hover { color: #333; text-decoration: none; }
    </style>
</head>
<body>
    <article>
        <h1>404</h1>
        <div>
            <p>Not Found</p>
        </div>
    </article>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """The main handler for the auth server.

        To initiate the flow, we are openining the browser at http://localhost:PORT/get_token, which
        generates the redirect header to the OpenID Connect Provider/

        Once user is logged in on OpenID Connect provider, it's redirected back to the
        http://localhost:PORT/?code=CODE&.... location, query string contains the authorization
        code in `code` parameter.

        Next we need to exchange the code for a jwt token, calling the token endpoint.
        """
        if "/get_token" in self.path:
            qs = urlencode(
                {
                    "client_id": self.config.client_id,
                    "response_type": "code",
                    "redirect_uri": self.config.redirect_uri,
                }
            )
            redirect_url = "{authorize_url}?{qs}".format(
                authorize_url=self.config.authorize_url, qs=qs
            )
            return self.redirect(redirect_url)

        if "code=" in self.path:
            qs = parse_qs(urlparse(self.path).query)
            code = qs["code"][0]
            return self.process_code(code)

        self.response(404, HTML_404_RESPONSE)

    def process_code(self, code):
        """Exchanges the authorization code, send back by OpenID Connect provider for the token,
        calling the token endpoint with the proper client_id and redirect_uri.

        Args:
            code (str): authorization_code string

        Returns:
            None, stops the webserver by exiting and stopping the thread.
        """
        token_url = self.config.token_url
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
        }
        r = requests.post(token_url, data, verify=self.config.verify_ssl)

        self.response(200, HTML_SUCCESS_RESPONSE)
        if r.status_code == 200:
            access_token = r.json()["access_token"]
            self.config.server.access_token = access_token
            logger.info("Received access_token")
        else:
            logger.error(
                "Accessed %s with %s and got error %s", token_url, data, r.text
            )
        sys.exit()

    def redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def response(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, format, *args):
        pass


class WebServer:
    def __init__(self, client_id, openid_configuration_url, port=None, verify_ssl=None):
        self.client_id = client_id
        self.verify_ssl = verify_ssl

        openid_conf = requests.get(
            openid_configuration_url, verify=self.verify_ssl
        ).json()
        self.authorize_url = openid_conf["authorization_endpoint"]
        self.token_url = openid_conf["token_endpoint"]
        if port is not None:
            self.port = port
        else:
            self.port = self.find_unused_port()
        self._access_token = None

    @property
    def access_token(self):
        return self._access_token

    @access_token.setter
    def access_token(self, value):
        if value:
            self._access_token = value

    def start(self):
        server_address = ("", self.port)
        config = Config(
            port=self.port,
            client_id=self.client_id,
            authorize_url=self.authorize_url,
            token_url=self.token_url,
            redirect_uri=self.localhost_url("/"),
            server=self,
            verify_ssl=self.verify_ssl,
        )
        handler = partial(RequestHandler, config)
        server = HTTPServer(server_address, handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        return thread

    def find_unused_port(self):
        """Returns an unused port number on localhost. Will be used for webserver port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def localhost_url(self, path="/get_token"):
        return "http://localhost:{port}{path}".format(port=self.port, path=path)
