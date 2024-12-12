from __future__ import unicode_literals

import base64
import datetime
import json
import logging
import os
import re
import socket
from os.path import basename
from posixpath import join
from urllib.parse import urlparse

import requests
import urllib3

from .. import errors
from ..errors import InvalidName, NoDefaultUrl
from ..utils.config import UPLOAD_TYPE_MAPPING

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("repo_cli")

ROLE_AUTHOR = "author"
ROLE_ADMIN = "admin"


class RepoApi:
    def __init__(self, base_url, user_token=None, verify_ssl=True):
        self.base_url = base_url
        self.verify_ssl = verify_ssl
        self._access_token = user_token
        self._jwt = None
        self._urls = None

    @property
    def urls(self):
        if self._urls is None:
            if not self.base_url:
                raise NoDefaultUrl()
            self._urls = {
                "authorize": join(self.base_url, "auth", "authorize"),
                "account": join(self.base_url, "account"),
                "account_tokens": join(self.base_url, "account", "tokens"),
                "login": join(self.base_url, "auth", "login"),
                "logout": join(self.base_url, "logout"),
                "channels": join(self.base_url, "channels"),
                "artifacts": join(self.base_url, "artifacts"),
                "user_channels": join(self.base_url, "account", "channels"),
                "token_info": join(self.base_url, "account", "token-info"),
                "user_tokens": join(self.base_url, "account", "tokens"),
                "scopes": join(self.base_url, "account", "scopes"),
                "cves": join(self.base_url, "cves"),
                "system_settings": join(self.base_url, "system", "settings"),
                "mirrors": join(self.base_url, "mirrors"),
                "repo": join(self.base_url, "repo"),
                "reports": join(self.base_url, "reports"),
                "users": join(self.base_url, "users"),
                "system": join(self.base_url, "system"),
                "diagnose": join(self.base_url, "diagnose"),
            }
        return self._urls

    def get_user_token_name(self):
        return "repo-cli-token-{hostname}".format(hostname=socket.gethostname())

    def create_access_token(self):
        # TODO: Should we remove expires_at and let the server pick the default?

        data = {
            "name": self.get_user_token_name(),
            "expires_at": str(
                datetime.datetime.now().date() + datetime.timedelta(days=30)
            ),
            "scopes": self.get_scopes(),
        }
        resp = requests.post(
            self.urls["account_tokens"],
            data=json.dumps(data),
            headers=self.bearer_headers,
            verify=self.verify_ssl,
        )
        if resp.status_code != 200:
            msg = "Error requesting a new user token! Server responded with %s: %s" % (
                resp.status_code,
                resp.content,
            )
            logger.error(msg)
            raise errors.RepoCLIError(msg)

        self._access_token = resp.json()["token"]
        return self._access_token

    @property
    def bearer_headers(self):
        headers = {
            "Authorization": f"Bearer {self._jwt}",
            "Content-Type": "application/json",
        }
        return headers

    @property
    def is_admin_jwt(self):
        if not self._jwt:
            return False

        payload = self._jwt.split(".")[1]
        data = json.loads(base64.b64decode(payload + "=" * (len(payload) % 4)))

        roles = data["realm_access"]["roles"]
        return ROLE_ADMIN in roles

    @property
    def xauth_headers(self):
        return self.get_xauth_headers()

    def get_xauth_headers(self, extra_headers=None):
        headers = {"X-Auth": self._access_token}
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def get_xauth_token(self, force_scopes):
        logger.debug("[API] Getting access token.. ")
        token_resp = requests.get(
            self.urls["account_tokens"],
            headers=self.bearer_headers,
            verify=self.verify_ssl,
        )
        if token_resp.status_code != 200:
            msg = "Error retrieving user token! Server responded with %s: %s" % (
                token_resp.status_code,
                token_resp.content,
            )
            logger.error(msg)
            raise errors.RepoCLIError(msg)

        user_tokens = token_resp.json().get("items", [])
        logger.debug(f"[LOGIN] Access token retrieved.. {len(user_tokens)}")
        if user_tokens:
            user_token_name = self.get_user_token_name()
            user_tokens = [
                user_token
                for user_token in user_tokens
                if user_token["name"] == user_token_name
            ]
        if not user_tokens:
            return

        token_id = user_tokens[0]["id"]
        token_url = join(self.urls["account_tokens"], token_id)
        if force_scopes:
            # user token should be recreated from scratch - we need to remove it
            logger.debug(
                f"[XAUTH_TOKEN] Deleteing token because force_scopes option {token_id}"
            )
            token_resp = requests.delete(
                token_url, headers=self.bearer_headers, verify=self.verify_ssl
            )
            if token_resp.status_code == 204:
                logger.debug("[XAUTH_TOKEN] Token Deleted")
            elif token_resp.status_code == 404:
                logger.debug("[XAUTH_TOKEN] Token not deleted because not found")
            else:
                msg = "Error deleting user token! Server responded with %s: %s" % (
                    token_resp.status_code,
                    token_resp.content,
                )
                logger.error(msg)
                raise errors.RepoCLIError(msg)
        else:
            # ok, we got the token. Now we need to refresh it
            logger.debug(f"[XAUTH_TOKEN] Refreshing token.. {token_id}")
            token_resp = requests.put(
                token_url,
                data="{}",
                headers=self.bearer_headers,
                verify=self.verify_ssl,
            )
            if token_resp.status_code != 200:
                msg = "Error refreshing user token! Server responded with %s: %s" % (
                    token_resp.status_code,
                    token_resp.content,
                )
                logger.error(msg)
                raise errors.RepoCLIError(msg)
            self._access_token = new_token = token_resp.json()["token"]
            logger.debug("[XAUTH_TOKEN] Token Refreshed")
            return new_token

    def get_authorize_url(self):
        s = requests.Session()
        logger.debug("[LOGIN] Getting authorize endpoint")
        resp = s.get(
            self.urls["authorize"], verify=self.verify_ssl, allow_redirects=False
        )
        if resp.status_code != 302:
            logger.info("[LOGIN] Error getting auth configuration")
            logger.debug(f"Server responded with response {resp.status_code}")
            raise errors.WrongRepoAuthSetup()

        url = urlparse(resp.headers["Location"])
        url = url._replace(query="")
        return url

    def login(self, username, password) -> requests.Response:
        """Login using direct grant and returns the jwt token."""
        data = {"username": username, "password": password}
        s = requests.Session()
        logger.debug(f"[LOGIN] Authenticating user {username}...")
        resp = s.post(
            self.urls["login"],
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
            verify=self.verify_ssl,
        )
        logger.debug("[LOGIN] Done")

        if resp.status_code != 200:
            logger.info("[LOGIN] Error logging in...")
            logger.debug(
                f"Server responded with response {resp.status_code}\nData: {resp.content}"
            )
            raise errors.Unauthorized()

        self._jwt = jwt_token = resp.json()["token"]
        return jwt_token

    def get_user_token(self, force_scopes):
        """Returns user token, used with X-Auth headers."""
        user_token = self.get_xauth_token(force_scopes)

        if not user_token:
            logger.debug("[LOGIN] Access token not found. Creating one...")
            # Looks like user doesn't have any valid token. Let's create a new one
            user_token = self.create_access_token()
            logger.debug("[LOGIN] Done.")

        # TODO: we are assuming the first token is the one we need... We need to improve this waaaaay more
        return user_token

    def get_default_channel(self):
        url = self.urls["account"]
        logger.debug(f"[UPLOAD] Getting user default channel from {url}")
        response = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(response, "getting account details")[
            "default_channel_name"
        ]

    def get_current_user(self):
        url = self.urls["account"]
        logger.debug(f"[UPLOAD] Getting current user from {url}")
        response = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(response, "getting account details")

    def upload_file(self, filepath, channel, package_type):
        if package_type not in UPLOAD_TYPE_MAPPING:
            raise errors.RepoCLIError("%s upload is not supported" % package_type)

        artifact_type = UPLOAD_TYPE_MAPPING[package_type]

        url = join(self.base_url, "channels", channel, "artifacts")
        statinfo = os.stat(filepath)
        filename = basename(filepath)
        logger.debug(f"[UPLOAD] Using token {self._access_token} on {self.base_url}")
        multipart_form_data = {
            "content": (filename, open(filepath, "rb")),
            "filetype": (None, artifact_type),
            "size": (None, statinfo.st_size),
        }
        logger.info(f"Uploading {package_type} artifact to {url}...")
        response = requests.post(
            url,
            files=multipart_form_data,
            headers=self.xauth_headers,
            verify=self.verify_ssl,
        )
        return response

    def create_channel(self, channel):
        """Create a new channel with name `channel` on the repo server at `base_url` using `token`
        to authenticate.

        Args:
              channel(str): name of the channel to be created

        Returns:
              response (http response object)
        """
        _channel = channel
        logger.debug(f"Creating channel {_channel} on {self.base_url}")
        if "/" in channel:
            # this is a subchannel....
            channel, subchannel = channel.split("/")
            url = join(self.urls["channels"], channel, "subchannels")
            data = {"name": subchannel}
            headers = self.get_xauth_headers({"Content-Type": "application/json"})
        else:
            url = join(self.urls["channels"])
            data = {"name": channel}
            headers = self.get_xauth_headers({"Content-Type": "application/json"})

        self._validate_channel_name(data["name"])
        response = requests.post(
            url, json=data, headers=headers, verify=self.verify_ssl
        )
        return self._manage_response(
            response, f"creating channel {_channel}", success_codes=[201]
        )

    def remove_channel(self, channel):
        url = self._get_channel_url(channel)
        logger.debug(f"Removing channel {channel} on {self.base_url}")
        logger.debug(f"Using token {self._access_token}")
        response = requests.delete(
            url,
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )
        if response.status_code in [202]:
            logger.info(f"Channel {channel} successfully removed")
            logger.debug(
                f"Server responded with {response.status_code}\nData: {response.content}"
            )
        else:
            msg = (
                f"Error removing {channel}."
                f"Server responded with status code {response.status_code}.\n"
                f"Error details: {response.content}"
            )
            logger.error(msg)
            if response.status_code in [403, 401]:
                raise errors.Unauthorized()
        return response

    def update_channel(self, channel, success_message=None, **data):
        # we don't need to validate channel to change privacy
        if "privacy" not in data:
            self._validate_channel_name(channel)

        url = self._get_channel_url(channel)
        logger.debug(f"Updating channel {channel} on {self.base_url}")
        logger.debug(f"Using token {self._access_token}")
        response = requests.put(
            url,
            json=data,
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )
        if not success_message:
            success_message = f"Channel {channel} successfully updated."
        if response.status_code in [200, 204]:
            logger.info(success_message)
            logger.debug(
                f"Server responded with {response.status_code}\nData: {response.content}"
            )
        else:
            msg = (
                f"Error updating {channel} - "
                f"Server responded with status code {response.status_code}\n"
                f"Error details: {response.content}"
            )
            logger.error(msg)
            if response.status_code in [403, 401]:
                raise errors.Unauthorized()
            # TODO: We should probably need to manage other error states
        return response

    def is_subchannel(self, channel):
        """Return True if channel is a path to a subchannel, False otherwise. For example:

        >> is_subchannel("main")
            False
        >> is_subchannel("main/stage")
            True

        Args:
            channel (str): name of the channel

        Returns:
            (bool)"""
        return "/" in channel

    def _get_channel_url(self, channel):
        """Return a channel url based on the fact that it's a normal channel or
         a subchannel

        Args:
            channel (str): name of the channel

        Returns:
            (str) url
        """
        if self.is_subchannel(channel):
            # this is a subchannel....
            channel, subchannel = channel.split("/")
            url = join(self.urls["channels"], channel, "subchannels", subchannel)
        else:
            url = join(self.urls["channels"], channel)
        return url

    def get_channel(self, channel):
        logger.debug(f"Getting channel {channel} on {self.base_url}")
        url = self._get_channel_url(channel)
        response = requests.get(
            url,
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )
        return self._manage_response(response, f"getting channel {channel}")

    def get_channel_history(self, channel, offset=0, limit=50):
        logger.debug(f"Getting channel {channel} history {self.base_url}")
        url = join(
            self._get_channel_url(channel),
            "history?offset=%s&limit=%s" % (offset, limit),
        )
        response = requests.get(
            url,
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )
        return self._manage_response(response, f"getting channel {channel} history")

    def list_user_channels(self):
        logger.debug(f"Getting user channels from {self.base_url}")
        response = requests.get(
            self.urls["user_channels"],
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )
        return self._manage_response(response, "getting user channels")

    def list_channels(self):
        logger.debug(f"Getting channels from {self.base_url}")
        response = requests.get(
            self.urls["channels"],
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )
        return self._manage_response(response, "getting channels")

    def get_channel_subchannels(self, channel):
        logger.debug(f"Getting channel {channel} subchannels on {self.base_url}")
        url = join(self.urls["channels"], channel, "subchannels")
        response = requests.get(
            url,
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )
        return self._manage_response(response, f"getting channel {channel} subchannel")

    def create_mirror(
        self, channel, source_root, name, mode, filters, type_, cron, run_now, proxy
    ):
        url = join(self._get_channel_url(channel), "mirrors")
        mirror_details = {
            "mirror_name": name,
            "source_root": source_root,
            "mirror_mode": mode,
            "cron": cron,
            "mirror_type": type_,
            "filters": filters,
            "run_now": run_now,
        }
        if proxy:
            mirror_details["proxy"] = proxy

        resp = requests.post(
            url,
            data=json.dumps(mirror_details),
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )
        return self._manage_response(
            resp,
            f"Creating mirror {name} on channel {channel}",
            success_codes=[201],
            auth_fail_codes=[401],
        )

    def update_mirror(
        self,
        mirror_id,
        channel,
        source_root,
        name,
        mode,
        filters,
        type_,
        cron,
        run_now,
        proxy,
    ):
        url = join(self._get_channel_url(channel), "mirrors", mirror_id)
        mirror_details = {
            "mirror_name": name,
            "source_root": source_root,
            "mirror_mode": mode,
            "cron": cron,
            "mirror_type": type_,
            "filters": filters,
            "run_now": run_now,
        }
        if proxy:
            mirror_details["proxy"] = proxy

        resp = requests.put(
            url,
            data=json.dumps(mirror_details),
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )
        return self._manage_response(
            resp,
            f"Updating mirror {name} on channel {channel}",
            success_codes=[202],
            auth_fail_codes=[401],
        )

    def get_mirrors(self, channel):
        url = join(self._get_channel_url(channel), "mirrors")
        resp = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(resp, f"Getting mirrors on channel {channel}")

    def get_mirror(self, channel, mirror_name):
        url = join(self._get_channel_url(channel), "mirrors", mirror_name)
        resp = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(
            resp, f"Getting mirror {mirror_name} on channel {channel}"
        )

    def get_all_mirrors(self):
        url = self.urls["mirrors"]
        params = {"limit": 100, "offset": 0, "sort": "mirror_name"}
        resp = requests.get(
            url,
            headers=self.xauth_headers,
            verify=self.verify_ssl,
            data=json.dumps(params),
        )

        if resp.ok:
            if resp.json()["total_count"] > 100:
                params = {
                    "limit": resp.json()["total_count"],
                    "offset": 0,
                    "sort": "mirror_name",
                }
                refetch_resp = requests.get(
                    url,
                    headers=self.xauth_headers,
                    verify=self.verify_ssl,
                    data=json.dumps(params),
                )
                return self._manage_response(refetch_resp, "Getting all mirrors")

        return self._manage_response(resp, "Getting global mirrors")

    def delete_mirror(self, channel, mirror_id, mirror_name):
        url = join(self._get_channel_url(channel), "mirrors", mirror_id)
        resp = requests.delete(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(
            resp,
            f"Mirror {mirror_name} was deleted on channel {channel}",
            success_codes=[204],
        )

    # TOKEN RELATED URLS
    def _manage_response(
        self, response, action="", success_codes=None, auth_fail_codes=None
    ):
        if not success_codes:
            success_codes = [200]
        if not auth_fail_codes:
            auth_fail_codes = [401, 403]
        if response.status_code in success_codes:
            # deletes shouldn't return anythings
            if response.status_code == 204:
                return
            return response.json()
        else:
            msg = (
                f"Error {action}. "
                f"Server responded with status code {response.status_code}.\n"
                f"Error details: {response.content or None}\n"
            )
            if response.status_code >= 500:
                msg += (
                    "\nPlease verify that the Anaconda Server server is online and "
                    "that your configuration is pointing to the active ATE server.\n"
                    "If Anaconda Server is still responding with 500 errors, "
                    "please contact your system administrator.\n"
                )
            if response.status_code == 401:
                logger.debug(msg)
            if response.status_code in auth_fail_codes:
                raise errors.Unauthorized()
            raise errors.RepoCLIError(msg)

    def get_token_info(self):
        response = requests.get(
            self.urls["token_info"], headers=self.xauth_headers, verify=self.verify_ssl
        )

        if response.status_code in [200, 204]:
            return response.json()
        else:
            msg = (
                f"Error getting token info."
                f"Server responded with status code {response.status_code}.\n"
                f"Error details: {response.content}"
            )
            logger.error(msg)
            if response.status_code in [403, 401]:
                raise errors.Unauthorized()
        return {}

    def get_user_tokens(self):
        response = requests.get(
            self.urls["account_tokens"],
            headers=self.bearer_headers,
            verify=self.verify_ssl,
        )

        if response.status_code in [200, 204]:
            return response.json()
        else:
            msg = (
                f"Error getting user tokens."
                f"Server responded with status code {response.status_code}.\n"
                f"Error details: {response.content}"
            )
            logger.error(msg)
            if response.status_code in [403, 401]:
                raise errors.Unauthorized()
        return []

    def remove_user_token(self, token):
        url = join(self.urls["account_tokens"], token)
        response = requests.delete(
            url, headers=self.bearer_headers, verify=self.verify_ssl
        )
        return self._manage_response(
            response, "removing user token", success_codes=[204]
        )

    def create_user_token(self, name, expiration, scopes=None, resources=None):
        data = {
            "name": name,
            "expires_at": expiration,
        }
        if scopes:
            data["scopes"] = scopes

        if resources:
            data["resources"] = resources

        response = requests.post(
            self.urls["account_tokens"],
            data=json.dumps(data),
            headers=self.bearer_headers,
            verify=self.verify_ssl,
        )
        return self._manage_response(
            response, "creating user token", success_codes=[200]
        )

    def get_scopes(self):
        response = requests.get(
            self.urls["scopes"], headers=self.bearer_headers, verify=self.verify_ssl
        )
        return self._manage_response(response, "getting scopes")["scopes"]

    # --------
    def channel_artifacts_bulk_actions(
        self, channel, action, artifacts, target_channel=None
    ):
        url = join(self._get_channel_url(channel), "artifacts", "bulk")
        data = {"action": action, "items": artifacts}
        if target_channel:
            if "/" in target_channel:
                # this is a subchannel....
                (
                    data["target_channel"],
                    data["target_subchannel"],
                ) = target_channel.split("/")
            else:
                data["target_channel"] = target_channel

        resp = requests.put(
            url,
            json=data,
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )

        if resp.status_code == 400:
            raise errors.ChannelFrozen(channel)

        if resp.status_code == 409:
            content = json.loads(resp.content)
            raise errors.BulkActionError(content["code"], content["message"])

        return self._manage_response(
            resp,
            "%s articfacts" % action,
            success_codes=[202],
            auth_fail_codes=[401, 403, 404],
        )

    def get_channel_artifacts(self, channel, offset=None, limit=None):
        query = []
        if offset is not None:
            query.append("offset=%s" % offset)
        if limit is not None:
            query.append("limit=%s" % limit)
        if query:
            url = join(
                self._get_channel_url(channel), "artifacts?{}".format("&".join(query))
            )
        else:
            url = join(self._get_channel_url(channel), "artifacts")
        resp = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(resp, "getting artifacts")

    def get_channel_artifacts_files(
        self, channel, family=None, package=None, version=None, filename=None
    ):
        file_family_parsers = {
            "conda": self._parse_conda_file,
            "cran": self._parse_cran_file,
            "python": self._parse_python_file,
            "anaconda_project": self._parse_project_file,
            "anaconda_env": self._parse_environment_file,
            "notebook": self._parse_notebook_file,
        }
        artifact_files = []
        total_count = 0
        if package:
            packages = [package]
        else:
            # url = join(self.urls['channels'], channel, 'artifacts')
            # resp = requests.get(url, headers=self.xauth_headers)
            # data = self._manage_response(resp, "getting articfacts")
            data = self.get_channel_artifacts(channel).get("items", [])

            if family:
                packages = [
                    {"name": pkg["name"], "family": pkg["family"]}
                    for pkg in data
                    if pkg["file_count"] > 0 and pkg["family"] == family
                ]
            else:
                packages = [
                    {"name": pkg["name"], "family": pkg["family"]}
                    for pkg in data
                    if pkg["file_count"] > 0
                ]

        for package in packages:
            if family:
                url = join(
                    self._get_channel_url(channel),
                    "artifacts",
                    family,
                    package["name"],
                    "files",
                )
            else:
                url = join(
                    self._get_channel_url(channel),
                    "artifacts",
                    package["family"],
                    package["name"],
                    "files",
                )

            resp = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
            files = self._manage_response(resp, "getting articfacts")
            total_count = files["total_count"]
            for file_data in files["items"]:
                if family:
                    rec = file_family_parsers[family](
                        file_data,
                        version,
                        filename,
                    )
                else:
                    rec = file_family_parsers[file_data["family"]](
                        file_data,
                        version,
                        filename,
                    )
                if rec:
                    for cve_field in ["cve_status", "cve_score"]:
                        if cve_field in file_data:
                            rec[cve_field] = file_data[cve_field]
                    artifact_files.append(rec)

        return artifact_files, total_count

    def _parse_conda_file(self, file_data, version, filename, return_raw=False):
        meta = (
            file_data["metadata"].get("repodata_record.json")
            or file_data["metadata"]["index.json"]
        )
        # TODO: We need to improve version checking... for now it's exact match
        if version and meta["version"] != version:
            return
        if filename and meta["fn"] != filename:
            return
        if return_raw:
            rec = file_data
        else:
            rec = {"name": file_data["name"], "ckey": file_data["ckey"]}
            rec.update({key: meta.get(key, "") for key in ["version", "fn"]})
            rec["platform"] = meta.get("subdir", "")

        return rec

    def _parse_cran_file(self, file_data, version, filename, return_raw=False):
        meta = file_data["metadata"]
        fn = file_data["ckey"].split("/")[-1]

        # TODO: We need to improve version checking... for now it's exact match
        if version and meta["Version"] != version:
            return
        if filename and fn != filename:
            return
        if return_raw:
            rec = file_data
        else:
            rec = {"name": file_data["name"], "ckey": file_data["ckey"]}
            # rec.update({key: meta.get(key, "") for key in ['version', 'fn', 'platform']})
            rec.update(meta)
            rec["version"] = rec.pop("Version")
            rec["fn"] = fn
            rec["platform"] = "n/a"
        return rec

    def _parse_project_file(self, file_data, version, filename, return_raw=False):
        fn = file_data["ckey"].split("/", maxsplit=1)[-1]
        platforms = file_data["metadata"].get("platforms")
        if isinstance(platforms, list):
            platforms = ",".join(platforms)
        rec = {
            "name": file_data["name"],
            "ckey": file_data["ckey"],
            "fn": fn,
            "platform": platforms if platforms else "n/a",
            "version": None,
        }
        return rec

    def _parse_environment_file(self, file_data, version, filename, return_raw=False):
        fn = file_data["ckey"].split("/", maxsplit=1)[-1]
        rec = {
            "name": file_data["name"],
            "ckey": file_data["ckey"],
            "fn": fn,
            "platform": "n/a",
            "version": None,
        }
        return rec

    def _parse_notebook_file(self, file_data, version, filename, return_raw=False):
        fn = file_data["ckey"].split("/", maxsplit=1)[-1]
        rec = {
            "name": file_data["name"],
            "ckey": file_data["ckey"],
            "fn": fn,
            "platform": "n/a",
            "version": None,
        }
        return rec

    def _parse_python_file(self, file_data, version, filename, return_raw=False):
        meta = file_data["metadata"]
        artifact_filename = file_data["ckey"].split("/")[-1]

        if version and meta["version"] != version:
            return

        if filename and artifact_filename != filename:
            return

        if return_raw:
            rec = file_data
        else:
            rec = {"name": file_data["name"], "ckey": file_data["ckey"]}
            rec.update(meta)
            rec["version"] = rec.pop("version")
            rec["artifact_filename"] = artifact_filename
            rec["platform"] = "n/a"

        return rec

    def get_artifacts(self, query, limit=50, offset=0, sort="-download_count"):
        url = "%s?q=%s&limit=%s&offset=%s&sort=%s" % (
            self.urls["artifacts"],
            query,
            limit,
            offset,
            sort,
        )
        response = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(response, "searching artifacts")

    def get_artifact_files(self, channel, package, family, limit=100):
        url = self.urls["channels"] + "/%s/artifacts/%s/%s/files?limit=%s" % (
            channel,
            family,
            package,
            limit,
        )
        response = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(response, "getting artifact files")

    # SBOM related endpoints
    def get_sbom_download(self, channel, ckey=""):
        url = "%s/%s/%s?sbom=download" % (self.urls["repo"], channel, ckey)
        return requests.get(url=url, headers=self.xauth_headers, verify=self.verify_ssl)

    # CVE related endpoints
    def get_cves(self, offset=0, limit=50):
        url = "%s?offset=%s&limit=%s" % (self.urls["cves"], offset, limit)
        response = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(response, "getting cves")

    def get_cve(self, cve_id):
        url = "%s/%s" % (self.urls["cves"], cve_id)
        response = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(response, "getting cve id")

    def get_cve_files(self, cve_id, offset=0, limit=50):
        url = "%s/%s/files?offset=%s&limit=%s" % (
            self.urls["cves"],
            cve_id,
            offset,
            limit,
        )
        response = requests.get(url, headers=self.xauth_headers, verify=self.verify_ssl)
        return self._manage_response(response, "getting cve files")

    def _validate_channel_name(self, name: str):
        if self.is_subchannel(name):
            try:
                channel, subchannel = name.split("/")
            except ValueError:
                error_message = (
                    f"Channel name {name} is not valid. It contains more than one '/'"
                )
                logger.error(error_message)
                raise InvalidName(error_message)
            else:
                self._validate_channel_name(channel)
                self._validate_channel_name(subchannel)
            return

        if not re.match(r"^[a-z][a-z0-9_-]*$", name):
            # try to assist user by pointing to invalid letters
            invalid_chars = set(r"""!"#$%&'()*+,./:;<=>?@[\]^`{|}~""")
            invalid_letters = list(invalid_chars.intersection(set(name)))
            error_message = (
                f"Channel name contains invalid sequence {', '.join(invalid_letters)}"
            )
            logger.error(error_message)
            raise InvalidName(error_message)

    # Admin settings endpoints
    def get_system_settings(self):
        response = requests.get(
            self.urls["system_settings"], headers=self.xauth_headers
        )
        return self._manage_response(response, "getting system settings")

    def update_system_settings(self, settings):
        response = requests.put(
            self.urls["system_settings"],
            data=json.dumps(settings),
            headers=self.get_xauth_headers({"Content-Type": "application/json"}),
            verify=self.verify_ssl,
        )
        return self._manage_response(response, "updating system settings")

    def stop_mirror(self, channel, mirror_id):
        url = (
            self._get_channel_url(channel) + "/mirrors/%s/stop_mirror_sync" % mirror_id
        )

        response = requests.post(
            url,
            headers=self.xauth_headers,
            verify=self.verify_ssl,
        )
        return self._manage_response(response, "stopping mirror")

    def get_report(
        self, data_from, data_to, user_names, channel_names, file_type="json"
    ):
        url = self.urls["reports"] + "/artifact_downloads"

        data = {
            "from_date": data_from,
            "to_date": data_to,
            "download_as": file_type,
        }

        if user_names:
            data["user_names"] = user_names
        if channel_names:
            data["channel_names"] = channel_names

        response = requests.post(
            url,
            headers=self.get_xauth_headers(
                {"Content-Type": "application/json", "Accept": "application/json"}
            ),
            verify=self.verify_ssl,
            data=json.dumps(data),
        )

        if response.status_code == 401:
            raise errors.Unauthorized

        if response.status_code != 200:
            return False

        if file_type == "json":
            return response.json().get("downloaded_items")

        if file_type == "csv":
            return response.text

    def get_notebook_download_url(self, channel, notebook_name):
        return self.urls["repo"] + "/%s/jupyter/%s.ipynb" % (channel, notebook_name)

    def post_blob_cleanup(self, array_sha):
        url = self.urls["system"] + "/blob_cleanup"

        response = requests.post(
            url,
            headers=self.get_xauth_headers(
                {"Content-Type": "application/json", "Accept": "application/json"}
            ),
            verify=self.verify_ssl,
            data=json.dumps(array_sha),
        )

        if response.status_code == 504:
            raise errors.SystemDiagonseError()

        return self._manage_response(response, "blob cleanup")

    def delete_blobs(self):
        url = self.urls["system"] + "/blob_cleanup"

        response = requests.delete(
            url,
            headers=self.xauth_headers,
            verify=self.verify_ssl,
        )

        return self._manage_response(response, "deleting blobs")

    def diagnose_blobs(self, channel_names):
        url = self.urls["diagnose"] + "/blobs"

        data = {"download_as": "json", "channel_names": channel_names}

        response = requests.post(
            url,
            headers=self.xauth_headers,
            verify=self.verify_ssl,
            data=json.dumps(data),
        )

        return response
