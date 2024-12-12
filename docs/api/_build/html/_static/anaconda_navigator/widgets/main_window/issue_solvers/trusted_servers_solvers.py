# -*- coding: utf-8 -*-

"""Extra solvers to migrate trusted_servers to domain-only format."""

from __future__ import annotations

__all__ = ()

from anaconda_navigator.config import CONF
from anaconda_navigator.utils import url_utils
from . import core


@core.CONFIGURATION_POOL.register
def migrate_trusted_servers_to_domains(context: core.ConfigurationContext) -> None:  # pylint: disable=unused-argument
    """Convert complete urls in `trusted_servers` to only domains."""
    trusted_servers: list[str] = CONF.get('ssl', 'trusted_servers', [])
    if trusted_servers:
        trusted_servers = [url_utils.netloc(item) for item in trusted_servers]
        CONF.set('ssl', 'trusted_servers', trusted_servers)
