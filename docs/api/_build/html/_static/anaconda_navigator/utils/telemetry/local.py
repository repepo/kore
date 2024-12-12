# -*- coding: utf8 -*-

"""Navigator-specific telemetry configuration."""

from __future__ import annotations

__all__ = ['ANALYTICS']

import typing

from anaconda_navigator import config
from anaconda_navigator.utils import singletons
from . import basic_pool
from . import core
from . import heap_provider


class NavigatorAnalyticsSingleton(singletons.Singleton[core.Analytics]):  # pylint: disable=too-few-public-methods
    """Customized singleton of the configured :class:`~core.Analytics` instance."""

    __slots__ = ()

    def _prepare(self) -> core.Analytics:
        """Configure and initialize :class:`~core.Analytics` instance."""
        session_identity: str | None = None
        user_identity: str | None

        try:
            import anaconda_anon_usage.tokens as anon_tokens  # pylint: disable=import-outside-toplevel
            session_identity = anon_tokens.token_string()
            user_identity = anon_tokens.client_token()
        except Exception:  # pylint: disable=broad-exception-caught
            user_identity = config.CONF.get('main', 'identity')

        context: core.Context = core.Context(
            session_identity=session_identity,
            user_identity=user_identity,
        )
        config.CONF.set('main', 'identity', context.user_identity)

        return core.Analytics(
            providers=[
                heap_provider.HeapProvider(context=context, app_id='4084878704'),
            ],
            pool=basic_pool.BasicPool(),
        )

    def _release(self) -> None:
        """Destroy singleton instance when :meth:`~Singleton.reset` is called."""
        self.instance.stop()


ANALYTICS: typing.Final[singletons.Singleton[core.Analytics]] = NavigatorAnalyticsSingleton()
