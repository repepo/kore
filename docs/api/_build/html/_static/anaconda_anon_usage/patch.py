# This module implements the changes to conda's context class
# needed to deploy the additional anonmyous user data. It pulls
# the token management functions themselves from the api module.

import sys

from conda.auxlib.decorators import memoizedproperty
from conda.base.context import Context, ParameterLoader, PrimitiveParameter, context

from .tokens import token_string
from .utils import _debug


def _new_user_agent(ctx):
    result = ctx._old_user_agent
    prefix = (
        getattr(Context, "checked_prefix", None) or context.target_prefix or sys.prefix
    )
    try:
        token = token_string(prefix, context.anaconda_anon_usage)
        if token:
            result += " " + token
    except Exception:  # pragma: nocover
        pass
    return result


def _new_check_prefix(prefix, json=False):
    Context.checked_prefix = prefix
    Context._old_check_prefix(prefix, json)


def _patch_check_prefix():
    _debug("Applying anaconda_anon_usage cli.install patch")
    from conda.cli import install as cli_install

    Context._old_check_prefix = cli_install.check_prefix
    cli_install.check_prefix = _new_check_prefix
    context._aau_initialized = True


def main(plugin=False):
    if hasattr(context, "_aau_initialized"):
        _debug("anaconda_anon_usage already active")
        return False
    _debug("Applying anaconda_anon_usage context patch")

    # conda.base.context.Context.user_agent
    # Adds the ident token to the user agent string
    Context._old_user_agent = Context.user_agent
    # Using a different name ensures that this is stored
    # in the cache in a different place than the original
    Context.user_agent = memoizedproperty(_new_user_agent)

    # conda.base.context.Context
    # Adds anaconda_anon_usage as a managed string config parameter
    _param = ParameterLoader(PrimitiveParameter(True))
    Context.anaconda_anon_usage = _param
    Context.parameter_names += (_param._set_name("anaconda_anon_usage"),)

    # conda.base.context.checked_prefix
    # Saves the prefix used in a conda install command
    Context.checked_prefix = None

    # conda.base.context._aau_initialized
    # This helps us determine if the patching is comlpete
    context._aau_initialized = False

    if plugin:
        # The pre-command plugin avoids the circular import
        # of conda.cli.install, so we can apply the patch now
        _patch_check_prefix()
    else:
        # We need to delay further. Schedule the patch for the
        # next time context.__init__ is called.
        _debug("Deferring anaconda_anon_usage cli.install patch")
        _old__init__ = context.__init__

        def _new_init(*args, **kwargs):
            _patch_check_prefix()
            context.__init__ = _old__init__
            _old__init__(*args, **kwargs)

        context.__init__ = _new_init

    return True
