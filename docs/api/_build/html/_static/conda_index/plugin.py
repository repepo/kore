"""
Conda command plugin, provide "conda index" command, renamed to
avoid clash with old "conda index" CLI.
"""
import conda.plugins

def command(args):
     import conda_index.cli
     return conda_index.cli.cli(prog_name="conda index", args=args)


@conda.plugins.hookimpl
def conda_subcommands():
    # hide plugin if conda-build<24.1.0
    try:
        import conda_build
        from packaging.version import parse

        if parse(conda_build.__version__) < parse("24.1.0"):
            return
    except ImportError:
        # ImportError: conda-build is not installed
        pass

    yield conda.plugins.CondaSubcommand(
        name="index",
        action=command,
        summary="Update package index metadata files."
    )
