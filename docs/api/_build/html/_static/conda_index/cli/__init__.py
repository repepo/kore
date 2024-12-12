"""
Updated command line interface for conda-index.
"""
import logging
import os.path

import click

from conda_index.index import MAX_THREADS_DEFAULT, ChannelIndex, logutil

from .. import yaml


@click.command(context_settings={"help_option_names": ['-h', '--help']})
@click.argument("dir")
@click.option("--output", help="Output repodata to given directory.")
@click.option(
    "--subdir",
    multiple=True,
    default=None,
    help="Subdir to index. Accepts multiple.",
)
@click.option(
    "-n",
    "--channel-name",
    help="Customize the channel name listed in each channel's index.html.",
)
@click.option(
    "--patch-generator",
    required=False,
    help="Path to Python file that outputs metadata patch instructions from its "
    "_patch_repodata function or a .tar.bz2/.conda file which contains a "
    "patch_instructions.json file for each subdir",
)
@click.option(
    "--channeldata/--no-channeldata",
    help="Generate channeldata.json.",
    default=False,
    show_default=True,
)
@click.option(
    "--rss/--no-rss",
    help="Write rss.xml (Only if --channeldata is enabled).",
    default=True,
    show_default=True,
)
@click.option(
    "--bz2/--no-bz2",
    help="Write repodata.json.bz2.",
    default=False,
    show_default=True,
)
@click.option(
    "--zst/--no-zst",
    help="Write repodata.json.zst.",
    default=False,
    show_default=True,
)
@click.option(
    "--run-exports/--no-run-exports",
    help="Write run_exports.json.",
    default=False,
    show_default=True,
)
@click.option(
    "--compact/--no-compact",
    help="Output JSON as one line, or pretty-printed.",
    default=True,
    show_default=True,
)
@click.option(
    "--current-index-versions-file",
    "-m",
    help="""
        YAML file containing name of package as key, and list of versions as values.  The current_index.json
        will contain the newest from this series of versions.  For example:

        python:
          - 3.8
          - 3.9

        will keep python 3.8.X and 3.9.Y in the current_index.json, instead of only the very latest python version.
        """,
)
@click.option("--threads", default=MAX_THREADS_DEFAULT, show_default=True)
@click.option(
    "--verbose",
    help="""
        Enable debug logging.
        """,
    default=False,
    is_flag=True,
)
def cli(
    dir,
    patch_generator=None,
    subdir=None,
    output=None,
    channeldata=False,
    verbose=False,
    threads=None,
    current_index_versions_file=None,
    channel_name=None,
    bz2=False,
    zst=False,
    rss=False,
    run_exports=False,
    compact=True,
):
    logutil.configure()
    if verbose:
        logging.getLogger("conda_index.index").setLevel(logging.DEBUG)

    if output:
        output = os.path.expanduser(output)

    channel_index = ChannelIndex(
        os.path.expanduser(dir),
        channel_name=channel_name,
        output_root=output,
        subdirs=subdir,
        write_bz2=bz2,
        write_zst=zst,
        threads=threads,
        write_run_exports=run_exports,
        compact_json=compact,
    )

    current_index_versions = None
    if current_index_versions_file:
        with open(current_index_versions_file) as f:
            current_index_versions = yaml.safe_load(f)

    channel_index.index(
        patch_generator=patch_generator,  # or will use outdated .py patch functions
        current_index_versions=current_index_versions,
        progress=False,  # clone is a batch job
    )

    if channeldata:  # about 2 1/2 minutes for conda-forge
        channel_index.update_channeldata(rss=rss)
