# (c) Continuum Analytics, Inc. / http://continuum.io
# All Rights Reserved
#
# conda is distributed under the terms of the BSD 3-clause license.
# Consult LICENSE or http://opensource.org/licenses/BSD-3-Clause


def update_index(
    dir_paths,
    output_dir=None,
    check_md5=False,
    channel_name=None,
    subdir=None,
    threads=None,
    patch_generator=None,
    verbose=False,
    progress=False,
    current_index_versions=None,
    write_run_exports=False,
):
    import os

    from conda_index.index import update_index
    from conda_index.utils import ensure_list

    from . import yaml

    # we basically expect there to be one path now
    dir_paths = [os.path.abspath(path) for path in ensure_list(dir_paths)]

    assert (
        output_dir is None or len(dir_paths) == 1
    ), "Cannot combine output_dir with multiple paths"

    if isinstance(current_index_versions, str):
        with open(current_index_versions) as f:
            current_index_versions = yaml.safe_load(f)

    for path in dir_paths:
        update_index(
            path,
            output_dir=output_dir,
            check_md5=check_md5,
            channel_name=channel_name,
            patch_generator=patch_generator,
            threads=threads,
            verbose=verbose,
            progress=progress,
            subdirs=ensure_list(subdir),
            current_index_versions=current_index_versions,
            write_run_exports=write_run_exports,
        )
