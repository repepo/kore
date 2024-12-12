# This file is generated by SciPy's build process
# It contains system_info results at the time of building this package.
from enum import Enum

__all__ = ["show"]
_built_with_meson = True


class DisplayModes(Enum):
    stdout = "stdout"
    dicts = "dicts"


def _cleanup(d):
    """
    Removes empty values in a `dict` recursively
    This ensures we remove values that Meson could not provide to CONFIG
    """
    if isinstance(d, dict):
        return { k: _cleanup(v) for k, v in d.items() if v != '' and _cleanup(v) != '' }
    else:
        return d


CONFIG = _cleanup(
    {
        "Compilers": {
            "c": {
                "name": "gcc",
                "linker": "ld.bfd",
                "version": "11.2.0",
                "commands": "/croot/scipy_1710947333060/_build_env/bin/x86_64-conda-linux-gnu-cc",
                "args": "-march=nocona, -mtune=haswell, -ftree-vectorize, -fPIC, -fstack-protector-strong, -fno-plt, -O2, -ffunction-sections, -pipe, -isystem, /home/jorgemc/anaconda3/include, -fdebug-prefix-map=/croot/scipy_1710947333060/work=/usr/local/src/conda/scipy-1.12.0, -fdebug-prefix-map=/home/jorgemc/anaconda3=/usr/local/src/conda-prefix, -DNDEBUG, -D_FORTIFY_SOURCE=2, -O2, -isystem, /home/jorgemc/anaconda3/include",
                "linker args": "-Wl,-O2, -Wl,--sort-common, -Wl,--as-needed, -Wl,-z,relro, -Wl,-z,now, -Wl,--disable-new-dtags, -Wl,--gc-sections, -Wl,-rpath,/home/jorgemc/anaconda3/lib, -Wl,-rpath-link,/home/jorgemc/anaconda3/lib, -L/home/jorgemc/anaconda3/lib, -march=nocona, -mtune=haswell, -ftree-vectorize, -fPIC, -fstack-protector-strong, -fno-plt, -O2, -ffunction-sections, -pipe, -isystem, /home/jorgemc/anaconda3/include, -fdebug-prefix-map=/croot/scipy_1710947333060/work=/usr/local/src/conda/scipy-1.12.0, -fdebug-prefix-map=/home/jorgemc/anaconda3=/usr/local/src/conda-prefix, -DNDEBUG, -D_FORTIFY_SOURCE=2, -O2, -isystem, /home/jorgemc/anaconda3/include",
            },
            "cython": {
                "name": "cython",
                "linker": "cython",
                "version": "3.0.8",
                "commands": "cython",
                "args": "",
                "linker args": "",
            },
            "c++": {
                "name": "gcc",
                "linker": "ld.bfd",
                "version": "11.2.0",
                "commands": "/croot/scipy_1710947333060/_build_env/bin/x86_64-conda-linux-gnu-c++",
                "args": "-fvisibility-inlines-hidden, -std=c++17, -fmessage-length=0, -march=nocona, -mtune=haswell, -ftree-vectorize, -fPIC, -fstack-protector-strong, -fno-plt, -O2, -ffunction-sections, -pipe, -isystem, /home/jorgemc/anaconda3/include, -fdebug-prefix-map=/croot/scipy_1710947333060/work=/usr/local/src/conda/scipy-1.12.0, -fdebug-prefix-map=/home/jorgemc/anaconda3=/usr/local/src/conda-prefix, -DNDEBUG, -D_FORTIFY_SOURCE=2, -O2, -isystem, /home/jorgemc/anaconda3/include",
                "linker args": "-Wl,-O2, -Wl,--sort-common, -Wl,--as-needed, -Wl,-z,relro, -Wl,-z,now, -Wl,--disable-new-dtags, -Wl,--gc-sections, -Wl,-rpath,/home/jorgemc/anaconda3/lib, -Wl,-rpath-link,/home/jorgemc/anaconda3/lib, -L/home/jorgemc/anaconda3/lib, -fvisibility-inlines-hidden, -std=c++17, -fmessage-length=0, -march=nocona, -mtune=haswell, -ftree-vectorize, -fPIC, -fstack-protector-strong, -fno-plt, -O2, -ffunction-sections, -pipe, -isystem, /home/jorgemc/anaconda3/include, -fdebug-prefix-map=/croot/scipy_1710947333060/work=/usr/local/src/conda/scipy-1.12.0, -fdebug-prefix-map=/home/jorgemc/anaconda3=/usr/local/src/conda-prefix, -DNDEBUG, -D_FORTIFY_SOURCE=2, -O2, -isystem, /home/jorgemc/anaconda3/include",
            },
            "fortran": {
                "name": "gcc",
                "linker": "ld.bfd",
                "version": "11.2.0",
                "commands": "/croot/scipy_1710947333060/_build_env/bin/x86_64-conda-linux-gnu-gfortran",
                "args": "-fopenmp, -march=nocona, -mtune=haswell, -ftree-vectorize, -fPIC, -fstack-protector-strong, -fno-plt, -O2, -ffunction-sections, -pipe, -isystem, /home/jorgemc/anaconda3/include, -fdebug-prefix-map=/croot/scipy_1710947333060/work=/usr/local/src/conda/scipy-1.12.0, -fdebug-prefix-map=/home/jorgemc/anaconda3=/usr/local/src/conda-prefix",
                "linker args": "-Wl,-O2, -Wl,--sort-common, -Wl,--as-needed, -Wl,-z,relro, -Wl,-z,now, -Wl,--disable-new-dtags, -Wl,--gc-sections, -Wl,-rpath,/home/jorgemc/anaconda3/lib, -Wl,-rpath-link,/home/jorgemc/anaconda3/lib, -L/home/jorgemc/anaconda3/lib, -fopenmp, -march=nocona, -mtune=haswell, -ftree-vectorize, -fPIC, -fstack-protector-strong, -fno-plt, -O2, -ffunction-sections, -pipe, -isystem, /home/jorgemc/anaconda3/include, -fdebug-prefix-map=/croot/scipy_1710947333060/work=/usr/local/src/conda/scipy-1.12.0, -fdebug-prefix-map=/home/jorgemc/anaconda3=/usr/local/src/conda-prefix",
            },
            "pythran": {
                "version": "0.15.0",
                "include directory": r"../../_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_p/lib/python3.11/site-packages/pythran"
            },
        },
        "Machine Information": {
            "host": {
                "cpu": "x86_64",
                "family": "x86_64",
                "endian": "little",
                "system": "linux",
            },
            "build": {
                "cpu": "x86_64",
                "family": "x86_64",
                "endian": "little",
                "system": "linux",
            },
            "cross-compiled": bool("False".lower().replace('false', '')),
        },
        "Build Dependencies": {
            "blas": {
                "name": "openblas",
                "found": bool("True".lower().replace('false', '')),
                "version": "0.3.21",
                "detection method": "pkgconfig",
                "include directory": r"/home/jorgemc/anaconda3/include",
                "lib directory": r"/home/jorgemc/anaconda3/lib",
                "openblas configuration": "USE_64BITINT= DYNAMIC_ARCH=1 DYNAMIC_OLDER= NO_CBLAS= NO_LAPACK=0 NO_LAPACKE= NO_AFFINITY=1 USE_OPENMP=0 PRESCOTT MAX_THREADS=128",
                "pc file directory": r"/home/jorgemc/anaconda3/lib/pkgconfig",
            },
            "lapack": {
                "name": "openblas",
                "found": bool("True".lower().replace('false', '')),
                "version": "0.3.21",
                "detection method": "pkgconfig",
                "include directory": r"/home/jorgemc/anaconda3/include",
                "lib directory": r"/home/jorgemc/anaconda3/lib",
                "openblas configuration": "USE_64BITINT= DYNAMIC_ARCH=1 DYNAMIC_OLDER= NO_CBLAS= NO_LAPACK=0 NO_LAPACKE= NO_AFFINITY=1 USE_OPENMP=0 PRESCOTT MAX_THREADS=128",
                "pc file directory": r"/home/jorgemc/anaconda3/lib/pkgconfig",
            },
            "pybind11": {
                "name": "pybind11",
                "version": "2.10.4",
                "detection method": "pkgconfig",
                "include directory": r"/home/jorgemc/anaconda3/include",
            },
        },
        "Python Information": {
            "path": r"/home/jorgemc/anaconda3/bin/python",
            "version": "3.11",
        },
    }
)


def _check_pyyaml():
    import yaml

    return yaml


def show(mode=DisplayModes.stdout.value):
    """
    Show libraries and system information on which SciPy was built
    and is being used

    Parameters
    ----------
    mode : {`'stdout'`, `'dicts'`}, optional.
        Indicates how to display the config information.
        `'stdout'` prints to console, `'dicts'` returns a dictionary
        of the configuration.

    Returns
    -------
    out : {`dict`, `None`}
        If mode is `'dicts'`, a dict is returned, else None

    Notes
    -----
    1. The `'stdout'` mode will give more readable
       output if ``pyyaml`` is installed

    """
    if mode == DisplayModes.stdout.value:
        try:  # Non-standard library, check import
            yaml = _check_pyyaml()

            print(yaml.dump(CONFIG))
        except ModuleNotFoundError:
            import warnings
            import json

            warnings.warn("Install `pyyaml` for better output", stacklevel=1)
            print(json.dumps(CONFIG, indent=2))
    elif mode == DisplayModes.dicts.value:
        return CONFIG
    else:
        raise AttributeError(
            f"Invalid `mode`, use one of: {', '.join([e.value for e in DisplayModes])}"
        )
