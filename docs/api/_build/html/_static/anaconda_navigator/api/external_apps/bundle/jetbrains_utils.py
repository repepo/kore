# -*- coding: utf-8 -*-

"""Additional utility functions to use with pycharm."""

from __future__ import annotations

__all__ = ['jetbrains_is_available', 'JetBrainsVersionChecker', 'JetBrainsDetector']

import contextlib
import functools
import json
import os
import re
import typing
from anaconda_navigator import config as navigator_config
from .. import detectors
from ..detectors import utilities as detector_utilities
from .. import exceptions
from .. import validation_utils
from . import detector_utils


JetBrainsProductCode = typing.Literal['PY', 'PC', 'PE', 'DS']

JET_BRAINS_PRODUCT_CODE_KEYS: typing.Final[typing.Tuple[JetBrainsProductCode, ...]] = (
    JetBrainsProductCode.__args__  # type: ignore
)


def merge(*args: typing.Sequence[str]) -> typing.Sequence[str]:
    """
    Merge multiple sequences into a single one.

    Each value in the output will be unique.

    Order of items will be preserved from arguments, but only first occurrence will be stored.
    """
    arg: typing.Sequence[str]
    result: typing.List[str] = []
    for arg in args:
        item: str
        for item in arg:
            if item not in result:
                result.append(item)
    return result


class ProductDetails(typing.NamedTuple):
    """Collection of custom details for a single JetBrains product."""

    osx_application_name: typing.Sequence[str] = ()
    osx_executable: typing.Sequence[str] = ()

    linux_directory_prefix: typing.Sequence[str] = ()
    linux_executable: typing.Sequence[str] = ()

    windows_directory_prefix: typing.Sequence[str] = ()
    windows_executable: typing.Sequence[str] = ()

    toolbox_directory: typing.Sequence[str] = ()
    snap_directory: typing.Sequence[str] = ()

    def merge(self, other: 'ProductDetails') -> 'ProductDetails':
        """Merge two instances into a single one containing values from both operands."""
        other_dict: typing.Mapping[str, typing.Sequence[str]] = other._asdict()  # pylint: disable=protected-access
        return ProductDetails(**{
            key: merge(value, other_dict[key])
            for key, value in self._asdict().items()  # pylint: disable=no-member
        })


_PYCHARM_BASE: ProductDetails = ProductDetails(
    osx_executable=['pycharm'],

    linux_directory_prefix=['pycharm'],
    linux_executable=['pycharm.sh'],

    windows_directory_prefix=['PyCharm'],
    windows_executable=['pycharm64.exe', 'pycharm32.exe', 'pycharm.exe'],
)
JETBRAINS_PRODUCTS: typing.Final[typing.Mapping[JetBrainsProductCode, ProductDetails]] = {
    'PY': _PYCHARM_BASE._replace(  # pylint: disable=protected-access
        osx_application_name=['PyCharm.app'],

        toolbox_directory=['PyCharm-P'],
        snap_directory=['pycharm-professional'],
    ),
    'PC': _PYCHARM_BASE._replace(  # pylint: disable=protected-access
        osx_application_name=['PyCharm CE.app'],

        toolbox_directory=['PyCharm-C'],
        snap_directory=['pycharm-community'],
    ),
    'PE': _PYCHARM_BASE._replace(  # pylint: disable=protected-access
        osx_application_name=['PyCharm Edu.app'],

        toolbox_directory=['PyCharm-E'],
        snap_directory=['pycharm-educational'],
    ),
    'DS': ProductDetails(
        osx_application_name=['DataSpell.app'],
        osx_executable=['dataspell'],

        linux_directory_prefix=['dataspell'],
        linux_executable=['dataspell.sh'],

        windows_directory_prefix=['DataSpell'],
        windows_executable=['dataspell64.exe', 'dataspell32.exe', 'dataspell.exe'],

        toolbox_directory=['DataSpell'],
        snap_directory=['dataspell'],
    ),
}


def jetbrains_is_available() -> bool:
    """Check if PyCharm should be available on a platform."""
    return navigator_config.BITS_64


class JetBrainsVersionChecker(detectors.Filter):  # pylint: disable=too-few-public-methods
    """
    Detect version of a JetBrains application.

    :param product_code: acceptable product code(s)

                         Known options:

                         - PY: PyCharm Pro
                         - PC: PyCharm Community
                         - PE: PyCharm Edu
                         - DS: DataSpell
    :param product_info_path: Path to the `product-info.json` file relative to the root of the application.
    :param build_path: Path to the `build.txt` file relative to the root of the application.
    """

    __slots__ = ('__build_path', '__product_code', '__product_info_path')

    build_pattern: typing.Final[typing.Pattern[str]] = re.compile(
        r'(?P<product>[a-zA-Z0-9]+)-(?P<build>[0-9.]+)',
    )
    known_builds: typing.Final[typing.Mapping[str, str]] = {
        '101.15': '1.1.1',
        '105.58': '1.2.1',
        '107.756': '1.5.4',
        '111.291': '2.0.2',
        '117.663': '2.5.2',
        '121.378': '2.6.3',
        '125.57': '2.7',
        '125.92': '2.7.1',
        '129.314': '2.7.2',
        '129.782': '2.7.3',
        '129.1566': '2.7.4',
        '131.19': '3.0',
        '131.339': '3.0.1',
        '131.618': '3.0.2',
        '131.849': '3.0.3',
        '133.804': '3.1',
        '133.881': '3.1.1',
        '133.1229': '3.1.2',
        '133.1347': '3.1.3',
        '133.1884': '3.1.4',
        '135.973': '3.4',
        '135.1057': '3.4.1',
        '135.1317': '3.4.2',
        '135.1318': '3.4.3',
        '135.1357': '3.4.4',
        '139.487': '4.0',
        '139.574': '4.0.1',
        '139.711': '4.0.2',
        '139.781': '4.0.3',
        '139.1001': '4.0.4',
        '139.1547': '4.0.5',
        '139.1659': '4.0.6',
        '139.1803': '4.0.7',
        '141.1116': '4.5',
        '141.1245': '4.5.1',
        '141.158': '4.5.2',
        '141.1899': '4.5.3',
        '141.2569': '4.5.4',
        '141.3058': '4.5.5',
        '143.589': '5.0',
        '143.595': '5.0.1',
        '143.1184': '5.0.2',
        '143.1559': '5.0.3',
        '143.1919': '5.0.4',
        '143.2370': '5.0.5',
        '143.2371': '5.0.6',
        '145.26': '2016.1',
        '145.598': '2016.1.1',
        '145.844': '2016.1.2',
        '145.971': '2016.1.3',
        '145.1504': '2016.1.4',
        '145.2073.10': '2016.1.5',
        '162.1237.1': '2016.2',
        '162.1628.8': '2016.2.1',
        '162.1812.1': '2016.2.2',
        '162.1967.10': '2016.2.3',
        '163.8233.8': '2016.3',
        '163.9735.8': '2016.3.1',
        '163.10154.50': '2016.3.2',
        '163.15188.4': '2016.3.3',
        '163.15529.21': '2016.3.4',
        '163.15529.24': '2016.3.5',
        '163.15529.25': '2016.3.6',
        '171.3780.115': '2017.1',
        '171.4249.47': '2017.1.2',
        '171.4424.42': '2017.1.3',
        '171.4694.38': '2017.1.4',
        '171.4694.67': '2017.1.5',
        '171.4694.79': '2017.1.6',
        '171.4694.87': '2017.1.7',
        '171.4694.94': '2017.1.8',
        '172.3317.103': '2017.2',
        '172.3544.46': '2017.2.1',
        '172.3757.67': '2017.2.2',
        '172.3968.37': '2017.2.3',
        '172.4343.24': '2017.2.4',
        '172.4574.27': '2017.2.5',
        '172.4574.33': '2017.2.6',
        '172.4574.37': '2017.2.7',
        '173.3727.137': '2017.3',
        '173.3942.36': '2017.3.1',
        '173.4127.16': '2017.3.2',
        '173.4301.16': '2017.3.3',
        '173.4674.37': '2017.3.4',
        '173.4674.54': '2017.3.5',
        '173.4674.57': '2017.3.6',
        '173.4674.62': '2017.3.7',
        '181.4203.547': '2018.1',
        '181.4445.76': '2018.1.1',
        '181.4668.75': '2018.1.2',
        '181.4892.64': '2018.1.3',
        '181.5087.37': '2018.1.4',
        '181.5540.17': '2018.1.5',
        '181.5540.34': '2018.1.6',
        '182.3684.100': '2018.2',
        '182.3911.33': '2018.2.1',
        '182.4129.34': '2018.2.2',
        '182.4323.49': '2018.2.3',
        '182.4505.26': '2018.2.4',
        '182.5107.22': '2018.2.5',
        '182.5107.44': '2018.2.6',
        '182.5107.56': '2018.2.7',
        '182.5262.4': '2018.2.8',
        '183.4284.139': '2018.3',
        '183.4588.64': '2018.3.1',
        '183.4886.43': '2018.3.2',
        '183.5153.39': '2018.3.3',
        '183.5429.31': '2018.3.4',
        '183.5912.18': '2018.3.5',
        '183.6156.13': '2018.3.6',
        '183.6156.16': '2018.3.7',
        '191.6183.50': '2019.1',
        '191.6605.12': '2019.1.1',
        '191.7141.48': '2019.1.2',
        '191.7479.30': '2019.1.3',
        '191.8026.44': '2019.1.4',
        '192.5728.105': '2019.2',
        '192.6262.63': '2019.2.1',
        '192.6603.34': '2019.2.2',
        '192.6817.19': '2019.2.3',
        '192.7142.42': '2019.2.4',
        '192.7142.56': '2019.2.5',
        '192.7142.79': '2019.2.6',
        '193.5233.109': '2019.3',
        '193.5662.61': '2019.3.1',
        '193.6015.41': '2019.3.2',
        '193.6494.30': '2019.3.3',
        '193.6911.25': '2019.3.4',
        '193.7288.30': '2019.3.5',
        '201.6668.115': '2020.1',
        '201.7223.92': '2020.1.1',
        '201.7846.77': '2020.1.2',
        '201.8538.36': '2020.1.3',
        '201.8743.11': '2020.1.4',
        '201.8743.20': '2020.1.5',
        '202.6397.98': '2020.2',
        '202.6948.78': '2020.2.1',
        '202.7319.64': '2020.2.2',
        '202.7660.27': '2020.2.3',
        '202.8194.15': '2020.2.4',
        '202.8194.22': '2020.2.5',
        '203.5981.165': '2020.3',
        '203.6682.86': '2020.3.1',
        '203.6682.179': '2020.3.2',
        '203.7148.72': '2020.3.3',
        '203.7717.65': '2020.3.4',
        '203.7717.81': '2020.3.5',
        '211.6693.115': '2021.1',
        '211.7142.13': '2021.1.1',
        '211.7442.45': '2021.1.2',
        '211.7628.24': '2021.1.3',
        '212.4746.96': '2021.2',
        '212.5080.64': '2021.2.1',
        '212.5284.44': '2021.2.2',
        '212.5457.59': '2021.2.3',
    }

    def __init__(
            self,
            product_code: typing.Union['JetBrainsProductCode', typing.Iterable['JetBrainsProductCode']],
            product_info_path: str = 'product-info.json',
            build_path: str = 'build.txt',
    ) -> None:
        """Initialize new :class:`~JetBrainsVersionChecker` instance."""
        super().__init__()

        if isinstance(product_code, str):
            product_code = (product_code,)
        else:
            product_code = tuple(product_code)

        self.__product_code: typing.Final[typing.Tuple[JetBrainsProductCode, ...]] = product_code
        self.__product_info_path: typing.Final[str] = product_info_path
        self.__build_path: typing.Final[str] = build_path

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> detectors.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        product_code_validator: validation_utils.ValueChecker
        product_code_validator = validation_utils.of_options(*JET_BRAINS_PRODUCT_CODE_KEYS)

        validation_utils.has_items(at_most=0)(args, field_name='args')

        product_code: typing.Union['JetBrainsProductCode', typing.Sequence['JetBrainsProductCode']]
        product_code = validation_utils.pop_mapping_item(kwargs, 'product_code')
        validation_utils.of_type(str, typing.Sequence)(product_code, field_name='product_code')
        if isinstance(product_code, str):
            product_code_validator(product_code, field_name='product_code')
        else:
            validation_utils.each_item(product_code_validator)(product_code, field_name='product_code')

        raw_product_info_path: typing.Union[str, typing.Sequence[str]]
        raw_product_info_path = validation_utils.pop_mapping_item(kwargs, 'product_info_path', 'product-info.json')
        with exceptions.ValidationError.with_field('product_info_path'):
            product_info_path: str = detector_utilities.parse_and_join(raw_product_info_path) or ''
            validation_utils.is_str(product_info_path, field_name='*')

        raw_build_path: typing.Union[str, typing.Sequence[str]]
        raw_build_path = validation_utils.pop_mapping_item(kwargs, 'build_path', 'build.txt')
        with exceptions.ValidationError.with_field('build_path'):
            build_path: str = detector_utilities.parse_and_join(raw_build_path) or ''
            validation_utils.is_str(build_path, field_name='*')

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls(product_code=product_code, product_info_path=product_info_path, build_path=build_path)

    def __parse_product_info(self, root: str) -> typing.Optional[str]:
        """Parse `product-info.json` file for application version."""
        with contextlib.suppress(BaseException):
            stream: typing.TextIO
            data: typing.Mapping[str, typing.Any]
            with open(os.path.join(root, self.__product_info_path), 'rt', encoding='utf-8') as stream:
                data = json.load(stream)

            if data['productCode'] in self.__product_code:
                return data['version']

        return None

    def __parse_build(self, root: str) -> typing.Optional[str]:
        """Parse `build.txt` file for application version."""
        with contextlib.suppress(BaseException):
            stream: typing.TextIO
            match: typing.Optional[typing.Match[str]]
            with open(os.path.join(root, self.__build_path), 'rt', encoding='utf-8') as stream:
                match = self.build_pattern.fullmatch(stream.read().strip())

            if match and (match.group('product') in self.__product_code):
                build: str = match.group('build')
                return self.known_builds.get(build, f'build {build}')

        return None

    def __call__(
            self,
            parent: typing.Iterator[detectors.DetectedApplication],
            *,
            context: detectors.DetectorContext,
    ) -> typing.Iterator[detectors.DetectedApplication]:
        """Iterate through detected applications."""
        application: detectors.DetectedApplication
        for application in parent:
            if not application.root:
                continue

            version: typing.Optional[str] = self.__parse_product_info(application.root)
            if version is None:
                version = self.__parse_build(application.root)

            if version is not None:
                yield application.replace(version=version)


class JetBrainsDetector(detectors.Group):  # pylint: disable=too-few-public-methods
    """Detector for Jetbrains products."""

    __slots__ = ()

    def __init__(self, *args: 'JetBrainsProductCode') -> None:
        """Initialize new :class:`~JetBrainsDetector` instance."""
        linux_toolbox_root: typing.Optional[str] = detectors.join(
            detectors.FOLDERS['linux/home'], '.local', 'share', 'JetBrains', 'Toolbox', 'apps',
        )
        osx_toolbox_root: typing.Optional[str] = detectors.join(
            detectors.FOLDERS['osx/home'], 'Library', 'Application Support', 'JetBrains', 'Toolbox', 'apps',
        )
        windows_toolbox_root: typing.Optional[str] = detectors.join(
            detectors.FOLDERS['windows/local_app_data'], 'JetBrains', 'Toolbox', 'apps',
        )

        product_details: ProductDetails = functools.reduce(
            ProductDetails.merge,
            (JETBRAINS_PRODUCTS[arg] for arg in args),
        )

        super().__init__(
            detectors.OsXOnly(
                detectors.Group(
                    detectors.CheckConfiguredRoots(),
                    detectors.CheckKnownOsXRoots(product_details.osx_application_name),
                    detectors.Group(
                        detectors.CheckKnownRoots(*(
                            detectors.join(osx_toolbox_root, toolbox_name)
                            for toolbox_name in product_details.toolbox_directory
                        )),
                        detectors.StepIntoRoot(starts_with='ch-'),
                        detectors.StepIntoRoot(reverse=True),
                        detectors.StepIntoRoot(equals=product_details.osx_application_name),

                    ),
                    detectors.AppendExecutable(*(
                        detectors.join('Contents', 'MacOS', exec_filename)
                        for exec_filename in product_details.osx_executable
                    )),
                ),

                detectors.Group(
                    detectors.CheckPATH(product_details.osx_executable),
                    detectors.AppendRoot(level=2),
                ),

                JetBrainsVersionChecker(
                    product_code=args,
                    product_info_path=detectors.join('Contents', 'Resources', 'product-info.json'),
                    build_path=detectors.join('Contents', 'Resources', 'build.txt'),
                ),
            ),

            detectors.LinuxOnly(
                detectors.Group(
                    detectors.CheckConfiguredRoots(),
                    detectors.Group(
                        detectors.CheckKnownRoots(
                            detectors.join(detectors.FOLDERS['linux/root'], 'opt'),
                        ),
                        detectors.StepIntoRoot(starts_with=product_details.linux_directory_prefix, reverse=True),
                    ),
                    detectors.Group(
                        detectors.CheckKnownRoots(*(
                            detectors.join(linux_toolbox_root, toolbox_name)
                            for toolbox_name in product_details.toolbox_directory
                        )),
                        detectors.StepIntoRoot(starts_with='ch-'),
                        detectors.StepIntoRoot(reverse=True),
                    ),
                    detectors.CheckKnownRoots(*(
                        detectors.join(snap_root, snap_name, 'current')
                        for snap_name in product_details.snap_directory
                        for snap_root in (
                            detectors.FOLDERS['linux/snap_primary'],
                            detectors.FOLDERS['linux/snap_secondary'],
                        )
                    )),
                    detectors.AppendExecutable(*(
                        detectors.join('bin', exec_filename)
                        for exec_filename in product_details.linux_executable
                    )),
                ),
                detectors.Group(
                    detectors.CheckPATH(product_details.linux_executable),
                    detectors.AppendRoot(level=1),
                ),
                JetBrainsVersionChecker(
                    product_code=args,
                ),
            ),

            detectors.WindowsOnly(
                detectors.Group(
                    detectors.CheckConfiguredRoots(),
                    detectors.Group(
                        detectors.CheckKnownRoots(
                            detectors.join(detectors.FOLDERS['windows/program_files_x64'], 'JetBrains'),
                            detectors.join(detectors.FOLDERS['windows/program_files_x86'], 'JetBrains'),
                        ),
                        detectors.StepIntoRoot(starts_with=product_details.windows_directory_prefix, reverse=True),

                    ),
                    detectors.Group(
                        detectors.CheckKnownRoots(*(
                            detectors.join(windows_toolbox_root, toolbox_name)
                            for toolbox_name in product_details.toolbox_directory
                        )),
                        detectors.StepIntoRoot(starts_with='ch-'),
                        detectors.StepIntoRoot(reverse=True),
                    ),
                    detectors.AppendExecutable(*(
                        detectors.join('bin', exec_filename)
                        for exec_filename in product_details.windows_executable
                    )),
                ),

                detectors.Group(
                    detectors.CheckPATH(product_details.windows_executable),
                    detectors.AppendRoot(level=1),
                ),

                detectors.Group(
                    detectors.CheckExecutableInWindowsRegistry(*(
                        detectors.RegistryKey(
                            root='HKEY_CLASSES_ROOT',
                            key=f'Applications\\{exec_file_name}\\shell\\open\\command',
                            converter=detector_utils.extract_app_from_command,
                        )
                        for exec_file_name in product_details.windows_executable
                    )),
                    detectors.AppendRoot(level=1),
                ),
                JetBrainsVersionChecker(
                    product_code=args,
                ),
            ),
        )

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> detectors.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        validation_utils.each_item(
            validation_utils.is_str,
            validation_utils.of_options(*JET_BRAINS_PRODUCT_CODE_KEYS),
        )(args, field_name='args')

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls(*args)
