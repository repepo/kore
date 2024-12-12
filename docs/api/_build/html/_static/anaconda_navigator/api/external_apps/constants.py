# -*- coding: utf-8 -*-

"""Constants for third-party applications."""

__all__ = ['EMPTY_PATCH', 'MANUAL_PATCHES']

import typing
from anaconda_navigator.static import images
from . import base


EMPTY_PATCH: typing.Final[base.AppPatch] = base.AppPatch()
MANUAL_PATCHES: typing.Final[typing.Mapping[str, base.AppPatch]] = {
    'anaconda-fusion': base.AppPatch(
        description=(
            'Integration between Excel ® and Anaconda via Notebooks. Run data science functions, interact with results '
            'and create advanced visualizations in a code-free app inside Excel.'
        ),
    ),
    'anaconda-mosaic': base.AppPatch(
        description=(
            'Interactive exploration of larger than memory datasets. Create data sources, perform transformations and '
            'combinations.'
        ),
    ),
    'anacondafusion': base.AppPatch(
        is_available=False,
    ),
    'console_shortcut': base.AppPatch(
        description='Run a cmd.exe terminal with your current environment from Navigator activated',
        display_name='CMD.exe Prompt',
    ),
    'glueviz': base.AppPatch(
        display_name='Glueviz',
        description=(
            'Multidimensional data visualization across files. Explore relationships within and among related datasets.'
        ),
        image_path=images.GLUEVIZ_ICON_1024_PATH,
    ),
    'ipython-notebook': base.AppPatch(
        image_path=images.NOTEBOOK_ICON_1024_PATH,
        is_available=False,
    ),
    'ipython-qtconsole': base.AppPatch(
        image_path=images.QTCONSOLE_ICON_1024_PATH,
        is_available=False,
    ),
    'jupyterlab': base.AppPatch(
        display_name='JupyterLab',
        description=(
            'An extensible environment for interactive and reproducible computing, based on the Jupyter Notebook and '
            'Architecture.'
        ),
        image_path=images.JUPYTERLAB_ICON_1024_PATH,
    ),
    'notebook': base.AppPatch(
        display_name='Notebook',
        description=(
            'Web-based, interactive computing notebook environment. Edit and run human-readable docs while describing '
            'the data analysis.'
        ),
        image_path=images.NOTEBOOK_ICON_1024_PATH,
    ),
    'orange-app': base.AppPatch(
        description=(
            'Component based data mining framework. Data visualization and data analysis for novice and expert. '
            'Interactive workflows with a large toolbox.'
        ),
        image_path=images.ORANGE_ICON_1024_PATH,
    ),
    'orange3': base.AppPatch(
        display_name='Orange 3',
        description=(
            'Component based data mining framework. Data visualization and data analysis for novice and expert. '
            'Interactive workflows with a large toolbox.'
        ),
        image_path=images.ORANGE_ICON_1024_PATH,
    ),
    'powershell_shortcut': base.AppPatch(
        description='Run a Powershell terminal with your current environment from Navigator activated',
        display_name='Powershell Prompt',
    ),
    'pyvscode': base.AppPatch(
        image_path=images.VSCODE_ICON_1024_PATH,
    ),
    'qt3dstudio': base.AppPatch(
        description=(
            'Rapidly build and prototype high quality 2D and 3D user interfaces using the built-in material and '
            'effects library or import your own design assets.'
        ),
        image_path=images.QTCREATOR_ICON_1024_PATH,
    ),
    'qtconsole': base.AppPatch(
        display_name='Qt Console',
        description=(
            'PyQt GUI that supports inline figures, proper multiline editing with syntax highlighting, graphical '
            'calltips, and more.'
        ),
        image_path=images.QTCONSOLE_ICON_1024_PATH,
    ),
    'qtcreator': base.AppPatch(
        description='Cross platform integrated development environment (IDE) to create C++ and QML applications.',
        image_path=images.QTCREATOR_ICON_1024_PATH,
    ),
    'rodeo': base.AppPatch(
        description=(
            'A browser-based IDE for data science with python. Includes autocomplete, syntax highlighting, IPython '
            'support.'
        ),
        image_path=images.RODEO_ICON_1024_PATH,
    ),
    'rstudio': base.AppPatch(
        display_name='RStudio',
        description=(
            'A set of integrated tools designed to help you be more productive with R. Includes R essentials and '
            'notebooks.'
        ),
        image_path=images.RSTUDIO_ICON_1024_PATH,
    ),
    'spyder': base.AppPatch(
        display_name='Spyder',
        description=(
            'Scientific PYthon Development EnviRonment. Powerful Python IDE with advanced editing, interactive '
            'testing, debugging and introspection features'
        ),
        image_path=images.SPYDER_ICON_1024_PATH,
    ),
    'spyder-app': base.AppPatch(
        image_path=images.SPYDER_ICON_1024_PATH,
        is_available=False,
    ),
    'veusz': base.AppPatch(
        description=(
            'Veusz is a GUI scientific plotting and graphing package. It is designed to produce publication-ready '
            'Postscript or PDF output.'
        ),
        image_path=images.VEUSZ_ICON_1024_PATH,
    ),
}
