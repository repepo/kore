"""
Indirection to preferred yaml library.
"""

import ruamel.yaml

# matches conda.common.serialize
parser = ruamel.yaml.YAML(typ="safe", pure=True)


def safe_load(string):
    """
    Examples:
        >>> yaml_safe_load("key: value")
        {'key': 'value'}

    """
    return parser.load(string)


def determined_load(string):
    """
    Load YAML, returning {} on error.
    """

    try:
        return safe_load(string)
    except ruamel.yaml.YAMLError:
        return {}
