"""
Provide a method for reading in the global configuration file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import os


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_config() -> dict:
    """
    Load the global configuration file, parse it as JSON, and return
    the  contents as a Python `dict`.

    Returns:
        A Python `dict` containing the configuration file contents.
    """

    # Check if a local version of the config file exists, which takes
    # precedence over the default version in the repository
    if os.path.exists('./CONFIG.local.json'):
        file_path = './CONFIG.local.json'

    # Otherwise, check of the default config file is available
    elif os.path.exists('./CONFIG.json'):
        file_path = './CONFIG.json'

    # If no configuration is found, raise an error
    else:
        raise FileNotFoundError('Configuration file not found!')

    # Read in and parse JSON
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    return config
