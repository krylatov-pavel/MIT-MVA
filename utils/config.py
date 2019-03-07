import json
from bunch import Bunch
import os


def get_config_from_json(name):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    fpath = os.path.join("configs", name + ".json")

    # parse the configurations from the config json file provided
    with open(fpath, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(name):
    config, _ = get_config_from_json(name)
    config.db_name = "vfdb"
    config.bypass_cache = False
    config.model_dir = os.path.join("data/experiments", config.model_name, config.exp_name)
    return config