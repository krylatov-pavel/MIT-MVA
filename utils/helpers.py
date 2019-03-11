import json
from bunch import Bunch
import os

def _dict_to_bunch(dictionary):
    bunch = Bunch(dictionary)

    for key in bunch.keys():
        if isinstance(bunch[key], dict):
            bunch[key] = _dict_to_bunch(bunch[key])

    return bunch


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
    #config = Bunch(config_dict)
    config = _dict_to_bunch(config_dict)

    return config

def get_class(name):
    parts = name.split(".")
    module_name = ".".join(parts[:-1])

    module = __import__(module_name)

    for constructor in parts[1:]:
        module = getattr(module, constructor)
    
    return module