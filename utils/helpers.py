import json
from bunch import Bunch
import os

def get_config_from_json(name):
    def merge(a, b):
        for key in b.keys():
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    a[key] = merge(a[key], b[key])
                else:
                    a[key] = b[key]
            else:
                a[key] = b[key]     
        
        return a

    def dict_to_bunch(dictionary):
        bunch = Bunch(dictionary)

        for key in bunch.keys():
            if isinstance(bunch[key], dict):
                bunch[key] = dict_to_bunch(bunch[key])

        return bunch

    config_dict = {}
    parts = name.split(".")
    
    for i in range(1, len(parts) + 1):
        fname = ".".join(parts[:i]) + ".json"
        fpath = os.path.join("configs",  fname)

        if os.path.lexists(fpath):
            # parse the configurations from the config json file provided
            with open(fpath, 'r') as config_file:
                config_dict = merge(config_dict, json.load(config_file))

    # convert the dictionary to a namespace using bunch lib
    #config = Bunch(config_dict)
    config = dict_to_bunch(config_dict)

    return config

def get_class(name):
    parts = name.split(".")
    module_name = ".".join(parts[:-1])

    module = __import__(module_name)

    for constructor in parts[1:]:
        module = getattr(module, constructor)
    
    return module