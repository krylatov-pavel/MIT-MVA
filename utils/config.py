import json
from bunch import Bunch
import os

class Config(object):
    def __init__(self, name):
        self.settings = self.__get_config_from_json(name)

    def save(self, model_dir):
        fpath = os.path.join(model_dir, "config.json")
        with open(fpath,'w') as file:
            json.dump(self.settings, file, sort_keys=False, indent=4)

    def __get_config_from_json(self, name):
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