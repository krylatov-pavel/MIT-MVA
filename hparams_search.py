import argparse
import os
import numpy as np
import pandas as pd
from utils.mutable_config import MutableConfig
from utils.config import Config
from utils.experiment import Experiment
from utils.dirs import create_dirs, clear_dir
from hooks.log_metrics import max_accuracy

def iteration_name_generator(num, directory):
    if os.path.exists(directory):
        existing_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    else:
        existing_names = []

    i = 0
    while num > 0:
        if not str(i) in existing_names:
            num -= 1
            yield str(i)
        i +=1

def search_stats(directory):
    iteration_dirs = [os.path.join(directory, d) for d in os.listdir(directory)]
    iteration_dirs = [d for d in iteration_dirs if os.path.isdir(d)]
    configs = [os.path.join(d, "config.json") for d in iteration_dirs]
    configs = [c for c in configs if os.path.exists(c)]
    configs = [Config(c) for c in configs]

    data_row = {
        "number": [c.settings.iteration for c in configs],
        "accuracy": [max_accuracy(c.model_dir) for c in configs],
        "learning_rate": [c.settings.model.hparams.learning_rate for c in configs],
        "conv_layers": [c.settings.model.hparams.conv_layers for c in configs],
        "filters_num": [c.settings.model.hparams.filters_num for c in configs],
        "dense_layers": [c.settings.model.hparams.dense_layers for c in configs],
        "dense_units": [c.settings.model.hparams.dense_units for c in configs],
        "dropout_rate": [c.settings.model.hparams.dropout_rate for c in configs],
        "slice_window": [c.settings.dataset.params.slice_window for c in configs],
        "include_augmented": [c.settings.dataset.params.include_augmented for c in configs]
    }

    stats = pd.DataFrame(data_row, columns=list(data_row.keys()))
    stats = stats.sort_values(by=["accuracy"], ascending=False)

    print(stats)

def random_search():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config file name (wihout extension)", type=str)
    parser.add_argument("--iterations", "-i", help="Number of search iterations", type=int)
    parser.add_argument("--stats", dest="stats", action="store_true")
    parser.set_defaults(stats=False)

    args = parser.parse_args()

    if args.config:
        config = MutableConfig(args.config)

        if args.stats:
            search_stats(os.path.dirname(config.model_dir))
        else:
            best_result = [0, 0]

            for iteration in iteration_name_generator(args.iterations, os.path.dirname(config.model_dir)):
                print("search iteration ", iteration)

                settings = config.mutate(iteration)
                experiment = Experiment(settings, config.model_dir)

                create_dirs([config.model_dir])
                config.save(config.model_dir)

                experiment.run()

                accuracy = max_accuracy(config.model_dir)
                if accuracy > best_result[0]:
                    best_result[0] = accuracy
                    best_result[1] = iteration
            
            print("Best accuracy: {:.3f} on iteration {}".format(*best_result))
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    random_search()