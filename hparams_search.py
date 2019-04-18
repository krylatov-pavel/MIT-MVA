import argparse
import os
from utils.mutable_config import MutableConfig
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

def random_search():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config file name (wihout extension)", type=str)
    parser.add_argument("--iterations", "-i", help="Number of search iterations", type=int)

    args = parser.parse_args()

    if args.config:
        config = MutableConfig(args.config)
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