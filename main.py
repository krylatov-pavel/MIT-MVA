import argparse
from utils.config import Config
from utils.dirs import create_dirs
from utils.experiment import Experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)

    parser.add_argument("--accuracy", dest="accuracy", action="store_true")
    parser.set_defaults(accuracy=False)

    parser.add_argument("--validate", dest="validate", action="store_true")
    parser.set_defaults(validate=False)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)

        experiment = Experiment(config.settings, config.model_dir)

        if args.accuracy:
            experiment.evaluate_accuracy()
        else:
            create_dirs([config.model_dir])
            config.save(config.model_dir)

            experiment.run()

            if args.validate:
                experiment.validate_dataset()
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()