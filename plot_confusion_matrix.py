import argparse
from utils.config import Config
from utils.experiment import Experiment
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)
    parser.add_argument("--checkpoint", help="Checkpoint number", type=int)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
        experiment = Experiment(config.settings, config.model_dir)
        
        labels = {config.settings.dataset.params.label_map[key]: key for key in config.settings.dataset.params.label_map}
        labels = [labels[i] for i in range(len(labels))]
        cm = experiment.confusion_matrix(args.checkpoint)
                
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True)
        plt.show()
        
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()