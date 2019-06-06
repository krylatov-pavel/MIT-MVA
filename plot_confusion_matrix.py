import argparse
from utils.config import Config
from utils.experiment import Experiment
import seaborn as sn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

LEGEND = {
    "N": "Normal sinus rhythm",
    "R": "Right bundle branch block beat",
    "B": "Ventricular bigeminy",
    "SBR": "Sinus bradycardia",
    "AFIB": "Atrial fibrillation"
}

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
        sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')

        plt.ylabel("Predicted")
        plt.xlabel("Actual")

        fpath = os.path.join(config.model_dir, "confusion_matrix.png")
        plt.savefig(fpath)
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()