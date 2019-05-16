import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.dirs import create_dirs
from utils.helpers import unzip_list

_FNAME = "stats.csv"

class LogMetricsHook(tf.train.SessionRunHook):
    def __init__(self, metrics, directory, model_name):
        """Args:
            metrics: dictionary, metric_name: tensor_name
        """
        self._metrics = metrics
        self._model_name = model_name

        self._fpath = os.path.join(directory, _FNAME)

        if not os.path.exists(directory):
            create_dirs([directory])

        self._columns = ["model", "step"] + [name for name in self._metrics]

    def end(self, session):
        graph = tf.get_default_graph()
        
        step = session.run(tf.train.get_global_step(graph))

        data_row = {
            "model": [self._model_name],
            "step": [step]
        }
        
        for metric_name, tensor_name in self._metrics.items():
            tensor = graph.get_tensor_by_name(tensor_name)
            value = session.run(tensor)
            data_row[metric_name] = [value]

        df = pd.DataFrame(data_row, columns=self._columns)

        stats = self._get_existing_stats()
        stats = stats.append(df)

        stats.to_csv(self._fpath, index=False)

    def _get_existing_stats(self):
        if os.path.exists(self._fpath):
            return pd.read_csv(self._fpath)
        else:
            return pd.DataFrame(columns=self._columns)

def plot_metrics(model_dir):
    fpath = os.path.join(model_dir, _FNAME)
    df = pd.read_csv(fpath)
    steps = df.groupby("step")

    metrics = [("accuracy", "b-"), ("accuracy0", "y-"), ("accuracy1", "g-")]
    plots = []

    fig = plt.figure()
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.0)
    
    for metric, color in metrics:
        m_mean = steps[metric].agg(np.mean)
        x, y = unzip_list(m_mean.iteritems())

        if metric == "accuracy":
            idx_max = np.argmax(y)
            step = x[idx_max]
            max_accuracy = y[idx_max]
            plt.text(0.05, 0.05, "max accuracy {:.3f} on step {}".format(max_accuracy, step))

        plot, = plt.plot(x, y, color, label=metric)
        plots.append(plot)

    plt.legend(plots, [name for name, _ in metrics])
    plt.legend(loc="upper left")

    fig.savefig(os.path.join(model_dir, "plot.png"))
    plt.close(fig)

def max_accuracy(model_dir):
    fpath = os.path.join(model_dir, _FNAME)
    df = pd.read_csv(fpath)
    steps = df.groupby("step")

    mean = steps["accuracy"].agg(np.mean)

    max_accuracy = np.max(mean)
    
    return max_accuracy