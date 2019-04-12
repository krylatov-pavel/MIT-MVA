import tensorflow as tf
import os
import numpy as np
import pandas as pd
from utils.dirs import create_dirs

class LogMetricsHook(tf.train.SessionRunHook):
    def __init__(self, metrics, directory, model_name):
        """Args:
            metrics: dictionary, metric_name: tensor_name
        """
        self._metrics = metrics
        self._model_name = model_name

        self._fpath = os.path.join(directory, "stats.csv")

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

        stats.to_csv(self._fpath)

    def _get_existing_stats(self):
        if os.path.exists(self._fpath):
            return pd.DataFrame.from_csv(self._fpath)
        else:
            return pd.DataFrame(columns=self._columns)