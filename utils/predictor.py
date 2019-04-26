import os
import tensorflow as tf
import numpy as np
import csv
from utils.helpers import get_class
from datasets.MIT.utils.name_generator import NameGenerator

class Predictor(object):
    def __init__(self, config, model_dir, checkpoint=None):
        self.estimators = []
        self.label_rythm_map = {value: key for key, value in config.dataset.params["label_map"].items()}

        model = get_class(config.model.name)(config.model.hparams)
        k = len(config.dataset.params.split_ratio)
        if k == 2:
            self.estimators.append(tf.estimator.Estimator(
                model_fn=model.build_model_fn(), 
                model_dir=model_dir
            ))
        else:
            for i in range(k):
                directory = os.path.join(model_dir, "fold_{}".format(i))
                self.estimators.append(tf.estimator.Estimator(
                    model_fn=model.build_model_fn(), 
                    model_dir=directory
                ))

    def infer(self, fpath):
        """Reads example from file and makes class prediction.
        reads true_label from file name
        Args:
            fpath: path to file with example
        Returns:
            (string predicted_label, string true_label)
        """
        with open(fpath, "r", newline='') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            signal = list(reader)

        input_fn = tf.estimator.inputs.numpy_input_fn(np.array(signal, dtype="float32"), shuffle=False)
        predictions = [list(estimator.predict(input_fn))[0] for estimator in self.estimators]

        predictions = sum(np.array(p["probabilities"] for p in predictions))
        predicted_label = np.argmax(predictions)
        predicted_label = self.label_rythm_map[predicted_label]

        true_label = NameGenerator(".csv").get_metadata(os.path.basename(fpath)).rythm

        return predicted_label, true_label