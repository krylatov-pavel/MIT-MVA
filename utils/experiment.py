import tensorflow as tf
import os
import numpy as np
from utils.helpers import get_class, avg_f1_score
from hooks.log_metrics import LogMetricsHook, plot_metrics

class Experiment():
    def __init__(self, config):
        self.config = config
        self.k = len(self.config.dataset.params.split_ratio)

    @property
    def model_dir(self):
        return os.path.join(
            "data/experiments",
            self.config.model.name.split(".")[-1],
            self.config.experiment
        )

    def run(self):
        #regular experiment
        if self.k == 2:
            self._train_model(self.model_dir)

        #k-fold crossvalidation 
        elif self.k > 2:
            for i in range(self.k):
                directory = os.path.join(self.model_dir, "fold_{}".format(i))
                self._train_model(directory, i)
            plot_metrics(self.model_dir)

    def evaluate_accuracy(self):
        def _evaluate_model(config, model_dir, fold_num=None):
            model = get_class(config.model.name)(config.model.hparams)
            dataset = get_class(config.dataset.name)(config.dataset.params)

            x, labels = dataset.get_eval_examples(fold_num)
            y = [config.dataset.params.label_map[label] for label in labels]

            input_fn = tf.estimator.inputs.numpy_input_fn(np.array(x, dtype="float32") , shuffle=False)

            estimator = tf.estimator.Estimator(
                model_fn=model.build_model_fn(), 
                model_dir=model_dir
            )

            predictions = list(estimator.predict(input_fn))
            predictions = [p["class_ids"][0] for p in predictions]

            tp = np.count_nonzero(np.equal(predictions, y))
            total = len(y)

            return tp, total

        tp, total = (0, 0)
        if self.k == 2:
            tp, total = _evaluate_model(self.config, self.model_dir)
        if self.k > 2:
            for i in range(self.k):
                directory = os.path.join(self.model_dir, "fold_{}".format(i))
                tp_curr, total_curr = _evaluate_model(self.config, directory, i)
                tp += tp_curr
                total += total_curr

        accuracy = tp / total

        print("Evaluated accuracy: ", accuracy)

    def validate_dataset(self):
        dataset = get_class(self.config.dataset.name)(self.config.dataset.params)
        if hasattr(dataset, "validate"):
            dataset.validate()
        else:
            print("This dataset object not support validation option")
    
    def _train_model(self, model_dir, fold_num=None):
        model = get_class(self.config.model.name)(self.config.model.hparams)
        dataset = get_class(self.config.dataset.name)(self.config.dataset.params)

        if hasattr(dataset, "dataset_stats"):
            dataset.dataset_stats(tf.estimator.ModeKeys.TRAIN, fold_num)
            dataset.dataset_stats(tf.estimator.ModeKeys.EVAL, fold_num)

        run_config = tf.estimator.RunConfig(
            model_dir=model_dir,
            save_summary_steps=100,
            log_step_count_steps=100,
            save_checkpoints_steps=250, #evaluation occurs after checkpoint save
            keep_checkpoint_max=3 
        )

        classifier = tf.estimator.Estimator(
            model_fn=model.build_model_fn(),
            model_dir=model_dir,
            config=run_config
        )

        train_spec = tf.estimator.TrainSpec(
            input_fn=dataset.get_input_fn(tf.estimator.ModeKeys.TRAIN, fold_num),
            max_steps=self.config.model.hparams.num_epochs
        )

        hooks = None
        if fold_num != None:
            hooks = [LogMetricsHook(
                metrics={
                    "accuracy": "accuracy/value:0",
                    "accuracy0": "accuracy_0/truediv:0",
                    "accuracy1": "accuracy_1/truediv:0"
                }, 
                directory=os.path.dirname(model_dir),
                model_name=fold_num
            )]

        eval_spec = tf.estimator.EvalSpec(
            input_fn=dataset.get_input_fn(tf.estimator.ModeKeys.EVAL, fold_num),
            steps=20,
            start_delay_secs=1,  # Start evaluating after 1 sec.
            throttle_secs=1,
            hooks=hooks
        )
        
        tf.logging.set_verbosity(tf.logging.ERROR)
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    
        return