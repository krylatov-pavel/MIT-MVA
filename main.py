import numpy as np
import tensorflow as tf
import os
import argparse
from utils.config import Config
from utils.helpers import get_class, avg_f1_score
from utils.dirs import create_dirs
from hooks.log_metrics import LogMetricsHook

def run_crossvalidation(config, model_dir, k):
    for i in range(k):
        directory = os.path.join(model_dir, "fold_{}".format(i))
        run_experiment(config, directory, i)

def run_experiment(config, model_dir, fold_num=None):
    model = get_class(config.model.name)(config.model.hparams)
    dataset = get_class(config.dataset.name)(config.dataset.params)

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
        max_steps=config.model.hparams.num_epochs
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
        start_delay_secs=1,  # Start evaluating after 10 sec.
        throttle_secs=1,
        hooks=hooks
    )
    
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    
    return

def evaluate_accuracy(config, model_dir):
    model = get_class(config.model.name)(config.model.hparams)
    dataset = get_class(config.dataset.name)(config.dataset.params)

    x, labels = dataset.get_eval_examples()
    y = [config.dataset.params.label_map[label] for label in labels]

    input_fn = tf.estimator.inputs.numpy_input_fn(np.array(x, dtype="float32") , shuffle=False)

    estimator = tf.estimator.Estimator(
        model_fn=model.build_model_fn(), 
        model_dir=model_dir
    )
    
    predictions = list(estimator.predict(input_fn))
    predictions = [p["class_ids"][0] for p in predictions]

    tp = np.equal(predictions, y)
    accuracy = np.count_nonzero(tp) / len(y)

    print("Evaluated accuracy: ", accuracy)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="Config file name (wihout extension)", type=str)

    parser.add_argument('--accuracy', dest='accuracy', action='store_true')
    parser.set_defaults(accuracy=False)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)

        model_dir = os.path.join(
            "data/experiments",
            config.settings.model.name.split(".")[-1],
            config.settings.experiment
        )
        create_dirs([model_dir])
        config.save(model_dir)

        if args.accuracy:
            evaluate_accuracy(config.settings, model_dir)
        else:
            k = len(config.settings.dataset.params.split_ratio)

            if k == 2:
                run_experiment(config.settings, model_dir)
            if k > 2:
                run_crossvalidation(config.settings, model_dir, k)
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
     main()