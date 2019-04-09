import numpy as np
import tensorflow as tf
import os
import argparse
from utils.config import Config
from utils.helpers import get_class, avg_f1_score
from utils.dirs import create_dirs

def run_experiment(config, model_dir):
    model = get_class(config.model.name)(config.model.hparams)
    dataset = get_class(config.dataset.name)(config.dataset.params)

    if hasattr(dataset, "dataset_stats"):
        dataset.dataset_stats(tf.estimator.ModeKeys.TRAIN)

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=100,
        log_step_count_steps=100,
        save_checkpoints_steps=500, #evaluation occurs after checkpoint save
        keep_checkpoint_max=3 
    )

    classifier = tf.estimator.Estimator(
        model_fn=model.build_model_fn(),
        model_dir=model_dir,
        config=run_config
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=dataset.get_input_fn(tf.estimator.ModeKeys.TRAIN),
        max_steps=config.model.hparams.num_epochs
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=dataset.get_input_fn(tf.estimator.ModeKeys.EVAL),
        steps=18,
        start_delay_secs=1,  # Start evaluating after 10 sec.
        throttle_secs=1
    )
    
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    
    return

def evaluate_accuracy(config, model_dir):
    model = get_class(config.model.name)(config.model.hparams)
    dataset = get_class(config.dataset.name)(config.dataset.params)

    x, labels = dataset.get_eval_data()
    y = [config.dataset.params.label_map[label] for label in labels]

    input_fn = tf.estimator.inputs.numpy_input_fn(np.array(x) , shuffle=False)

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
    parser.add_argument("--infer", "-i", type=bool)
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

        if args.infer:
            evaluate_accuracy(config.settings, model_dir)
        else:
            run_experiment(config.settings, model_dir)
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
     main()