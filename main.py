import tensorflow as tf
import os
from utils.helpers import get_config_from_json, get_class
from utils.dirs import create_dirs

def run_experiment(config, model_dir):
    model = get_class(config.model.name)(config.model.hparams)
    dataset = get_class(config.dataset.name)(config.dataset.params)

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=10,
        log_step_count_steps=10,
        save_checkpoints_steps=100,
        keep_checkpoint_max=5
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
        steps=None,
        start_delay_secs=10,  # Start evaluating after 10 sec.
        throttle_secs=30
    )

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    print("completed")
    
    return

def main():
    config = get_config_from_json("CNNModel")

    model_dir = os.path.join("data/experiments", config.model.name.split(".")[-1], config.experiment)
    create_dirs([model_dir])

    run_experiment(config, model_dir)

if __name__ == "__main__":
     main()