import tensorflow as tf
from utils.config import process_config
from utils.dirs import create_dirs
from data_loader.dataset import Dataset
from models.cnn_model import CNNModel

def run_experiment(config):
    model = CNNModel(config)

    run_config = tf.estimator.RunConfig(
        model_dir=config.model_dir,
        save_summary_steps=10,
        log_step_count_steps=10,
        save_checkpoints_steps=100,
        keep_checkpoint_max=5
    )

    classifier = tf.estimator.Estimator(
        model_fn=model.build_model_fn(),
        model_dir=config.model_dir,
        config=run_config
    )

    dataset = Dataset(config)
    
    train_spec = tf.estimator.TrainSpec(
        input_fn=dataset.get_input_fn(tf.estimator.ModeKeys.TRAIN),
        max_steps=config.num_epochs
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
    config = process_config("app")

    create_dirs([config.model_dir])

    run_experiment(config)

if __name__ == "__main__":
     main()