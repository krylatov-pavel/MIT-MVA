import tensorflow as tf
from utils.config import process_config
from utils.dirs import create_dirs
from data_loader.dataset import Dataset
from models.cnn_model import CNNModel

def run_experiment(config):
    model = CNNModel(config)

    run_config = tf.estimator.RunConfig(
        model_dir=config.model_dir,
        save_summary_steps=2,
        log_step_count_steps=2,
        save_checkpoints_steps=10,
    )

    classifier = tf.estimator.Estimator(
        model_fn=model.build_model_fn(),
        model_dir=config.model_dir,
        config=run_config
    )

    dataset = Dataset(config)

    print("start learning")

    classifier.train(dataset.get_input_fn(tf.estimator.ModeKeys.TRAIN), max_steps=config.num_epochs)

    print("completed\nstart evaluating")

    classifier.evaluate(dataset.get_input_fn(tf.estimator.ModeKeys.EVAL))

    print("completed")
    
    return

def main():
    config = process_config("app")

    create_dirs([config.model_dir])

    run_experiment(config)

if __name__ == "__main__":
     main()