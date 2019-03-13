import tensorflow as tf
import os
import argparse
from utils.helpers import get_config_from_json, get_class, avg_f1_score
from utils.dirs import create_dirs

def run_experiment(config, model_dir):
    model = get_class(config.model.name)(config.model.hparams)
    dataset = get_class(config.dataset.name)(config.dataset.params)

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
        steps=None,
        start_delay_secs=1,  # Start evaluating after 10 sec.
        throttle_secs=1
    )
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    predict_input_fn, labels = dataset.get_predict_data()
    predictions_gen = classifier.predict(
        input_fn=predict_input_fn
    )
    predictions = [pred_dict["class_ids"][0] for pred_dict in predictions_gen]

    print("Predictions:")
    avg_f1_score(labels, predictions)
    
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="Config file name (wihout extension)", type=str)
    args = parser.parse_args()

    if args.config:
        config = get_config_from_json(args.config)

        model_dir = os.path.join("data/experiments", config.model.name.split(".")[-1], config.experiment)
        create_dirs([model_dir])

        run_experiment(config, model_dir)
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
     main()