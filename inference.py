import argparse
import os
from utils.config import Config
from utils.predictor import Predictor

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config file name (wihout extension) or full path", type=str)
    parser.add_argument("--examples", "-e", help="Path to examples directory", type=str)
    parser.add_argument("--checkpoint", "-p", help="Checkpoint number, use the last one if not specified", type=int)
    
    args = parser.parse_args()
    config = Config(args.config)

    predictor = Predictor(config.settings, config.model_dir)

    examples = [os.path.join(args.examples, f) for f in os.listdir(args.examples)]
    examples = [f for f in examples if os.path.isfile(f)]

    predictions = [predictor.infer for e in examples]

    correct = sum(int(prediction == true_label) for prediction, true_label in predictions)
    accuracy = correct / len(predictions)

    print("Accuracy: ", accuracy)

    #TODO print misclassified examples

if __name__ == "__main__":
    inference()