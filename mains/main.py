from utils.config import process_config
from data_loader.dataset import Dataset

def main():
    config, _ = process_config("app")
    dataset = Dataset(config)

    print(dataset.train["y"])