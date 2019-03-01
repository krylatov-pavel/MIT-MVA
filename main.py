from utils.config import process_config
from data_loader.dataset import Dataset

def main():
    config = process_config("app")
    dataset = Dataset(config)

    print(len(dataset.train["x"][0]))

if __name__ == "__main__":
    main()