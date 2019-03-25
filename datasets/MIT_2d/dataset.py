import pandas as pd
from datasets.base_dataset import BaseDataset
from datasets.MIT_2d.database_provider import DatabaseProvider
from datasets.MIT_2d.ecg import ECG

class Dataset(BaseDataset):
    def __init__(self, db_name):
        self.db_name = db_name