import logging
from logging.handlers import RotatingFileHandler
import datetime
import sys
import os
from dotenv import load_dotenv

load_dotenv()

K = os.getenv("K")

time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

def setup_logger(log_file:str=f'./parafac-log/training-{time}.log', console_level=logging.INFO, file_level=logging.DEBUG) -> logging:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


score_log = {'init': [], 'n_iter': [], 'n_components': [], 'user_factors': [], 'item_factors': [], 'time_factors': [], 'map_score': [], 'recall_score': [], "f1_score": [],
              'test_data_map_score': [], "test_data_recall_score": [], "test_data_f1_score": [], "time": []}
output_recs = {'user_id': [], 'init': [], 'n_iter': [], 'n_components': []}
for i in range(K):
    output_recs[f"item_{i+1}"] = []

class TrainTestLog():
    def __init__(self):
        self.user_factors = None
        self.item_factors = None
        self.time_factors = None
        self.map_score = None
        self.recall_score = None
        self.f1_score = None
        self.test_data_map_score = None
        self.test_data_recall_score = None
        self.test_data_f1_score = None
        self.time = None

        # For output_recs
        self.user_id = None
        self.init = None
        self.n_iter = None
        self.n_components = None
    
    def update_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

    def update_output_recs(self, params: dict):
        for key, value in params.items():
            if key not in output_recs:
                output_recs[key] = []
            output_recs[key].append(value)

    def update_score_log(self, params: dict):
        for key, value in params.items():
            if key not in score_log:
                score_log[key] = []
            score_log[key].append(value)

    # def update_train_user_recs(self, **params):
    #     for key, value in params.items():
    #         if key not in train_user_recs:
    #             train_user_recs[key] = []
    #         train_user_recs[key].append(value)

    # def update_test_user_recs(self, **params):
    #     for key, value in params.items():
    #         if key not in test_user_recs:
    #             test_user_recs[key] = []
    #         test_user_recs[key].append(value)

    # def get_train_user_recs(self):
    #     return train_user_recs

    # def get_test_user_recs(self):
    #     return test_user_recs

    def get_score_log(self):
        return score_log

    def get_output_recs(self):
        return output_recs

    def get_params(self):
        return {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'time_factors': self.time_factors,
            'map_score': self.map_score,
            'recall_score': self.recall_score,
            'f1_score': self.f1_score,
            'test_data_map_score': self.test_data_map_score,
            'test_data_recall_score': self.test_data_recall_score,
            'test_data_f1_score': self.test_data_f1_score,
            'time': self.time
        }

    def get_output_recs_params(self):
        return {
            'user_id': self.user_id,
            'init': self.init,
            'n_iter': self.n_iter,
            'n_components': self.n_components
        }