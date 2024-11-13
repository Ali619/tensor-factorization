import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
import datetime
import sys

time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

def logger(log_file:str=f'./parafac-log/training-{time}.log', console_level=logging.INFO, file_level=logging.DEBUG) -> logging:
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

class TrainTestLog():
    def __init__(self, k):
        self.k = k
        self.score_log = {'init': [], 'n_iter': [], 'n_components': [], 'map_score': [], 'recall_score': [], "f1_score": [],
                            'test_data_map_score': [], "test_data_recall_score": [], "test_data_f1_score": [], "time": []}
        self.output_recs = {'user_id': [], 'init': [], 'n_iter': [], 'n_components': []}
        for i in range(self.k):
            self.output_recs[f"item_{i+1}"] = []

        # For score log
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
    
    def update_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def update_output_recs(self, params: dict):
        for key, value in params.items():
            if key not in self.output_recs:
                self.output_recs[key] = []
            self.output_recs[key].append(value)

    def update_score_log(self, params: dict):
        self.update_params(params=params)
        for key, value in params.items():
            if key not in self.score_log:
                self.score_log[key] = []
            self.score_log[key].append(value)

    def __get_score_log_train(self):
        return f"""
            'mAP': {self.map_score:.5f} \
            'recall': {self.recall_score:.5f} \
            'f1_score': {self.f1_score:.5f} \
            'time': {self.time:.2f}
        """
    
    def __get_score_log_test(self):
        return f"""
            'mAP': {self.test_data_map_score:.5f} \
            'recall': {self.test_data_recall_score:.5f} \
            'f1_score': {self.test_data_f1_score:.5f} \
            'time': {self.time:.2f}
        """

    def get_score_log(self):
        return f"""
            Train data Evaluation: {self.__get_score_log_train()}
            Test data Evaluation: {self.__get_score_log_test()}
        """
    
    def get_output_recs(self):
        return self.output_recs

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

    def create_csv(self):
        self.score_log_df = pd.DataFrame(self.score_log)
        self.output_recs_df = pd.DataFrame(self.output_recs)
        self.score_log_df.to_csv(f'./parafac-log/training-log-top{self.k}-{time}.csv', index=False)
        self.output_recs_df.to_csv(f'./parafac-log/Top-{self.k} Recommendation on testData - {time}.csv', index=False)
