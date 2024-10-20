import pandas as pd
from pandas import to_datetime
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from timeit import default_timer as timer
import logging
from logging.handlers import RotatingFileHandler
import sys

# Set up logging
def setup_logger(log_file:str='./parafac-log/training.log', console_level=logging.INFO, file_level=logging.DEBUG) -> logging:
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


RANDOM_STATE = 42
INIT_KERNEL = ["random", "svd"]
N_ITER = [100]
N_COMPONENTS = [10, 20, 30, 50, 100, 200, 300]
K = 1

# RANDOM_STATE = 42
# INIT_KERNEL = ["random"]
# N_ITER = [5]
# N_COMPONENTS = [5]
# K = 1

logger = setup_logger(f'./parafac-log/top{K}-training.log')

# log classification_report manually
open(f"./parafac-log/classification_report-top{K}.txt", "w")

# Set the TensorLy backend to NumPy for better performance
tl.set_backend('numpy')
np.random.seed(RANDOM_STATE)

def preprocess_data(df:pd.DataFrame, test_size:float=0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["time"] = to_datetime(df["time"])
    df['timestamp'] = df['time'].astype(int) // 10**9
    df = df.sort_values(by="time")
    
    split = int(len(df) * test_size)
    train_df, test_df = df[:split], df[split:]
    return train_df, test_df

# Read the CSV file
df = pd.read_csv('./data/tensor.csv')

train_df, test_df = preprocess_data(df)
print(f'len train: {len(train_df)} & len test: {len(test_df)}')

# Encode categorical variables
le_item = LabelEncoder()
le_user = LabelEncoder()
le_time = LabelEncoder()

train_df['item_encoded'] = le_item.fit_transform(train_df['item'])
train_df['user_encoded'] = le_user.fit_transform(train_df['user'])
train_df['time_encoded'] = le_time.fit_transform(train_df['timestamp'])

# Create the tensor
tensor_shape = (
    train_df['user_encoded'].max() + 1,
    train_df['item_encoded'].max() + 1,
    train_df['time_encoded'].max() + 1
)

tensor = np.zeros(tensor_shape)

for _, row in train_df.iterrows():
    tensor[row['user_encoded'], row['item_encoded'], row['time_encoded']] = row['rate']

# Evaluate recommendations using Mean Average Precision (MAP)
def calculate_map(test_df:pd.DataFrame, k:int=5) -> float:
    ap_sum = 0
    num_users = 0

    for user in test_df['user'].unique():
        actual_items = set(test_df[test_df['user'] == user]['item'])
        recommended_items = set(get_top_k_recommendations(user, k))
        
        precision_sum = sum([1 if item in actual_items else 0 for item in recommended_items])
        ap = precision_sum / min(k, len(actual_items))
        
        ap_sum += ap
        num_users += 1

    return ap_sum / num_users

def calculate_recall(test_df:pd.DataFrame, k:int=5) -> float:
    recall_sum = 0
    num_users = 0

    for user in test_df['user'].unique():
        actual_items = set(test_df[test_df['user'] == user]['item'])
        recommended_items = set(get_top_k_recommendations(user, k))

        recall = len(actual_items.intersection(recommended_items)) / len(actual_items)
        recall_sum += recall
        num_users += 1

    return recall_sum / num_users

def get_top_k_recommendations(user_id, k:int=5):
    try:
        user_encoded = le_user.transform([user_id])[0]
        user_vector = user_factors[user_encoded]
    except ValueError:
        logger.warning(f"User {user_id} not found in the training data. Using average user vector.")
        user_vector = np.mean(user_factors, axis=0)
    
    time_vector = np.mean(time_factors, axis=0)  # Average over time
    
    scores = np.dot(item_factors, weights * user_vector * time_vector)
    top_k_items = np.argsort(scores)[::-1][:k]
    
    return [le_item.inverse_transform([item])[0] for item in top_k_items]

def preprocess_for_classification_report(test_df:pd.DataFrame) -> pd.DataFrame:
    if "is_buying" not in test_df.columns:
        test_df["is_buying"] = False
    test_df["is_buying"] = test_df["rate"] > 0
    return test_df

def get_classification_report(test_df:pd.DataFrame, k:int=5):
    recommendations = []
    y_true = []
    users = preprocess_for_classification_report(test_df)
    for _, row in users.iterrows():
        user_recs = get_top_k_recommendations(row["user"], k)
        for item in user_recs:
            recommendations.append(1 if item in row["item"] else 0)
            y_true.append(1 if row["is_buying"] else 0)
    report = classification_report(y_true, recommendations, labels=[0, 1])
    return report

# Perform CP decomposition
score_log = {'init': [], 'n_iter': [], 'n_components': [], 'user_factors': [], 'item_factors': [], 'time_factors': [], 'map_score': [], 'recall_score': [],
              'test_data_map_score': [], "test_data_recall_score": [], "time": []}
test_df_recommendation = {'user_id': [], 'init': [], 'n_iter': [], 'n_components': [], 'item_1': []}

for init_kernel in INIT_KERNEL:
    for n_components in N_COMPONENTS:
        for n_iter in N_ITER:
            logger.info(f"Training for init: {init_kernel} | n_components: {n_components} | n_iter: {n_iter}")
            start = timer()
            try:
                cp_tensor = parafac(tensor, rank=n_components, n_iter_max=n_iter, init=init_kernel, random_state=RANDOM_STATE, verbose=1)
            except Exception as e:
                logger.error(f"ERROR: {e}")
                logger.error(f"Failed for init: {init_kernel} | n_components: {n_components} | n_iter: {n_iter}")
                continue
            stop = timer()
            
            # Extract factors from CP tensor
            weights, factors = cp_tensor
            user_factors, item_factors, time_factors = factors
            score_log['user_factors'].append(str(user_factors.shape))
            score_log['item_factors'].append(str(item_factors.shape))
            score_log['time_factors'].append(str(time_factors.shape))

            logger.info(f"Getting top-k recommendations for test data")

            map_score = calculate_map(train_df, k=K)
            recall_score = calculate_recall(train_df, k=K)

            test_data_map_score = calculate_map(test_df, k=K)
            test_data_recall_score = calculate_recall(test_df, k=K)
            
            logger.info(f"Mean Average Precision (MAP@{K}) | train data: {map_score} | test data: {test_data_map_score}")
            logger.info(f"Recall@{K} | train data: {recall_score} | test data: {test_data_recall_score}")
            logger.info(f"Taken Time: {stop - start:.2f} seconds\n")

            score_log['init'].append(init_kernel)
            score_log['n_iter'].append(n_iter)
            score_log['n_components'].append(n_components)
            score_log["map_score"].append(map_score)
            score_log['recall_score'].append(recall_score)
            score_log['test_data_map_score'].append(test_data_map_score)
            score_log['test_data_recall_score'].append(test_data_recall_score)
            score_log['time'].append(round(stop-start, 2))

            report = get_classification_report(test_df, k=K)
            logger.info(f"Classification Report is done: \n{report}")
            
            with open(f"./parafac-log/classification_report-top{K}.txt", 'a') as f:
                f.write("{:<10} | {:<6} | {:<12} | {:<13} | {:<13} | {:<13} | {:<8} | {:<8} | {:<20} | {:<20} | {:<4}\n".format(
                            "init", "n_iter", "n_components", "user_factors", "item_factors", 
                            "time_factors", "map_score", "recall_score", "test_data_map_score", 
                            "test_data_recall_score", "time"
                        ))
                f.write(f"{init_kernel:<10} | {n_iter:<6} | {n_components:<12} | "
                        f"{str(user_factors.shape):<13} | {str(item_factors.shape):<13} | "
                        f"{str(time_factors.shape):<13} | {map_score:.8f} | "
                        f"{recall_score:.8f} | {test_data_map_score:<20} | "
                        f"{test_data_recall_score:<23} | {round(stop-start, 2):<4}\n")
                f.write(f"{report}\n")
                f.write("-" * 200 + "\n")

score_log_df = pd.DataFrame(score_log)
score_log_df.to_csv(f'./parafac-log/train-log-top{K}.csv', index=False)
test_df_recommendation = pd.DataFrame(test_df_recommendation)
test_df_recommendation.to_csv(f'./parafac-log/top{K}-recommendation-test.csv', index=False)
