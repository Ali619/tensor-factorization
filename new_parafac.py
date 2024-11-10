import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.preprocessing import LabelEncoder
from timeit import default_timer as timer
from metrics import calculate_map, calculate_recall, calculate_f1_score, get_top_k_recommendations, get_classification_report
from logger import TrainTestLog, setup_logger
from preprocess import preprocess_data
from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = os.getenv("data_path")

RANDOM_STATE = 42
INIT_KERNEL = ["random", "svd"]
N_ITER = [1]
N_COMPONENTS = [10]
K = 1

logger = setup_logger()
train_test_log = TrainTestLog()

# Set the TensorLy backend to NumPy for better performance
tl.set_backend('numpy')
np.random.seed(RANDOM_STATE)

train_df, test_df = preprocess_data(path=DATA_PATH)
full_df =  preprocess_data(path=DATA_PATH, split=False)

print(f'len train: {len(train_df)} & len test: {len(test_df)}')
print(f'len data (full): {len(full_df)}')

# Encode categorical variables
le_item = LabelEncoder()
le_user = LabelEncoder()
le_time = LabelEncoder()

full_df['item_encoded'] = le_item.fit_transform(full_df['item'])
full_df['user_encoded'] = le_user.fit_transform(full_df['user'])
full_df['time_encoded'] = le_time.fit_transform(full_df['timestamp'])

# Create the tensor
tensor_shape = (
    full_df['user_encoded'].max() + 1,
    full_df['item_encoded'].max() + 1,
    full_df['time_encoded'].max() + 1
)

tensor = np.zeros(tensor_shape)
for _, row in full_df.iterrows():
    tensor[row['user_encoded'], row['item_encoded'], row['time_encoded']] = row['rate']

test_tensor = tensor.copy()
split = len(train_df['timestamp'].unique())
tensor[:, :, split:] = 0

train_user_recs = {}
test_user_recs = {}

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
            
            factorized_tensor = tl.cp_to_tensor(cp_tensor)
            sparsity = 1.0 - (np.count_nonzero(factorized_tensor) / float(factorized_tensor.size))
            print(f'sparsity % after factorization: {sparsity}')

            def get_recommendations(user_id, time_id, k=5):
                """Get top-k recommendations with scores"""
                user_predictions = factorized_tensor[user_id, :, time_id]
                top_items = np.argsort(user_predictions)[-k:][::-1]
                top_scores = user_predictions[top_items]
                
                return list(zip(top_items, top_scores))

            def get_actual_items(user_id, time_id, test_tensor, threshold=0):
                """Get actual items that user interacted with"""
                return set(np.where(test_tensor[user_id, :, time_id] > threshold)[0])

            def get_recommended_items(user_id, time_id, factorized_tensor, k):
                """Get top-k recommended items"""
                user_predictions = factorized_tensor[user_id, :, time_id]
                return set(np.argsort(user_predictions)[-k:])

            def calculate_metrics(actual_items, recommended_items, k: int=5):
                """Calculate Recall, Precision, and F1 score"""
                if not actual_items:
                    return 0.0, 0.0, 0.0
                
                # Number of relevant items in top-k recommendations
                relevant_and_recommended = len(actual_items.intersection(recommended_items))
                
                recall = relevant_and_recommended / len(actual_items)
                precision = relevant_and_recommended / k
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                
                return recall, precision, f1

            def calculate_ap(actual_items, recommended_items_list):
                """Calculate Average Precision"""
                if not actual_items:
                    return 0.0
                
                precisions = []
                relevant_count = 0
                
                for i, item in enumerate(recommended_items_list, 1):
                    if item in actual_items:
                        relevant_count += 1
                        precisions.append(relevant_count / i)
                
                if not precisions:
                    return 0.0
                
                return sum(precisions) / len(actual_items)
            
            from collections import defaultdict
            def evaluate_recommendations(test_tensor, factorized_tensor, split, k: int=5):
                """Evaluate recommendations using multiple metrics at different K values"""
                metrics = defaultdict(list)
                
                # For each user and time in test period
                for user_id in range(test_tensor.shape[0]):
                    for time_id in range(split, test_tensor.shape[2]):
                        actual_items = get_actual_items(user_id, time_id, test_tensor)
                        
                        if not actual_items:
                            continue
                            
                        # Get predictions for this user and time
                        predictions = factorized_tensor[user_id, :, time_id]
                        recommended_items_list = np.argsort(predictions)[::-1]
                        
                        recommended_items = set(recommended_items_list[:k])
                        
                        recall, precision, f1 = calculate_metrics(actual_items, recommended_items, k)
                        ap = calculate_ap(actual_items, recommended_items_list[:k])
                        
                        metrics[f'recall@{k}'].append(recall)
                        metrics[f'map@{k}'].append(ap)
                        metrics[f'f1@{k}'].append(f1)
                
                # Calculate average metrics
                final_metrics = {}
                for metric, values in metrics.items():
                    final_metrics[metric] = np.mean(values)
                
                return final_metrics

            metrics = evaluate_recommendations(test_tensor, factorized_tensor, split, k=K)

            # Print results
            print("\nEvaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            # Function to get recommendations for a specific user and time
            def get_recommendations(user_id, time_id, k: int=5):
                """Get top-k recommendations with scores"""
                user_predictions = factorized_tensor[user_id, :, time_id]
                top_items = np.argsort(user_predictions)[-k:][::-1]
                # top_scores = user_predictions[top_items]
                
                # return list(zip(top_items, top_scores))
                return top_items
            
            logger.info(f"Getting top-k recommendations from test data to start evaluation")
            print("get user_recs for train")
            for user in train_df['user'].unique():
                users = train_df[train_df["user"] == user]
                for time_id in users["timestamp"].unique():
                    # print(f"user: {user}, time: {time_id}")
                    train_user_recs[user] = get_recommendations(user_id=user, time_id=time_id, k=K)
            
            print("getting eval for mAP and others for train")
            map_score = calculate_map(train_df, train_user_recs, k=K)
            recall_score = calculate_recall(train_df, train_user_recs, k=K)
            f1_score = calculate_f1_score(train_df, train_user_recs, k=K)

            print("get user_recs for test")
            for user in test_df['user'].unique():
                users = test_df[test_df["user"] == user]
                for time_id in users["timestamp"].unique():
                    test_user_recs[user] = get_recommendations(user_id=user, time_id=time_id, k=K)

            print("getting eval for mAP and others for test")
            test_data_map_score = calculate_map(test_df, test_user_recs, k=K)
            test_data_recall_score = calculate_recall(test_df, test_user_recs, k=K)
            test_data_f1_score = calculate_f1_score(test_df, test_user_recs, k=K)

            logger.info(f"Mean Average Precision (mAP@{K}) | train data: {map_score} | test data: {test_data_map_score} ")
            logger.info(f"Recall@{K} | train data: {recall_score} | test data: {test_data_recall_score}")
            logger.info(f"F1@{K} | train data: {f1_score} | test data: {test_data_f1_score}")
            logger.info(f"Taken Time: {stop - start:.2f} seconds\n")

            train_test_log.update_score_log({'init': init_kernel, 'n_iter': n_iter, 'n_components': n_components, 
                                            'map_score': map_score, 'recall_score': recall_score, 'f1_score': f1_score, 
                                            'test_data_map_score': test_data_map_score, 'test_data_recall_score': test_data_recall_score, 'test_data_f1_score': test_data_f1_score, 
                                            'time': round(stop-start, 2)})
            # score_log['init'].append(init_kernel)
            # score_log['n_iter'].append(n_iter)
            # score_log['n_components'].append(n_components)
            # score_log["map_score"].append(map_score)
            # score_log['recall_score'].append(recall_score)
            # score_log["f1_score"].append(f1_score)
            # score_log['test_data_map_score'].append(test_data_map_score)
            # score_log['test_data_recall_score'].append(test_data_recall_score)
            # score_log["test_data_f1_score"].append(test_data_f1_score)
            # score_log['time'].append(round(stop-start, 2))

            print("getting eval for classification report")
            report = get_classification_report(test_df, test_user_recs, k=K)
            logger.info(f"Classification Report is done: \n{report}")
            
            for user in test_df['user'].unique():
                # print(f"Getting top-k recommendations for user {user}")
                output_recs['init'].append(init_kernel)
                output_recs['n_iter'].append(n_iter)
                output_recs['n_components'].append(n_components)
                output_recs['user_id'].append(user)
                result = get_top_k_recommendations(user_id=user, k=K)
                for i, item in enumerate(result):
                    output_recs[f"item_{i+1}"].append(item)

score_log_df = pd.DataFrame(score_log)
score_log_df.to_csv(f'./parafac-log/train-log-top{K}.csv', index=False)
output_recs = pd.DataFrame(output_recs)
output_recs.to_csv(f'./parafac-log/top{K}-recommendation-test.csv', index=False)
