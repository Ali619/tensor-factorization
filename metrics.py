import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Function to get recommendations for a specific user and time
def get_recommendations(user_id, time_id, le_user: LabelEncoder, le_time: LabelEncoder, factorized_tensor, k) -> np.ndarray:
    """Get top-k recommendations with scores"""
    user_encoded = le_user.transform([user_id])[0]
    time_encoded = le_time.transform([time_id])[0]

    user_predictions = factorized_tensor[user_encoded, :, time_encoded]
    top_items = np.argsort(user_predictions)[-k:][::-1] # Get K-latest values and reverse output from highest to lowest
   
    return top_items

def eval_flatten_calc(y_true: np.array, y_pred: np.array) -> dict:
    """This function will get Original test data tensor and Refactore test data tensor, convert them to 1-d array (`flatten`)
    and return evaluation metrics values.    
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same."
    assert np.all(y_true >= 0) and np.all(y_pred >= 0), "There is negative values in y_true or y_pred"
    
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    true_positives = np.sum(y_true_bool & y_pred_bool)

    # Calculate precision
    predicted_positives = np.sum(y_pred_bool)
    if predicted_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / predicted_positives

    # Calculate recall
    actual_positives = np.sum(y_true_bool)
    if actual_positives == 0:
        recall = 0.0
    else:
        recall = true_positives / actual_positives

    # Calculate F1 score
    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {'precision': float(precision), 'recall': float(recall), 'f1_score': float(f1_score)}

def calculate_map(test_df: pd.DataFrame, user_recs: dict, le_item: LabelEncoder, k: int=5) -> float:
    ap_sum = 0
    num_users = 0
    for user in test_df['user'].unique():
        actual_items = (test_df[test_df['user'] == user]['item']).to_list()
        recommended_items = [le_item.inverse_transform(np.array([i])).item() for i in user_recs[user]]
        if not actual_items:
            continue
        ap = 0
        hit_count = 0
        for i, item in enumerate(recommended_items, start=1):
            if item in actual_items:
                hit_count += 1
                ap += hit_count / i

        ap /= min(len(actual_items), k)
        ap_sum += ap
        num_users += 1
    return ap_sum / num_users if num_users > 0 else 0

def calculate_recall(test_df: pd.DataFrame, user_recs: dict, le_item: LabelEncoder, k:int=5) -> float:
    recall_sum = 0
    num_users = 0

    for user in test_df['user'].unique():
        actual_items = set(test_df[test_df['user'] == user]['item'])
        recommended_items = set([le_item.inverse_transform(np.array([i])).item() for i in user_recs[user]])
        if not actual_items:
            continue
        recall = len(actual_items.intersection(recommended_items)) / len(actual_items)
        recall_sum += recall
        num_users += 1

    return recall_sum / num_users if num_users > 0 else 0

def calculate_f1_score(test_df: pd.DataFrame, user_recs: dict, le_item: LabelEncoder, k: int = 5) -> float:
    f1_sum = 0
    num_users = 0

    for user in test_df['user'].unique():
        actual_items = set(test_df[test_df['user'] == user]['item'])
        # recommended_items = set(get_top_k_recommendations(user, k))
        recommended_items = set([le_item.inverse_transform(np.array([i])).item() for i in user_recs[user]])
        if not actual_items or not recommended_items:
            continue
        
        precision = len(actual_items.intersection(recommended_items)) / len(recommended_items)
        recall = len(actual_items.intersection(recommended_items)) / len(actual_items)
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        f1_sum += f1
        num_users += 1
    return f1_sum / num_users if num_users > 0 else 0
