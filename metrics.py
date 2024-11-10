import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from preprocess import preprocess_for_classification_report

def get_top_k_recommendations(user_id, time_id, le_user, user_factors,
                                le_time, time_factors, 
                                le_item, item_factors, weights, logger, k:int=5):
    try:
        user_encoded = le_user.transform([user_id])[0]
    except ValueError:
        logger.warning(f"User {user_id} or time {time_id} not found in the training data.")
        user_vector = np.mean(user_factors, axis=0) # Average over users
    
    try:
        time_encoded = le_time.transform([time_id])[0]
    except ValueError:
        logger.warning(f"Time {time_id} not found in the training data.")
        time_vector = np.mean(time_factors, axis=0)  # Average over time
    
    user_vector = user_factors[user_encoded]
    time_vector = time_factors[time_encoded]
    
    # Calculate predicted rate for all items at the specific time
    scores = []
    for item in range(item_factors.shape[0]):
        item_vector = item_factors[item]
        prediction = sum(weights[r] * user_vector[r] * item_vector[r] * time_vector[r] 
                        for r in range(len(weights)))
        scores.append(prediction)
    
    scores = np.array(scores)
    top_k_items = np.argsort(scores)[::-1][:k]
    
    return [le_item.inverse_transform([item])[0] for item in top_k_items]

def calculate_map(test_df: pd.DataFrame, user_recs: dict, k: int = 5) -> float:
    ap_sum = 0
    num_users = 0

    for user in test_df['user'].unique():
        actual_items = test_df[test_df['user'] == user]['item'].tolist()
        # recommended_items = get_top_k_recommendations(user, k)
        recommended_items = user_recs[user]
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

def calculate_recall(test_df: pd.DataFrame, user_recs: dict, k:int=5) -> float:
    recall_sum = 0
    num_users = 0

    for user in test_df['user'].unique():
        actual_items = set(test_df[test_df['user'] == user]['item'])
        # recommended_items = set(get_top_k_recommendations(user, k))
        recommended_items = set(user_recs[user])
        if not actual_items:
            continue
        recall = len(actual_items.intersection(recommended_items)) / len(actual_items)
        recall_sum += recall
        num_users += 1

    return recall_sum / num_users if num_users > 0 else 0

def calculate_f1_score(test_df: pd.DataFrame, user_recs: dict, k: int = 5) -> float:
    f1_sum = 0
    num_users = 0

    for user in test_df['user'].unique():
        actual_items = set(test_df[test_df['user'] == user]['item'])
        # recommended_items = set(get_top_k_recommendations(user, k))
        recommended_items = set(user_recs[user])
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

def get_classification_report(test_df:pd.DataFrame, user_recs: dict, k:int=5):
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