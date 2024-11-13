import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.preprocessing import LabelEncoder
from timeit import default_timer as timer
from metrics import calculate_map, calculate_recall, calculate_f1_score, get_recommendations
from logger import TrainTestLog, logger
from preprocess import preprocess_data
from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
RANDOM_STATE = int(os.getenv('RANDOM_STATE'))
INIT_KERNEL = os.getenv('INIT_KERNEL').split(',')
N_ITER = list(map(int, os.getenv('N_ITER').split(',')))
N_COMPONENTS = list(map(int, os.getenv('N_COMPONENTS').split(',')))
K = int(os.getenv('K'))

logger = logger()
train_test_log = TrainTestLog(k=K)

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

org_tensor = tensor.copy()
split = len(train_df['timestamp'].unique())
tensor[:, :, split:] = 0

# Store user recommendations for train and test users to reduce processing time in evaluations
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
            logger.info(f'sparsity % after factorization: {sparsity}')
            
            logger.info(f"Getting top-k recommendations for train and test data to start evaluation")
            for user in train_df['user'].unique():
                users = train_df[train_df["user"] == user]
                for time_id in users["timestamp"].unique():
                    # print(f"user: {user}, time: {time_id}")
                    train_user_recs[user] = get_recommendations(user_id=user, time_id=time_id, k=K,
                                                                le_user=le_user, le_time=le_time, factorized_tensor=factorized_tensor)
            
            # print("getting eval for mAP and others for train")
            map_score = calculate_map(train_df, train_user_recs, le_item, k=K)
            recall_score = calculate_recall(train_df, train_user_recs, le_item, k=K)
            f1_score = calculate_f1_score(train_df, train_user_recs, le_item, k=K)

            for user in test_df['user'].unique():
                users = test_df[test_df["user"] == user]
                for time_id in users["timestamp"].unique():
                    test_user_recs[user] = get_recommendations(user_id=user, time_id=time_id, k=K,
                                                               le_user=le_user, le_time=le_time, factorized_tensor=factorized_tensor)
            
            test_data_map_score = calculate_map(test_df, test_user_recs, le_item, k=K)
            test_data_recall_score = calculate_recall(test_df, test_user_recs, le_item, k=K)
            test_data_f1_score = calculate_f1_score(test_df, test_user_recs, le_item, k=K)
            
            train_test_log.update_score_log({'init': init_kernel, 'n_iter': n_iter, 'n_components': n_components, 
                                            'map_score': map_score, 'recall_score': recall_score, 'f1_score': f1_score, 
                                            'test_data_map_score': test_data_map_score, 'test_data_recall_score': test_data_recall_score, 'test_data_f1_score': test_data_f1_score, 
                                            'time': round(stop-start, 2)})
            logger.info(train_test_log.get_score_log())

            for user in test_df['user'].unique():
                item = []
                train_test_log.update_output_recs({'init': init_kernel, 'n_iter': n_iter, 'n_components': n_components, 'user_id': user})
                users = test_df[test_df['user'] == user]
                for time_id in users['timestamp'].unique():
                    result = get_recommendations(user_id=user, time_id=time_id, k=K, 
                                                 le_user=le_user, le_time=le_time, factorized_tensor=factorized_tensor)
                for i, item in enumerate(result):
                    train_test_log.update_output_recs({f"item_{i+1}": item})

train_test_log.create_csv()

