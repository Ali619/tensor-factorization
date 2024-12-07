import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from sklearn.preprocessing import LabelEncoder
from timeit import default_timer as timer
from metrics import eval_flatten_calc, user_item_history, get_recommendations
from logger import TrainTestLog, logger
from preprocess import preprocess_data

RANDOM_STATE = 42
DATA_PATH = './data/tensor.csv'
INIT_KERNEL = ['random']
N_ITER = [1000]
N_COMPONENTS = [500]
K = 1
TOLERANCE = 1e-6 # Default: 1e-6. The algorithm is considered to have found the global minimum when the reconstruction error is less than tolerance

train_test_log = TrainTestLog(k=K)
logger = logger()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logger.info(f"Device is: {DEVICE}")
logger.info(f"K is: {K}")

# Set the TensorLy backend to NumPy for better performance
tl.set_backend('numpy' if DEVICE == 'cpu' else 'pytorch')
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# train_df, test_df = preprocess_data(path=DATA_PATH)
full_df =  preprocess_data(path=DATA_PATH, split=False)

# print(f'len train: {len(train_df)} & len test: {len(test_df)}')
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
tensor[full_df['user_encoded'], full_df['item_encoded'], full_df['time_encoded']] = full_df['rate'].values

org_tensor = tensor.copy()
if DEVICE == 'cuda':
    tensor = torch.from_numpy(tensor).type(dtype=torch.float32).to(DEVICE)

unique_timestamps = sorted(full_df['time_encoded'].unique())
window_size = [100]

predictions = []
actual_values = []
test_user_recs = {}

# Sliding window approach
for window in window_size:
    for i in range(len(unique_timestamps) - window):

        start_idx = i
        end_idx = i + window
        pred_idx = i + window  # The day to predict

        # train_df = full_df[full_df['time_encoded'] <= unique_timestamps[start_idx]]
        test_df = full_df[full_df['time_encoded'] == unique_timestamps[pred_idx]]  

        train_window = tensor[:, :, start_idx:end_idx]
        test_window = tensor[:, :, pred_idx]
        if torch.is_tensor(test_window):
            test_window = test_window.cpu().numpy()

        for init_kernel in INIT_KERNEL:
            for n_components in N_COMPONENTS:
                for n_iter in N_ITER:
                    logger.info(f"Training for window_size: {window} | range (day): {start_idx} to {end_idx} | init: {init_kernel} | n_components: {n_components} | n_iter: {n_iter} | tolerance: {TOLERANCE}")
                    start = timer()
                    try:
                        cp_tensor = non_negative_parafac(
                            train_window, 
                            rank=n_components, 
                            n_iter_max=n_iter, 
                            init=init_kernel,
                            tol=TOLERANCE,
                            random_state=RANDOM_STATE,
                            verbose=1
                        )
                    except Exception as e:
                        logger.error(f"ERROR: {e}")
                        logger.error(f"Failed for window_size: {window} | range (day): {start_idx} to {end_idx} | init: {init_kernel} | n_components: {n_components} | n_iter: {n_iter}")
                        continue
                    stop = timer()

                    factorized_tensor = tl.cp_to_tensor(cp_tensor)
                    if torch.is_tensor(factorized_tensor):
                        factorized_tensor = factorized_tensor.cpu().numpy()

                    predictions.append(factorized_tensor[:, :, -1])  # Last day predictions
                    actual_values.append(test_window)

                    metrics = eval_flatten_calc(y_true=np.array(actual_values), y_pred=np.array(predictions))
                    logger.info(f"eval metrics based on flatten tensors: {[(key, value) for key, value in metrics.items()]}")
 
