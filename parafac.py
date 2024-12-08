from timeit import default_timer as timer

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

import tensorly as tl
from logger import TrainTestLog, logger
from metrics import (
    calculate_f1_score,
    calculate_map,
    calculate_recall,
    convert_result_to_org_format,
    eval_flatten_calc,
    get_recommendations,
    user_item_history,
)
from preprocess import preprocess_data
from tensorly.decomposition import non_negative_parafac

RANDOM_STATE = 42
DATA_PATH = "./data/tensor.csv"
INIT_KERNEL = ["random"]
N_ITER = [1000]
N_COMPONENTS = [300]
K = 1
TOLERANCE = 1e-10  # Default in tensorly: 1e-6

train_test_log = TrainTestLog(k=K)
logger = logger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Device is: {DEVICE}")
logger.info(f"K is: {K}")

# Set the TensorLy backend to NumPy for better performance
tl.set_backend("numpy" if DEVICE == "cpu" else "pytorch")
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

train_df, test_df = preprocess_data(path=DATA_PATH)
full_df = preprocess_data(path=DATA_PATH, split=False)

print(f"len train: {len(train_df)} & len test: {len(test_df)}")
print(f"len data (full): {len(full_df)}")

# Encode categorical variables
le_item = LabelEncoder()
le_user = LabelEncoder()
le_time = LabelEncoder()

full_df["item_encoded"] = le_item.fit_transform(full_df["item"])
full_df["user_encoded"] = le_user.fit_transform(full_df["user"])
full_df["time_encoded"] = le_time.fit_transform(full_df["timestamp"])

# Create the tensor
tensor_shape = (
    full_df["user_encoded"].max() + 1,
    full_df["item_encoded"].max() + 1,
    full_df["time_encoded"].max() + 1,
)

tensor = np.zeros(tensor_shape)
tensor[full_df["user_encoded"], full_df["item_encoded"], full_df["time_encoded"]] = (
    full_df["rate"].values
)

org_tensor = tensor.copy()
if DEVICE == "cuda":
    tensor = torch.from_numpy(tensor).type(dtype=torch.float32).to(DEVICE)
split = len(train_df["timestamp"].unique())
tensor[:, :, split:] = 0

# Store user recommendations for train and test users to reduce processing time in evaluations
train_user_recs = {}
test_user_recs = {}

for init_kernel in INIT_KERNEL:
    for n_components in N_COMPONENTS:
        for n_iter in N_ITER:
            logger.info(
                f"Training for init: {init_kernel} | n_components: {n_components} | n_iter: {n_iter} | tolerance: {TOLERANCE}"
            )
            start = timer()
            try:
                cp_tensor = non_negative_parafac(
                    tensor,
                    rank=n_components,
                    n_iter_max=n_iter,
                    init=init_kernel,
                    tol=TOLERANCE,
                    random_state=RANDOM_STATE,
                    verbose=1,
                )
            except Exception as e:
                logger.error(f"ERROR: {e}")
                logger.error(
                    f"Failed for init: {init_kernel} | n_components: {n_components} | n_iter: {n_iter}"
                )
                continue
            stop = timer()

            factorized_tensor = tl.cp_to_tensor(cp_tensor)
            if torch.is_tensor(factorized_tensor):
                factorized_tensor = factorized_tensor.cpu().numpy()

            sparsity = 1.0 - (
                np.count_nonzero(factorized_tensor) / float(factorized_tensor.size)
            )
            logger.info(f"sparsity % after factorization: {sparsity}")

            results_in_org_format = convert_result_to_org_format(
                test_df=test_df,
                le_user=le_user,
                le_item=le_item,
                le_time=le_time,
                k=K,
                factorized_tensor=factorized_tensor,
            )
            results_in_org_format.to_csv(
                f"./parafac-log/org_format_result-kenel-{init_kernel}-n_components-{n_components}-n_iter-{n_iter}.csv",
                index=False,
            )

            metrics = eval_flatten_calc(
                y_true=org_tensor[:, :, split:], y_pred=factorized_tensor[:, :, split:]
            )
            logger.info(
                f"eval metrics based on flatten tensors: {[(key, value) for key, value in metrics.items()]}"
            )

            logger.info(
                f"Getting top-k recommendations for train and test data to start evaluation"
            )
            for user in train_df["user"].unique():
                users = train_df[train_df["user"] == user]
                for time_id in users["timestamp"].unique():
                    # print(f"user: {user}, time: {time_id}")
                    train_user_recs[user] = get_recommendations(
                        user_id=user,
                        time_id=time_id,
                        k=K,
                        le_user=le_user,
                        le_time=le_time,
                        factorized_tensor=factorized_tensor,
                    )

            # print("getting eval for mAP and others for train")
            map_score = calculate_map(train_df, train_user_recs, le_item, k=K)
            recall_score = calculate_recall(train_df, train_user_recs, le_item, k=K)
            f1_score = calculate_f1_score(train_df, train_user_recs, le_item, k=K)

            for user in test_df["user"].unique():
                users = test_df[test_df["user"] == user]
                for time_id in users["timestamp"].unique():
                    test_user_recs[user] = get_recommendations(
                        user_id=user,
                        time_id=time_id,
                        k=K,
                        le_user=le_user,
                        le_time=le_time,
                        factorized_tensor=factorized_tensor,
                    )

            logger.info(
                f"Item accuracy: %{user_item_history(test_df=test_df, user_recs=test_user_recs)}"
            )

            test_data_map_score = calculate_map(test_df, test_user_recs, le_item, k=K)
            test_data_recall_score = calculate_recall(
                test_df, test_user_recs, le_item, k=K
            )
            test_data_f1_score = calculate_f1_score(
                test_df, test_user_recs, le_item, k=K
            )

            train_test_log.update_score_log(
                {
                    "init": init_kernel,
                    "n_iter": n_iter,
                    "n_components": n_components,
                    "map_score": map_score,
                    "recall_score": recall_score,
                    "f1_score": f1_score,
                    "test_data_map_score": test_data_map_score,
                    "test_data_recall_score": test_data_recall_score,
                    "test_data_f1_score": test_data_f1_score,
                    "time": round(stop - start, 2),
                }
            )
            logger.info(f"{train_test_log.get_score_log()}\n")

            # Get prediction. Will be used to create a csv file after training will be done
            for user in test_df["user"].unique():
                item = []
                train_test_log.update_output_recs(
                    {
                        "init": init_kernel,
                        "n_iter": n_iter,
                        "n_components": n_components,
                        "user_id": user,
                    }
                )
                users = test_df[test_df["user"] == user]
                for time_id in users["timestamp"].unique():
                    result = get_recommendations(
                        user_id=user,
                        time_id=time_id,
                        k=K,
                        le_user=le_user,
                        le_time=le_time,
                        factorized_tensor=factorized_tensor,
                    )
                for i, item in enumerate(result):
                    train_test_log.update_output_recs({f"item_{i+1}": item})

train_test_log.create_csv()
