# Tensor Factorization

## Why and How

As we want to predict the item that a user will buy, we can consider our task as a recommendation system, which can be addressed by matrix factorization.

The score we will predict in this code is the amount of an item that a user will buy, which we call it **rate**. So, the top **K** items that a user will buy is the top **K** items that have the highest rate.

If we factorize the matrix to 3 lower rank matrices, then the dot product of the 3 matrices will be the matrix that we want. 

```pesudocode
user_factors, item_factors, time_factors = factors
scores = np.dot(item_factors, weights * user_vector * time_vector)
```

But also, we need a **weight** matrix, so we can **tune** it and use it in the dot product to get the matrix we want. This way we can fill the *Nan* parameters in the original matrix.

Therefore, we used two tensor factorization method named **PARAFAC (Parallel Factor Analysis) Decomposition** and **Tucker Decomposition**.

```pesudocode
# For Tucker
from tensorly.decomposition import tucker
function tucker(tensor: Any, rank: List[Any], n_iter_max: int, init: str,) -> (TuckerTensor | tuple[TuckerTensor, list])

str init_kernel,
int n_components,
list rank,
int n_iter,

tucker_tensor = tucker(tensor, rank=rank, n_iter_max=n_iter, init=init_kernel, random_state=RANDOM_STATE)
    
core, factors = tucker_tensor
user_factors, item_factors, time_factors = factors
```

So we can have the factor matrices we want and assuming `n_components = 10` and `rank = [33, n_components, 408]`, their shape would be like this:

`user_factors: (33, 33) | item_factors: (2227, 10) | time_factors: (408, 408) | core: (33, 10, 408), tucker_tensor: (33, 2227, 408)`

For PARAFAC method, it would be like this:
```pesudocode
from tensorly.decomposition import parafac
function tucker(tensor: Any, rank: List[Any], n_iter_max: int, init: str,) -> (weight, List[factors])

str init_kernel,
int n_components,
int n_iter,

cp_tensor = parafac(tensor, rank=n_components, n_iter_max=n_iter, init=init_kernel, random_state=RANDOM_STATE)

weights, factors = cp_tensor
user_factors, item_factors, time_factors = factors
```

Assuming `n_components = 10`, their shape would be like this:

`user_factors: (33, 10) | item_factors: (2227, 10) | time_factors: (408, 10) | weights: (10,), cp_tensor: (33, 2227, 408)`

## What is our data and data prepocess?

Our data has 4 columns:
- **user**: the id of the user.
- **item**: the id of the item.
- **time**: the id of the time.
- **rate**: the amount of the item that the user bought.

Here is a sample of the data:
user | item | time | rate
--- | --- | --- | ---
697466FA | 411603 | 2023-01-04 | 3.57
697466FA | 238238 | 2023-01-04 | 2.0
697466FA | 142943 | 2023-01-04 | 6.0

After reading the data, we need to do some preprocessing to make it ready for the model and split it into *train* and *test* data. Here is the preprocessing steps:

```pesudocode
df = read_csv('./data/tensor.csv')

function preprocess_data(df, test_size) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["time"] = convert to datetime(df["time"])
    df['timestamp'] = convert time to integer in seconds
    df['rate'] = normalize rates for each item by dividing by the maximum absolute value
    df = sort df by "time"
    
    split = calculate split index based on test_size
    train_df, test_df = split df into train and test portions
    return train_df, test_df

train_df, test_df = preprocess_data(df)
```

After that we need to encode the data to numerical values:

```pesudocode
# Encode categorical variables
le_item = instantiate sklearn.preprocessing.LabelEncoder
le_user = instantiate sklearn.preprocessing.LabelEncoder
le_time = instantiate sklearn.preprocessing.LabelEncoder

train_df['item_encoded'] = encode items using le_item
train_df['user_encoded'] = encode users using le_user
train_df['time_encoded'] = encode timestamps using le_time
```

## Create the tensor

Now that we have the data ready to be used in the model, we have to create the tensor that we will use to train the model. First we will create a tensor with the shape of our encoded-data and fill it with zeros. The shape of this tensor is based on the length of the longest encoded data for each column. Here is the code:

```python
# Create the tensor
tensor_shape = (
    train_df['user_encoded'].max() + 1,
    train_df['item_encoded'].max() + 1,
    train_df['time_encoded'].max() + 1
)

tensor = np.zeros(tensor_shape)
```

Now we can fill the tensor with the data from the encoded data:
```python
for _, row in train_df.iterrows():
    tensor[row['user_encoded'], row['item_encoded'], row['time_encoded']] = row['rate']
```

Here is an example of how does the tensor looks like:
```basic
Original DataFrame:
      user       movie        time  rating
0    Alice   Inception  2023-01-01     4.5
1      Bob  The Matrix  2023-01-02     5.0
2    Alice   Inception  2023-01-03     4.0
3  Charlie  The Matrix  2023-01-02     3.5
4      Bob Interstellar  2023-01-03     4.5

DataFrame with encoded values:
      user       movie        time  rating  user_encoded  movie_encoded  time_encoded
0    Alice   Inception  2023-01-01     4.5             0              1             0
1      Bob  The Matrix  2023-01-02     5.0             1              2             1
2    Alice   Inception  2023-01-03     4.0             0              1             2
3  Charlie  The Matrix  2023-01-02     3.5             2              2             1
4      Bob Interstellar  2023-01-03     4.5             1              0             2

Resulting Tensor:
[[[0.  0.  0. ]
  [4.5 0.  4. ]
  [0.  0.  0. ]]

 [[0.  0.  4.5]
  [0.  0.  0. ]
  [0.  5.  0. ]]

 [[0.  0.  0. ]
  [0.  0.  0. ]
  [0.  3.5 0. ]]]
```
So in this tensor, each matrix will be a **user** and then, in each matrix we have the **rows for movies and the columns for times**. Then **each indices could represent the rate** for each specific combination of **user-item-time**.

## Getting recommendations

The funciton we use for this porpuse is `get_top_k_recommendations()`. This function will return the top k recommendations for a given user. Here is how it works:

```pseudocode
function get_top_k_recommendations(user_id, k) -> list:
    user_vector = get user vector for user_id
    time_vector = average time vector
    
    scores = calculate scores for items based on user_vector, time_vector, and weights
    top_k_items = get top k items based on scores
    
    return top_k_items
```

> **Note:** This aproach is using the mean of the time vector as a way to get the average *time* for each *item*. which is wrong and not consistent with the projects goals and needs. One better approach is to calculate scores for *each item and time combination*.

> **Note:** A brief comparison between PARAFAC and Tucker decomposition:
>
> *CP/PARAFAC*: Simplifies into a sum of rank-one tensors.
>
> *Tucker*: Generalizes CP by allowing a core tensor connected to factor matrices.

## Evaluation Metrics

### mAP (Mean Average Precision)

Here is the code:
```pseudocode
function calculate_map(test_df: pd.Dataframe, k: int) -> float:
    ap_sum = 0
    num_users = 0
    
    for each user in unique users in test_df:
        actual_items = get items for user from test_df
        recommended_items = get_top_k_recommendations(user, k)
        
        if actual_items is empty:
            continue
        
        ap = 0
        hit_count = 0
        
        for each item in recommended_items with index i:
            if item is in actual_items:
                hit_count += 1
                ap += hit_count / i
        
        ap /= min(length of actual_items, k)
        ap_sum += ap
        num_users += 1
    
    return ap_sum / num_users if num_users > 0 else 0
```
**Explanation:**

Actual Items: `actual_items` are collected as a list to preserve order.
- Recommended Items: `recommended_items` should be fetched from your recommendation system.
- Average Precision Calculation: We calculate the precision at each position where a relevant item appears and then average these precisions.
- Mean Average Precision: We sum up the average precision (AP) for each user and divide by the number of users to get the mean average precision (mAP).

### Recall
```pseudocode
function calculate_recall(test_df, k) -> float:
    recall_sum = 0
    num_users = 0
    
    for each user in unique users in test_df:
        actual_items = get unique actual items for user from test_df
        recommended_items = get unique recommended items for user from get_top_k_recommendations(user, k)
        
        if actual_items is empty:
            continue
        
        recall = length of intersection between actual_items and recommended_items / length of actual_items
        recall_sum += recall
        num_users += 1
    
    return recall_sum / num_users if num_users > 0 else 0
```
**Explanation:**

- Recall Calculation: Calculate the recall as the ratio of relevant items in the recommendations to the total number of relevant items.
- Averaging Recall: Sum the recall values for all users and then average them.

### F1 Score
```pseudocode
function calculate_f1_score(test_df, k) -> float:
    f1_sum = 0
    num_users = 0
    
    for each user in unique users in test_df:
        actual_items = get unique actual items for user from test_df
        recommended_items = get unique recommended items for user from get_top_k_recommendations(user, k)
        
        if actual_items is empty or recommended_items is empty:
            continue
        
        precision = length of intersection between actual_items and recommended_items / length of recommended_items
        recall = length of intersection between actual_items and recommended_items / length of actual_items
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        f1_sum += f1
        num_users += 1
    
    return f1_sum / num_users if num_users > 0 else 0
```
**Explanation:**

- Precision Calculation: Calculates the precision for each user.
- Recall Calculation: Calculates the recall for each user.
- F1 Score Calculation: Computes the F1 score for each user using the harmonic mean formula

```math
    \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```
- Average F1 Score: Sums the F1 scores for all users and computes the average.
