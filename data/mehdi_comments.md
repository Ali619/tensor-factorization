**General Comments:**

- Please add a brief description at the beginning of each method, including a link to relevant articles and specifying the method’s input and output types.
- To ensure valid evaluation, we need to separate training and test data, making sure there is no visibility of test data during training. However, in this case, since the prediction range is known and we need to build the tensor accordingly, we can have only the tensor size visible.

**File: `parafac`**

- In `preprocess_data`, avoid taking the absolute value during normalization.
- Apply the training-test separation principle mentioned above in `General Comments` as well. (All data should be normalized using the max value from training only.)
- `get_top_k_recommendations` is not operating based on time; it’s considering the average time instead. Our objective is to recommend an item based on the times it’s recommended, but the code currently considers the average time and recommends based on that.
- Why does `get_top_k_recommendations` use the average of `user_factors` if it fails to obtain `user_vector`?

In general, we should review our approach together to clarify the exact goal of our implementation.