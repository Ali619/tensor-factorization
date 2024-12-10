import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

from preprocess import preprocess_data

DATA_PATH = "./data/tensor.csv"

full_df = preprocess_data(path=DATA_PATH, split=False)

print(f"len data (full): {len(full_df)}")

le_item = LabelEncoder()
le_user = LabelEncoder()
le_time = LabelEncoder()

full_df["item_encoded"] = le_item.fit_transform(full_df["item"])
full_df["user_encoded"] = le_user.fit_transform(full_df["user"])
full_df["time_encoded"] = le_time.fit_transform(full_df["timestamp"])

tensor_shape = (
    full_df["user_encoded"].max() + 1,
    full_df["item_encoded"].max() + 1,
    full_df["time_encoded"].max() + 1,
)

tensor = np.zeros(tensor_shape)
tensor[full_df["user_encoded"], full_df["item_encoded"], full_df["time_encoded"]] = (
    full_df["rate"].values
)

# THe tensor/array shape is: (33, 2519, 514)
averaged_data = np.mean(tensor, axis=2)  # Shape will be (33, 2519)

# print(plt.style.available)
# plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# 1. Heatmap
plt.figure(figsize=(20, 8))
sample_size = min(100, averaged_data.shape[1])
sampled_indices = np.linspace(0, averaged_data.shape[1] - 1, sample_size, dtype=int)
sampled_data = averaged_data[:, sampled_indices]

sns.heatmap(
    sampled_data,
    cmap="viridis",
    xticklabels=False,
    yticklabels=True,
    cbar_kws={"label": "Average Recommendation Strength"},
)
plt.title(
    "User-Item Interaction Patterns\nShowing recommendation strength between users and items"
)
plt.xlabel("Items (Sampled) - Each column represents a different item")
plt.ylabel("Users - Each row represents a different user")
plt.show()

# 2. User Behavior Analysis
plt.figure(figsize=(15, 6))
user_means = np.mean(averaged_data, axis=1)
user_stds = np.std(averaged_data, axis=1)

plt.errorbar(
    range(len(user_means)), user_means, yerr=user_stds, fmt="o", capsize=5, markersize=8
)
plt.title(
    "User Interaction Patterns and Variability\nError bars show consistency in user behavior"
)
plt.xlabel("User ID - Each point represents a unique user")
plt.ylabel(
    "Average Recommendation Score\nHigher values indicate stronger recommendations"
)
plt.grid(True, alpha=0.3)
plt.show()

# 3. Item Performance Distribution
plt.figure(figsize=(15, 6))
item_means = np.mean(averaged_data, axis=0)
item_stds = np.std(averaged_data, axis=0)

sns.kdeplot(item_means, fill=True)
plt.title(
    "Distribution of Item Performance\nShowing how items are typically rated across all users"
)
plt.xlabel("Average Item Score\nHigher values indicate better performing items")
plt.ylabel("Density\nHigher values indicate more items with this score")
plt.grid(True, alpha=0.3)
plt.show()

# 4. Correlation Matrix for Items (using a sample)
plt.figure(figsize=(12, 10))
sample_size_corr = min(100, averaged_data.shape[1])
sampled_data_corr = averaged_data[:, :sample_size_corr]
correlation_matrix = np.corrcoef(sampled_data_corr.T)

sns.heatmap(
    correlation_matrix, cmap="coolwarm", center=0, xticklabels=False, yticklabels=False
)
plt.title("Item-Item Correlation Matrix\nShowing how items are related to each other")
plt.xlabel("Items - Each cell shows correlation between two items")
plt.ylabel("Items - Red indicates positive correlation, Blue indicates negative")
plt.show()


# 5. Top and Bottom Items Analysis
n_top = 10
top_items_idx = np.argsort(item_means)[-n_top:]
bottom_items_idx = np.argsort(item_means)[:n_top]

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.bar(range(n_top), item_means[top_items_idx][::-1])
plt.title("Top 10 Performing Items")
plt.xlabel("Item Rank")
plt.ylabel("Average Response")

plt.subplot(1, 2, 2)
plt.bar(range(n_top), item_means[bottom_items_idx])
plt.title("Bottom 10 Performing Items")
plt.xlabel("Item Rank")
plt.ylabel("Average Response")
plt.tight_layout()
plt.show()


# 6. User Engagement Analysis
plt.figure(figsize=(15, 6))
# Define threshold for considering an item as "engaged with"
threshold = np.mean(averaged_data)
user_activity = np.sum(averaged_data > threshold, axis=1)

plt.bar(range(len(user_activity)), sorted(user_activity, reverse=True))
plt.title(
    "User Engagement Levels\nShowing how many items each user significantly interacts with"
)
plt.xlabel("User Rank\nUsers sorted by engagement level")
plt.ylabel("Number of Items Engaged With\nBased on above-average interactions")
plt.grid(True, alpha=0.3)
plt.show()

# 7. Item Coverage Analysis
plt.figure(figsize=(15, 6))
item_coverage = np.sum(averaged_data > threshold, axis=0)
coverage_pct = (item_coverage / averaged_data.shape[0]) * 100

plt.hist(coverage_pct, bins=50, edgecolor="black")
plt.title("Item Coverage Distribution\nShowing how widely items are recommended")
plt.xlabel(
    "Percentage of Users Receiving Recommendations\nHigher values indicate more widely recommended items"
)
plt.ylabel("Number of Items\nCount of items in each coverage range")
plt.grid(True, alpha=0.3)
plt.show()

# 8. Recommendation Strength Distribution
plt.figure(figsize=(15, 6))
plt.hist2d(
    np.repeat(range(averaged_data.shape[0]), averaged_data.shape[1]),
    averaged_data.flatten(),
    bins=(33, 50),
    cmap="viridis",
)
plt.colorbar(label="Frequency")
plt.title(
    "Distribution of Recommendation Strengths Across Users\nShowing how recommendations vary for each user"
)
plt.xlabel("User ID\nEach column represents a user")
plt.ylabel("Recommendation Strength\nHigher values indicate stronger recommendations")
plt.show()
