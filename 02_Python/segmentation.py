import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. DATA INGESTION
# ============================================================

# Load the raw customer dataset
df = pd.read_csv('../01_Data/raw_customers.csv')


# ============================================================
# 2. DATA PREPARATION
# ============================================================

# Rename columns for consistency and easier analysis
df.columns = [
    'CustomerID',
    'Gender',
    'Age',
    'Income',
    'SpendingScore'
]


# ============================================================
# 3. DATA QUALITY CHECKS
# ============================================================

# Check dataset dimensions, missing values, duplicate IDs,
# and data types before performing clustering

print("DATA QUALITY CHECKS")

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate Customer IDs:")
print(df['CustomerID'].duplicated().sum())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 4. FEATURE SELECTION
# ============================================================

# Select Annual Income and Spending Score as clustering features
# Age and Gender are retained for customer profiling

X = df[['Income', 'SpendingScore']]


# ============================================================
# 5. ELBOW METHOD
# ============================================================

# Calculate Within-Cluster Sum of Squares (WCSS)
# for different numbers of clusters

wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        init='k-means++',
        random_state=42
    )

    model.fit(X)
    wcss.append(model.inertia_)

# Visualize the Elbow Method results
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(range(1, 11))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 6. SILHOUETTE SCORE COMPARISON
# ============================================================

# Evaluate clustering quality for k values from 2 to 10
# Higher scores generally indicate better cluster separation

print("\nSILHOUETTE SCORE COMPARISON")

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        init='k-means++',
        random_state=42
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    print(f"k={k}: {score:.4f}")


# ============================================================
# 7. FINAL K-MEANS MODEL
# ============================================================

# Use k=5 based on the Elbow Method, Silhouette Score
# comparison, and interpretability of customer segments

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

# Assign each customer to a cluster
df['Cluster'] = kmeans.fit_predict(X)

# Evaluate the final clustering model
final_score = silhouette_score(X, df['Cluster'])

print(f"\nFinal Silhouette Score (k=5): {final_score:.4f}")


# ============================================================
# 8. CLUSTER VISUALIZATION
# ============================================================

# Visualize customer segments using Annual Income
# and Spending Score

plt.figure(figsize=(10, 6))

plt.scatter(
    df['Income'],
    df['SpendingScore'],
    c=df['Cluster'],
    cmap='viridis',
    s=100
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation Analysis')

plt.colorbar(label='Cluster ID')

plt.tight_layout()
plt.show()


# ============================================================
# 9. CLUSTER PROFILING
# ============================================================

# Summarize the characteristics of each customer segment
# using customer count, average age, income, and spending score

cluster_profile = df.groupby('Cluster').agg(
    Customers=('CustomerID', 'count'),
    Avg_Age=('Age', 'mean'),
    Avg_Income=('Income', 'mean'),
    Avg_SpendingScore=('SpendingScore', 'mean')
).round(2)

print("\nCUSTOMER SEGMENT PROFILE")
print(cluster_profile)


# ============================================================
# 10. DATA EXPORT
# ============================================================

# Export customer records with their assigned cluster IDs
# for further analysis and visualization in Tableau

df.to_csv(
    '../01_Data/cleaned_customers.csv',
    index=False
)

print(
    "\nSegmentation Analysis Complete. "
    "Cleaned data exported successfully."
)
