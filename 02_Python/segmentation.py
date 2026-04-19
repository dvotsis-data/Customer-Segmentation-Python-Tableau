import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. DATA INGESTION
# Load the raw customer dataset from the data directory
df = pd.read_csv('../01_Data/raw_customers.csv')

# 2. DATA PRE-PROCESSING
# Rename columns for standardization and ease of access
df.columns = ['CustomerID', 'Gender', 'Age', 'Income', 'SpendingScore']

# 3. FEATURE SELECTION
# Selecting key features for the clustering model: Annual Income and Spending Score
X = df[['Income', 'SpendingScore']]

# 4. ELBOW METHOD FOR OPTIMAL K
# Calculate Within-Cluster Sum of Square (WCSS) for different cluster counts
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Visualization of the Elbow Curve to identify the 'elbow' point
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# 5. K-MEANS MODEL IMPLEMENTATION
# Applying K-Means algorithm using the optimal number of clusters (k=5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 6. CLUSTER VISUALIZATION
# Visualizing the final segments based on Income and Spending Score
plt.figure(figsize=(10, 6))
plt.scatter(df['Income'], df['SpendingScore'], c=df['Cluster'], cmap='viridis', s=100)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation Analysis')
plt.colorbar(label='Cluster ID')
plt.show()

# 7. DATA EXPORT
# Export the segmented data for further visualization in Tableau
df.to_csv('../01_Data/cleaned_customers.csv', index=False)

print("Segmentation Analysis Complete. Cleaned data exported successfully.")