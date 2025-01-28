# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Data Preprocessing

# a. Loading Data
data = pd.read_csv('Mall_Customers.csv')

# b. Data Cleaning
print("Missing values per column:")
print(data.isnull().sum())

print("\nNumber of duplicate rows:", data.duplicated().sum())
data = data.drop_duplicates()

# Outlier detection and removal for 'Age'
Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)
IQR = Q3 - Q1
age_lower = Q1 - 1.5 * IQR
age_upper = Q3 + 1.5 * IQR
data = data[(data['Age'] >= age_lower) & (data['Age'] <= age_upper)]

# Similarly for 'Annual Income (k$)'
Q1 = data['Annual Income (k$)'].quantile(0.25)
Q3 = data['Annual Income (k$)'].quantile(0.75)
IQR = Q3 - Q1
income_lower = Q1 - 1.5 * IQR
income_upper = Q3 + 1.5 * IQR
data = data[(data['Annual Income (k$)'] >= income_lower) & (data['Annual Income (k$)'] <= income_upper)]

# Similarly for 'Spending Score (1-100)'
Q1 = data['Spending Score (1-100)'].quantile(0.25)
Q3 = data['Spending Score (1-100)'].quantile(0.75)
IQR = Q3 - Q1
spending_lower = Q1 - 1.5 * IQR
spending_upper = Q3 + 1.5 * IQR
data = data[(data['Spending Score (1-100)'] >= spending_lower) & (data['Spending Score (1-100)'] <= spending_upper)]

# c. Feature Selection
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# d. Data Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Exploratory Data Analysis (EDA)

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())

# Histograms
data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].hist(bins=10, figsize=(10,7))
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], diag_kind='kde')
plt.show()

# Correlation Matrix
corr = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 3. Implementing K-Means Clustering

# a. Choosing the Number of Clusters (k) - Elbow Method
wcss = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, wcss, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.show()

# a. Choosing the Number of Clusters (k) - Silhouette Analysis
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8,5))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.show()

# Based on Elbow and Silhouette, choose k=5
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters

# b. Cluster Analysis
cluster_summary = data.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'}).reset_index()

print("\nCluster Summary:")
print(cluster_summary)

# 4. Evaluation of Clusters
score = silhouette_score(X_scaled, data['Cluster'])
print(f'\nSilhouette Score for k={k}: {score:.4f}')

# 5. Visualization

# a. PCA for Dimensionality Reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = data['Cluster']

# b. Scatter Plot of Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set1', alpha=0.6)
plt.title('Customer Segments (PCA Reduced)')
plt.show()

# c. Cluster Centroids in Original Scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
print("\nCluster Centroids (Original Scale):")
print(centroid_df)

# d. Additional Visualizations

# Age Distribution by Cluster
plt.figure(figsize=(8,6))
sns.boxplot(x='Cluster', y='Age', hue='Cluster', data=data, palette='Set2')
plt.title('Age Distribution by Cluster')
plt.show()

# Annual Income Distribution by Cluster
plt.figure(figsize=(8,6))
sns.boxplot(x='Cluster', y='Annual Income (k$)', hue='Cluster', data=data, palette='Set2')
plt.title('Annual Income Distribution by Cluster')
plt.show()

# Spending Score Distribution by Cluster
plt.figure(figsize=(8,6))
sns.boxplot(x='Cluster', y='Spending Score (1-100)', hue='Cluster', data=data, palette='Set2')
plt.title('Spending Score Distribution by Cluster')
plt.show()
