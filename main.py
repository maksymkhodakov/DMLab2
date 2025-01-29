# Import Libraries
import os  # Added for directory operations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Ensure the 'images' directory exists
if not os.path.exists('images'):
    os.makedirs('images')

# 1. Data Preprocessing

# a. Loading Data
# Read the dataset from a CSV file into a pandas DataFrame
data = pd.read_csv('Mall_Customers.csv')

# b. Data Cleaning

# Display the number of missing values per column to identify any incomplete data
print("Missing values per column:")
print(data.isnull().sum())

# Display the number of duplicate rows to ensure data uniqueness
print("\nNumber of duplicate rows:", data.duplicated().sum())

# Remove duplicate rows to prevent bias in analysis
data = data.drop_duplicates()

# Outlier detection and removal for 'Age'
# Calculate the first (Q1) and third (Q3) quartiles
Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)
IQR = Q3 - Q1  # Interquartile Range

# Define the lower and upper bounds for outliers
age_lower = Q1 - 1.5 * IQR
age_upper = Q3 + 1.5 * IQR

# Filter the data to remove outliers in the 'Age' column
data = data[(data['Age'] >= age_lower) & (data['Age'] <= age_upper)]

# Similarly, outlier detection and removal for 'Annual Income (k$)'
Q1 = data['Annual Income (k$)'].quantile(0.25)
Q3 = data['Annual Income (k$)'].quantile(0.75)
IQR = Q3 - Q1
income_lower = Q1 - 1.5 * IQR
income_upper = Q3 + 1.5 * IQR
data = data[(data['Annual Income (k$)'] >= income_lower) & (data['Annual Income (k$)'] <= income_upper)]

# Similarly, outlier detection and removal for 'Spending Score (1-100)'
Q1 = data['Spending Score (1-100)'].quantile(0.25)
Q3 = data['Spending Score (1-100)'].quantile(0.75)
IQR = Q3 - Q1
spending_lower = Q1 - 1.5 * IQR
spending_upper = Q3 + 1.5 * IQR
data = data[(data['Spending Score (1-100)'] >= spending_lower) & (data['Spending Score (1-100)'] <= spending_upper)]

# c. Feature Selection
# Select relevant features for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# d. Data Normalization
# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Exploratory Data Analysis (EDA)

# Descriptive Statistics
# Provide summary statistics for the selected features
print("\nDescriptive Statistics:")
print(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())

# Separate Histograms

# Histogram for Age
plt.figure(figsize=(8,6))
sns.histplot(data['Age'], bins=10, kde=True, color='blue')
plt.title('Гістограма Віку')  # Title in Ukrainian: Age Histogram
plt.xlabel('Вік')  # X-axis label: Age
plt.ylabel('Частота')  # Y-axis label: Frequency
plt.savefig('images/hist_age.png')  # Save the Age histogram
plt.show()  # Display the plot

# Histogram for Annual Income
plt.figure(figsize=(8,6))
sns.histplot(data['Annual Income (k$)'], bins=10, kde=True, color='green')
plt.title('Гістограма Річного Доходу')  # Annual Income Histogram
plt.xlabel('Річний Дохід (k$)')  # Annual Income (k$)
plt.ylabel('Частота')  # Frequency
plt.savefig('images/hist_income.png')  # Save the Annual Income histogram
plt.show()

# Histogram for Spending Score
plt.figure(figsize=(8,6))
sns.histplot(data['Spending Score (1-100)'], bins=10, kde=True, color='red')
plt.title('Гістограма Рейтингу Витрат')  # Spending Score Histogram
plt.xlabel('Рейтинг Витрат (1-100)')  # Spending Score (1-100)
plt.ylabel('Частота')  # Frequency
plt.savefig('images/hist_spending.png')  # Save the Spending Score histogram
plt.show()

# Pairplot
# Visualize pairwise relationships between features
pairplot_fig = sns.pairplot(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], diag_kind='kde')
pairplot_fig.savefig('images/pairplot.png')  # Save the pairplot
plt.show()

# Correlation Matrix
# Compute and visualize the correlation matrix to understand relationships between features
plt.figure(figsize=(8,6))
corr = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')  # Title for the correlation matrix
plt.savefig('images/correlation_matrix.png')  # Save the correlation matrix plot
plt.show()

# 3. Implementing K-Means Clustering

# a. Choosing the Number of Clusters (k) - Elbow Method
# The Elbow Method helps determine the optimal number of clusters by plotting WCSS against different k values
wcss = []  # List to store Within-Cluster Sum of Squares
K_range = range(1, 11)  # Testing k from 1 to 10
for k_val in K_range:
    kmeans = KMeans(n_clusters=k_val, random_state=42)  # Initialize KMeans with current k
    kmeans.fit(X_scaled)  # Fit KMeans to the scaled data
    wcss.append(kmeans.inertia_)  # Append the WCSS to the list

# Plot the Elbow Method graph
plt.figure(figsize=(8,5))
plt.plot(K_range, wcss, 'bo-')  # Plot k vs WCSS
plt.xlabel('Number of Clusters (k)')  # X-axis label
plt.ylabel('WCSS')  # Y-axis label
plt.title('Elbow Method for Optimal k')  # Title of the plot
plt.savefig('images/elbow_method.png')  # Save the Elbow Method plot
plt.show()  # Display the plot

# a. Choosing the Number of Clusters (k) - Silhouette Analysis
# Silhouette Analysis measures how similar an object is to its own cluster compared to other clusters
silhouette_scores = []  # List to store silhouette scores
K_silhouette = range(2, 11)  # Silhouette score is not defined for k=1
for k_val in K_silhouette:
    kmeans = KMeans(n_clusters=k_val, random_state=42)  # Initialize KMeans with current k
    cluster_labels = kmeans.fit_predict(X_scaled)  # Fit KMeans and predict cluster labels
    score = silhouette_score(X_scaled, cluster_labels)  # Calculate silhouette score
    silhouette_scores.append(score)  # Append the score to the list

# Plot the Silhouette Analysis graph
plt.figure(figsize=(8,5))
plt.plot(K_silhouette, silhouette_scores, 'bo-')  # Plot k vs Silhouette Score
plt.xlabel('Number of Clusters (k)')  # X-axis label
plt.ylabel('Silhouette Score')  # Y-axis label
plt.title('Silhouette Analysis for Optimal k')  # Title of the plot
plt.savefig('images/silhouette_analysis.png')  # Save the Silhouette Analysis plot
plt.show()  # Display the plot

# Based on Elbow and Silhouette, choose k=5
k = 5  # Optimal number of clusters determined from the plots
kmeans = KMeans(n_clusters=k, random_state=42)  # Initialize KMeans with k=5
clusters = kmeans.fit_predict(X_scaled)  # Fit KMeans and predict cluster labels
data['Cluster'] = clusters  # Assign cluster labels to the original data

# b. Cluster Analysis
# Summarize the characteristics of each cluster
cluster_summary = data.groupby('Cluster').agg({
    'Age': 'mean',  # Average age in each cluster
    'Annual Income (k$)': 'mean',  # Average annual income in each cluster
    'Spending Score (1-100)': 'mean',  # Average spending score in each cluster
    'CustomerID': 'count'  # Number of customers in each cluster
}).rename(columns={'CustomerID': 'Count'}).reset_index()

# Display the cluster summary
print("\nCluster Summary:")
print(cluster_summary)

# 4. Evaluation of Clusters
# Calculate the overall silhouette score to evaluate clustering quality
score = silhouette_score(X_scaled, data['Cluster'])
print(f'\nSilhouette Score for k={k}: {score:.4f}')  # Display the silhouette score

# 5. Visualization

# a. PCA for Dimensionality Reduction
# Reduce the data to 2 dimensions for visualization purposes
pca = PCA(n_components=2)  # Initialize PCA to reduce to 2 principal components
principal_components = pca.fit_transform(X_scaled)  # Fit PCA and transform the data
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])  # Create a DataFrame for PCA results
pca_df['Cluster'] = data['Cluster']  # Add cluster labels to the PCA DataFrame

# b. Scatter Plot of Clusters
# Visualize the clusters in the PCA-reduced feature space
plt.figure(figsize=(8,6))
scatter_fig = sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='Cluster',
    data=pca_df,
    palette='Set1',
    alpha=0.6
)  # Create a scatter plot with clusters colored differently
plt.title('Customer Segments (PCA Reduced)')  # Title of the plot
plt.legend(title='Cluster')  # Add legend with title
plt.savefig('images/pca_clusters.png')  # Save the PCA scatter plot
plt.show()  # Display the plot

# c. Cluster Centroids in Original Scale
# Transform the cluster centers back to the original scale for interpretation
centroids = scaler.inverse_transform(kmeans.cluster_centers_)  # Inverse transform to original feature scale
centroid_df = pd.DataFrame(centroids, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])  # Create DataFrame for centroids
print("\nCluster Centroids (Original Scale):")
print(centroid_df)  # Display the centroids

# d. Additional Visualizations

# Boxplot for Age by Cluster
plt.figure(figsize=(8,6))
sns.boxplot(
    x='Cluster',
    y='Age',
    hue='Cluster',
    data=data,
    palette='Set2'
)  # Create a boxplot for Age distribution across clusters
plt.title('Age Distribution by Cluster')  # Title of the plot
plt.legend([], [], frameon=False)  # Disable the legend as hue is redundant with x-axis
plt.savefig('images/boxplot_age.png')  # Save the boxplot
plt.show()  # Display the plot

# Boxplot for Annual Income by Cluster
plt.figure(figsize=(8,6))
sns.boxplot(
    x='Cluster',
    y='Annual Income (k$)',
    hue='Cluster',
    data=data,
    palette='Set2'
)  # Create a boxplot for Annual Income distribution across clusters
plt.title('Annual Income Distribution by Cluster')  # Title of the plot
plt.legend([], [], frameon=False)  # Disable the legend as hue is redundant with x-axis
plt.savefig('images/boxplot_income.png')  # Save the boxplot
plt.show()  # Display the plot

# Boxplot for Spending Score by Cluster
plt.figure(figsize=(8,6))
sns.boxplot(
    x='Cluster',
    y='Spending Score (1-100)',
    hue='Cluster',
    data=data,
    palette='Set2'
)  # Create a boxplot for Spending Score distribution across clusters
plt.title('Spending Score Distribution by Cluster')  # Title of the plot
plt.legend([], [], frameon=False)  # Disable the legend as hue is redundant with x-axis
plt.savefig('images/boxplot_spending.png')  # Save the boxplot
plt.show()  # Display the plot
