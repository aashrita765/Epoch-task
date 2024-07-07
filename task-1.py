import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("clustering_data.csv", dtype= {'Latitude': 'str'})
df_telangana = df[df["StateName"] == 'TELANGANA']
df_telangana = df_telangana.drop_duplicates()
df_telangana = df_telangana.dropna()
df_telangana['Latitude'] = df_telangana['Latitude'].astype(float)
df_telangana['Longitude'] = df_telangana['Longitude'].astype(float)

# plot the original points
plt.scatter(df_telangana['Longitude'], df_telangana['Latitude'])
plt.title('Geographical Distribution of Pincodes in Telangana')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

## k means clustering
# Initialize random centroids
data = df_telangana[['Latitude','Longitude']]

def random_centroids(data, k):
    centroids = data.sample(n=k).to_numpy()
    return centroids

# Label each data point
def get_labels(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Update centroids
def update_centroids(data, labels, k):
    centroids = np.array([data[labels == i].mean(axis = 0) for i in range(k)])
    return centroids

# Iterating

def kmeans(data, k, max_iterations = 100):
    centroids = random_centroids(data, k)
    old_centroids = np.zeros_like(centroids)
    iteration = 1

    while iteration< max_iterations and not np.all(centroids==old_centroids):
        old_centroids = centroids
        labels = get_labels(data.to_numpy(), centroids)
        centroids = update_centroids(data.to_numpy(), labels, k)
        iteration+=1

    return labels, centroids

# Elbow method

def compute_wcss(data, max_k):
    wcss =[]
    for k in range(1, max_k+1):
        labels, centroids = kmeans(data, k)
        wcss_k = 0
        for i in range(k):
            cluster_point = data.to_numpy()[labels == i]
            wcss_k += np.sum((cluster_point - centroids[i])**2)
        wcss.append(wcss_k)
    return wcss

# Silhouette method
def compute_silhouette(data, max_k):
    silhouette_scores = []
    for k in range(2, max_k + 1): 
        labels, centroids = kmeans(data, k)
        silhouette_scores_k = []
        for i in range(len(data)):
            same_cluster = data[labels == labels[i]]
            other_clusters = data[labels != labels[i]]
            if len(same_cluster) > 1:
                a = np.mean(np.linalg.norm(same_cluster - data.iloc[i], axis=1))
            else:
                a = 0
            b = np.min([np.mean(np.linalg.norm(data[labels == label] - data.iloc[i], axis=1)) for label in set(labels) if label != labels[i]])
            silhouette_scores_k.append((b - a) / max(a, b))
        silhouette_scores.append(np.mean(silhouette_scores_k))
    return silhouette_scores



max_k = 10
# visualisation of elbow method
wcss = compute_wcss(data, max_k)
plt.figure(figsize = (10,6))
plt.plot(range(1, max_k+1), wcss, marker = 'o')
plt.title("Elbo method to find k")
plt.xlabel("Number of clusters(k)")
plt.ylabel("wcss")
plt.show()

# visualisation of silhoutte method
silhouette_scores = compute_silhouette(data, max_k)
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
plt.title("Silhouette Method to find k")
plt.xlabel("Number of clusters(k)")
plt.ylabel("Silhouette Score")
plt.show()

# after both methods chose optimal k
k =5
labels, centroids = kmeans(data,k)
plt.figure(figsize = (10,6))
plt.scatter(df_telangana['Longitude'], df_telangana['Latitude'], c=labels, cmap = 'viridis', marker = 'o', edgecolors= 'k', s =50)
plt.title("Clusters of Pincodes")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label = "Cluster")
plt.show()
