import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# from IPython import clear_output
import matplotlib as mpl
import matplotlib.pyplot as plt
import gradio as gr

#specify number of centroids
#initialize centroids randomly
#assign each data point to the nearest centroid
# update the centroid based on the geometrinc mean of all the data points in the cluster
# iterate until the centroids stop changing


# read the file

def read_file(file_path,percentage):
    data=pd.read_csv(file_path)
    percentage=percentage/100
    sampled_data = data.sample(frac=percentage)
    sampled_data = sampled_data.reset_index(drop=True)
    #normaileze data using min max scaler 
    sampled_data = sampled_data.drop(columns=['CustomerID'], errors='ignore')
    
    scaler = MinMaxScaler()
    features_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    print(sampled_data.columns)
    data_scaled = scaler.fit_transform(sampled_data[features_to_scale])

# Create a new DataFrame with the scaled data
    normalized_data = pd.DataFrame(data_scaled, columns=features_to_scale)

    # Add 'Gender' column back
    normalized_data['Gender'] = sampled_data['Gender'].values
    normalized_data['Gender'] = normalized_data['Gender'].map({'Male': 0, 'Female': 1})

    print(normalized_data.describe())
    return normalized_data
    
def random_centroids(data, k):
    # Randomly select k data points as initial centroids without replacement by selecting random indices 
    # then iloc return the dataframes of those indices
    random_indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data.iloc[random_indices].reset_index(drop=True)

def find_nearest_centroid(data, centroids):
    distances = pd.DataFrame()
    # data = data.reset_index(drop=True)
    for idx, centroid in centroids.iterrows():
        dist = np.sqrt(((data - centroid) ** 2).sum(axis=1))
        distances[idx] = dist

    nearest_centroid = distances.idxmin(axis=1)
    return nearest_centroid

def update_centroids(data, nearest_centroid, k):
    new_centroids = []
    data['nearest_centroid'] = nearest_centroid.values
    for i in range(k):
        cluster_points = data[data['nearest_centroid'] == i]
        cluster_points = cluster_points.drop(columns=['nearest_centroid'], errors='ignore')
        new_centroid = cluster_points.mean(axis=0)
        new_centroids.append(new_centroid)
    return pd.DataFrame(new_centroids)

def plot_clusters(data, nearest_cluster, iteration, centroids):
    data_no_cluster = data.drop(columns=['nearest_centroid'], errors='ignore')
    plt.figure(figsize=(10, 6)) 
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_no_cluster)
    centroids_2d = pca.transform(centroids)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=nearest_cluster, cmap='viridis', marker='o', s=50)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title(f'Iteration {iteration}')
    plt.legend()
    plt.show()
    plt.close()

def find_outliers(data, nearest_centroid):
    # data['nearest_centroid'] = nearest_centroid.values
    data['distance_to_centroid'] = np.sqrt(((data - data.mean()) ** 2).sum(axis=1))
    threshold = data['distance_to_centroid'].quantile(0.95)  # top 5%
    outliers = data[data['distance_to_centroid'] > threshold]
    return outliers


#find the outlier calculate the distance between each point and it centroid
#if the distance of a point is grater than mean of distance plus 2 times the standard deviation
def k_means(k,percentage ,file_path):
    # Read the dataset
    data=read_file(file_path,percentage)
    # Initialize centroids randomly
    centroids = random_centroids(data, k)
    # Iterate until the centroids stop changing
    max_iterations = 100
    count =0
    while True:
        previous_centroids = centroids.copy()
        # Assign each data point to the nearest cluster
        nearest_centroid_clusters =find_nearest_centroid(data, centroids)
        #plot_clusters(data, nearest_centroid_clusters, count, centroids)
        centroids=update_centroids(data, nearest_centroid_clusters, k)

        if(centroids.equals(previous_centroids)):
            break
        count+=1
        if(max_iterations==count):
            break
    

    
    data['nearest_centroid'] = nearest_centroid_clusters.values
    outliers= find_outliers(data, nearest_centroid_clusters)
    data['Gender'] = data['Gender'].map({ 0:'Male',  1:"Female"})
    return data.sort_values(by='nearest_centroid').reset_index(drop=True),outliers



#print (k_means(4, 30, "data/SS2025_Clustering_SuperMarketCustomers.csv"))

interface = gr.Interface(
    fn=k_means,  # Function to be called
    inputs=[
        gr.Number(label="enter k"),
        gr.Number(label="Percentage of Dataset to Use"),
        gr.Textbox(label="file path")  # Hidden input for file path
    ],
    outputs="text",  # Output type
    title="Clustering Analysis",
    description="Enter the number of clusters, percentage of the dataset to be clustered and the file path"
)

# Launch the interface
interface.launch()
