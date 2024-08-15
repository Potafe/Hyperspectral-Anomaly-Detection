import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.io as sio

# Function to load hyperspectral data from a .mat file
def load_data(mat_file_path, variable_name):
    try:
        mat_contents = sio.loadmat(mat_file_path)
        return mat_contents[variable_name]
    except KeyError:
        raise ValueError(f"Variable '{variable_name}' not found in the .mat file.")
    except Exception as e:
        raise RuntimeError("Error loading the .mat file.") from e

# Function to preprocess the hyperspectral data
def preprocess_data(data):
    # Ensure data is float32 for processing
    data = data.astype(np.float32)

    # Remove bad bands based on variance
    variances = np.var(data, axis=(0, 1))
    variance_threshold = 0.05 * np.max(variances)  # Example threshold
    good_bands = variances > variance_threshold
    filtered_data = data[:, :, good_bands]

    # Normalize the data
    normalized_data = np.empty_like(filtered_data, dtype=np.float32)
    for i in range(filtered_data.shape[2]):
        band_min = np.min(filtered_data[:, :, i])
        band_max = np.max(filtered_data[:, :, i])
        if band_max != band_min:
            normalized_data[:, :, i] = (filtered_data[:, :, i] - band_min) / (band_max - band_min)
        else:
            normalized_data[:, :, i] = 0  # Handle division by zero

    return normalized_data

# Function to determine the optimal number of clusters using the Elbow Method
def find_optimal_clusters(data, max_k=10):
    wcss = []  # List to store within-cluster sums of squares
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data.reshape(-1, data.shape[2]))
        wcss.append(kmeans.inertia_)  # Inertia is the sum of squared distances of samples to their closest cluster center

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')  # Within-cluster sum of squares
    plt.show()

# Function to detect anomalies using k-means clustering
def detect_anomalies_kmeans(data, n_clusters):
    # Reshape data for clustering
    data_reshaped = data.reshape(-1, data.shape[2])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data_reshaped)
    cluster_centers = kmeans.cluster_centers_

    # Predict the cluster for each data point
    cluster_labels = kmeans.predict(data_reshaped)

    # Calculate the distance of each point to its cluster center
    distances = np.linalg.norm(data_reshaped - cluster_centers[cluster_labels], axis=1)

    # Define an anomaly threshold based on the 95th percentile of distances
    anomaly_threshold = np.percentile(distances, 96)

    # Identify anomalies
    anomalies = distances > anomaly_threshold

    return anomalies, distances, cluster_labels, kmeans

def plot_original_image(data):
    image_data = data[:, :, 29]
    plt.figure(figsize=(10, 8))
    plt.imshow(image_data, cmap='gray')
    plt.title('Original Image')
    plt.show()

# Function to plot anomalies on the image using jet colormap
def plot_anomalies_on_image(data, anomalies, height, width):
    # Reshape data for plotting
    image_data = data[:, :, 0]  # Using the first band as a grayscale image for simplicity

    # Create an anomaly image
    anomaly_image = np.zeros_like(image_data)

    # Mark anomalies on the image
    anomaly_indices = np.where(anomalies)[0]
    anomaly_positions = np.array([(idx // width, idx % width) for idx in anomaly_indices])
    anomaly_image[anomaly_positions[:, 0], anomaly_positions[:, 1]] = 1

    # Plot the original image with anomalies using jet colormap
    plt.figure(figsize=(10, 8))
    plt.imshow(image_data, cmap='jet', alpha=0.7)
    plt.imshow(anomaly_image, cmap='hot', alpha=0.3)  # Anomalies marked in bright color
    plt.colorbar(label='Intensity')
    plt.title('Anomalies on Hyperspectral Image')
    plt.show()

# Main process
if __name__ == '__main__':
    # Paths to the .mat file and the variable containing the hyperspectral data
    mat_file_path = r"C:\Users\Yazat\OneDrive\Desktop\Anomaly_Codes\Data\pavia.mat"
    variable_name = 'data'  # Replace with the actual variable name in the .mat file

    try:
        # Load and preprocess the data
        data = load_data(mat_file_path, variable_name)
        print(f"Data loaded with shape: {data.shape}")

        plot_original_image(data)

        normalized_data = preprocess_data(data)
        print(f"Data normalized with shape: {normalized_data.shape}")

        # Determine the optimal number of clusters using the Elbow Method
        find_optimal_clusters(normalized_data)

        # Assuming the optimal number of clusters is determined, for example, 3
        optimal_clusters = 6  # This should be set based on your Elbow Method result

        # Detect anomalies using k-means clustering
        anomalies, distances, cluster_labels, kmeans = detect_anomalies_kmeans(normalized_data, optimal_clusters)

        print(f"Anomalies detected: {np.sum(anomalies)}")

        # Plot the results
        plt.figure(figsize=(12, 10))
        plt.scatter(range(len(anomalies)), distances, c='b', marker='o', label='Normal')
        plt.scatter(np.where(anomalies)[0], distances[anomalies], c='r', marker='x', label='Anomaly')
        plt.axhline(y=np.percentile(distances, 95), color='r', linestyle='--', label='95th percentile threshold')
        plt.title('Anomaly Detection Using k-Means Clustering')
        plt.xlabel('Data Point Index')
        plt.ylabel('Distance to Cluster Center')
        plt.legend()
        plt.show()

        # Optionally, plot anomalies on the image
        plot_anomalies_on_image(data, anomalies, data.shape[0], data.shape[1])

    except Exception as e:
        print(f"An error occurred: {e}")


