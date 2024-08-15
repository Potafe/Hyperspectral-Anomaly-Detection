import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from tqdm import tqdm

# Load the .mat file containing hyperspectral data
print("Loading hyperspectral data...")
mat_data = loadmat(r'<path_to_your_dataset>')

# Extract hyperspectral data
hyperspectral_data = mat_data['<mat_key>']  # Replace 'data' with the actual variable name in your .mat file

# Assuming 'hyperspectral_data' is a 3D array: (height, width, num_bands)
height, width, num_bands = hyperspectral_data.shape

# Reshape the data to have pixels as rows and bands as columns
reshaped_data = np.reshape(hyperspectral_data, (height * width, num_bands))

# Step 1: Normalize the data
print("Normalizing data...")
scaler = StandardScaler()
normalized_data = scaler.fit_transform(reshaped_data)

# Step 2: Apply PCA for dimensionality reduction
print("Applying PCA for dimensionality reduction...")
pca_components = 5  # Reduce the number of components further
pca = PCA(n_components=pca_components)
pca_data = pca.fit_transform(normalized_data)
print(f"Data reduced to {pca_components} components.")

# Step 3: Train a One-Class SVM model
print("Training One-Class SVM model...")
svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)  # Adjust gamma and nu parameters as needed
svm_model.fit(pca_data)
print("One-Class SVM model training complete.")

# Step 4: Predict anomalies in batches
print("Detecting anomalies...")

# Initialize anomaly scores
anomaly_scores = np.zeros(len(pca_data))

# Define batch size
batch_size = 1000  # Adjust based on memory constraints

# Create a progress bar
pbar = tqdm(total=len(pca_data), desc="Processing Batches")

# Process data in batches
for start in range(0, len(pca_data), batch_size):
    end = min(start + batch_size, len(pca_data))
    batch_data = pca_data[start:end]
    anomaly_scores[start:end] = svm_model.decision_function(batch_data)
    pbar.update(end - start)

# Close progress bar
pbar.close()

print("Anomaly scores computed.")

# Determine anomaly threshold
threshold_score = np.percentile(anomaly_scores, 5)  # Adjust percentile as per your specific application

# Step 5: Create anomaly map
anomaly_map = anomaly_scores < threshold_score
anomaly_map = np.reshape(anomaly_map, (height, width))

# Step 6: Visualize results
print("Visualizing results...")
plt.figure(figsize=(10, 8))

# Display the image with anomalies using a jet colormap
anomaly_brightness = np.zeros_like(hyperspectral_data[:, :, 0], dtype=float)
anomaly_brightness[anomaly_map] = 1.0  # Highlight anomalies with increased brightness

plt.imshow(anomaly_brightness, cmap='jet', alpha=0.7)
plt.colorbar(label='Anomaly Intensity')

plt.title('Hyperspectral Image with Anomalies (One-Class SVM)')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.show()

print("Anomaly detection and visualization complete.")
