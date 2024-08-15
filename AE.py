import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_autoencoder(input_dim, latent_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(latent_dim, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    return model

def load_data(file_path):
    mat = sio.loadmat(file_path)
    return mat['img']

def main():
    file_path = r"<path_to_your_dataset>"  
    data = load_data(file_path)

    rows, cols, bands = data.shape
    data_normalized = data.astype('float32') / np.max(data)

    data_flattened = np.reshape(data_normalized, (rows * cols, bands))

    input_dim = bands
    latent_dim = 32

    autoencoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    autoencoder.fit(data_flattened, data_flattened, epochs=100, batch_size=128, validation_split=0.1, shuffle=True, verbose=1, callbacks=[reduce_lr, early_stopping])

    reconstructed = autoencoder.predict(data_flattened)
    reconstruction_error = np.mean(np.square(data_flattened - reconstructed), axis=1)

    threshold = np.percentile(reconstruction_error, 95)

    anomalies = reconstruction_error > threshold

    anomalies_image = np.reshape(anomalies, (rows, cols))

    # Plot the anomalies using a jet colormap with increased brightness
    plt.figure(figsize=(10, 8))
    anomaly_brightness = np.zeros_like(data_normalized[:, :, 0], dtype=float)
    anomaly_brightness[anomalies_image] = 1.0  # Highlight anomalies with increased brightness

    plt.imshow(data_normalized[:, :, 0], cmap='jet')
    plt.imshow(anomaly_brightness, cmap='hot', alpha=0.9)
    plt.colorbar(label='Anomaly Intensity')

    plt.title('Anomalies on Hyperspectral Image')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.show()

if __name__ == "__main__":
    main()
