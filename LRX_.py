import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def LRXfunc(image, InnerWindowSize, OuterWindowSize, delta):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    Length_image, Width_image, Bands_image = image.shape
    num_pixel = Length_image * Width_image
    image_transform = image.reshape(num_pixel, Bands_image).astype(float)

    d_LRX = np.zeros((Length_image, Width_image))

    temp_outer = (OuterWindowSize - 1) // 2
    large_image = np.tile(image, (3, 3, 1))
    large_image = large_image[Length_image - temp_outer : 2 * Length_image + temp_outer,
                              Width_image - temp_outer : 2 * Width_image + temp_outer, :]
    
    Background_Window = np.ones((OuterWindowSize, OuterWindowSize))
    Background_Window[(OuterWindowSize - InnerWindowSize) // 2 : (OuterWindowSize + InnerWindowSize) // 2,
                      (OuterWindowSize - InnerWindowSize) // 2 : (OuterWindowSize + InnerWindowSize) // 2] = 0
    ID_Window = np.where(Background_Window.flatten() == 1)[0]

    for i in range(Length_image):
        for j in range(Width_image):
            Background_Area = large_image[i : i + OuterWindowSize, j : j + OuterWindowSize, :]
            Background = Background_Area.reshape(OuterWindowSize * OuterWindowSize, Bands_image)
            Background = Background[ID_Window, :]  # Extract background pixels

            if Background.shape[0] == 0:
                print(f"Skipping pixel ({i}, {j}): Background size is zero.")
                continue
            
            if Background.shape[1] != Bands_image:
                print(f"Skipping pixel ({i}, {j}): Incorrect Background shape: {Background.shape}")
                continue

            mean_Background = np.mean(Background, axis=0)
            cov_Background = np.dot(Background.T, Background)

            if cov_Background.shape[0] != cov_Background.shape[1]:
                print(f"Skipping pixel ({i}, {j}): Incorrect covariance matrix dimensions: {cov_Background.shape}")
                continue

            BInv = np.linalg.inv(cov_Background + delta * np.eye(cov_Background.shape[0]))
            
            pixel_diff = image[i, j, :].reshape(1, Bands_image) - mean_Background
            d_LRX[i, j] = np.dot(np.dot(pixel_diff, BInv), pixel_diff.T)

    d_LRX = d_LRX.reshape(Length_image, Width_image)
    d_LRX_show = (d_LRX - np.min(d_LRX)) / (np.max(d_LRX) - np.min(d_LRX))
    
    return d_LRX_show

# Load the data
data = scipy.io.loadmat(r'C:\Users\Yazat\OneDrive\Desktop\Anomaly_Codes\Data\pavia.mat')['data']  # Replace 'data' with the actual variable name in the .mat file

# Run the LRX function
d_LRX_show = LRXfunc(data, 3, 5, 1e-5)

# Display the result
plt.imshow(d_LRX_show, cmap='jet')
plt.colorbar()
plt.show()
