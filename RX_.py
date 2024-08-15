import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def RXfunc(x):
    # Input: 3D HSI
    H, W, B = x.shape
    n = H * W
    
    x = normalization(x, b=1)
    x = x.reshape(n, B).T  # B, n

    m = np.mean(x, axis=1, keepdims=True)
    data = x - np.tile(m, (1, n))

    c = np.cov(data)
    
    R = np.dot(np.dot(data.T, np.linalg.inv(c)), data)
    R = np.diag(R)
    R = R.reshape(H, W)

    return R

def normalization(s, b=0):
    if b == 0:
        max_s = np.max(s)
        min_s = np.min(s)
        s_new = (s - min_s) / (max_s - min_s)
    elif b == 1:
        bands = s.shape[2]
        s_new = np.zeros_like(s)
        for i in range(bands):
            max_s = np.max(s[:, :, i])
            min_s = np.min(s[:, :, i])
            s_new[:, :, i] = (s[:, :, i] - min_s) / (max_s - min_s)
    return s_new

# Load the data
data = scipy.io.loadmat(r'<path_to_your_dataset>')['<mat_key>']  # Replace 'data' with the actual variable name in the .mat file

# Run the RX function
R = RXfunc(data)
RR = normalization(R)

# Display the result
plt.imshow(RR, cmap='jet')
plt.colorbar()
plt.show()
