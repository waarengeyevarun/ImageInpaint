# The Transport inpainting method is defined here which is tweaked from the original version
# The Poisson problem is solved through SOR and Anisotropic diffusion
# The Transport equation is a combination of the diffusion and convection

# Importing the required libraries which will be used
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image through opencv,the image should be in the working directory or a full path of image should be given.
img = cv2.imread('orange.jpg')

# Convert the loaded image into a numpy array
# Working on numpy arrays is simpler and comes with a lot of inbuilt scientific computations
a = np.array(img)

# Because we want only a binary image and the numpy shape is(height, width[, depth])
height, width = a.shape[1::-1]

print height, width
print a.size

# As we want to create a mask and impose it on the image
mask = np.zeros(img.shape[:-1])

# Create couple of masks boundaries to fill in with values
mask[90:120, 120:130] = 1
mask[200:210, 130:150] = 1

# Fill the mask with black pixel values
b = img.copy()
for l in range(b.shape[-1]):
    b[np.where(mask)] = 0

# Display the image with holes/masks in it
cv2.imshow('g', b)

# Find the image pixels which went on missing
unfilled_pixels = np.where(b == 0)
# Create the numpy array for the image with holes in it
r = np.array(unfilled_pixels, np.int32)
# Display them
print r

# Set the parameters for the Iteration of the algorithm

# As the image shape, we define the Boundary for the initial part
hm = height
wm = width
tx = 1
ty = 1
# Multiple values tried for the below variable
alpha = 0.022
i = 0
j = 0
# Velocity divergent or Diffusion coefficient
nu = 2
# Set the decreasing function
beta = 3
# Number of iterations needed for Anisotropic Diffusion
tm = 3
g = np.empty(b.shape)

# SOR implementation (Iteration inside the pixels)
for i in range(1, hm-1, 1):
    for j in range(1, wm-1, 1):
        b[i][j] = b[i][j] + alpha * 0.25 * (b[i + 1][j] + b[i - 1][j] + b[i][j + 1] + b[i][j - 1] - 4.0 * a[i][j])
        # Calculate the Anisotropic Diffusion on the region
        # Here the diffusion function g is monotonically decreasing and
        # depends on the size of the gradient of the image
        for n in range(1, tm - 1, 1):
            g[i][j] = 1 / (1 + beta * ((((b[i + 1][j] - b[i - 1][j]) / tx)*2) + (((b[i][j + 1] - b[i][j - 1]) / ty)*2)))
            b[i][j] = b[i][j] + alpha * ((nu * ((0.5 * (((g[i+1][j] + g[i][j])*(b[i+1][j] - b[i][j]) -
                                                        ((g[i][j] + g[i-1][j])*(b[i][j] - b[i-1][j])))/tx))) +
                                         (nu * ((0.5 * ((g[i][j+1] + g[i][j])*(b[i][j+1] - b[i][j]) -
                                                        ((g[i][j-1] + g[i][j])*(b[i][j] - b[i][j-1])))/ty)))))

# Display the resultant image
cv2.imshow('n', b)

# Creates a plotting area for the figure
plt.imshow(b, cmap='viridis')
plt.show()

