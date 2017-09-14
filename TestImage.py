# To check the validity of the algorithm first.
# code the functions and try to extract the image from each test image function
# and then calculate the error in the resulting image.

# Importing the required libraries which will be used in the process
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Set the parameters for the function so as to generate a proper graph and image boundaries
m = 100.0
n = 100.0
pi = 3.14
lx = 90.0
ly = 30.0


# Taking the one component of the function as x axis pixel of image(x,y)
def f(x):
    return 1.0 + np.cos((m * pi * x) / lx)


# Taking the one component of the function as y axis pixel of image(x,y)
def g(y):
    return 1.0 + np.cos((n * pi * y) / ly)


# Calculating the laplacian of the equation, one component turns out to be the below function
# for x axis pixel of image(x,y)
def f1(a, b):
    return (((- np.cos((m * pi * a) / lx)) * m ** 2 * pi ** 2) / lx ** 2) * (1.0 + np.cos((n * pi * b) / ly))


# Function for y axis pixel of image(x,y)
def g1(q, r):
    return (((- np.cos((n * pi * q) / ly)) * n ** 2 * pi ** 2) / ly ** 2) * (1.0 + np.cos((m * pi * r) / lx))

# Range of values to be passed inside the test image
c = np.arange(1, 150, 1)
d = np.arange(1, 150, 1)

# Displaying the graph for linear function
plt.plot(c, f(c), 'r--', d, g(d), 'k')
plt.show()

# Create a new black image
img1 = Image.new('RGB', (150, 150), "black")
# Create the pixel map
pixels = img1.load()

# For every pixel
for i in range(img1.size[0]):
    for j in range(img1.size[1]):
        # Set the colour accordingly
        pixels[i, j] = (int(f(i)), int(g(j)), 100)

# Save the image created by the process
img1.save("test2.png")
# Convert to numpy array and display the array
r = np.array(img1)
print r
# Display the image
plt.imshow(img1)
plt.show()

# Create a new black image
img = Image.new('RGB', (150, 150), "black")
# Create the pixel map
pixels = img.load()

# For every pixel
for i in range(img.size[0]):
    for j in range(img.size[1]):
        # Set the colour accordingly
        pixels[i, j] = (int(f1(i, j)), int(g1(j, i)), 100)

# Save the image created by the process
img.save("test1.png")
# Convert to numpy array and display the array
o = np.array(img)
print o
# Display the image
plt.imshow(img)
plt.show()

# Set the parameters for the Iteration of the algorithm

# As the image shape, we define the Boundary for the initial part
jm = 150
tx = 1
ty = 1
tm = 5
# Multiple values tried for the below variable
omega = 0.8
i = 0
j = 0
nu = 2
# Set the constant for the decreasing function
beta = 3
g = np.empty(o.shape)
# Set an array with initial values for later use
err = np.empty(o.shape)

#SOR implementation (Iteration inside the pixels)
for i in range(1, jm - 1, 1):
    for j in range(1, jm - 1, 1):
        o[i][j] = o[i][j] + omega * 0.25 * (o[i + 1][j] + o[i - 1][j] + o[i][j + 1] + o[i][j - 1] - 4 * o[i][j])
        for n in range(1, tm - 1, 1):
            # Calculate the Anisotropic Diffusion on the region
            # Here the diffusion function g is monotonically decreasing and
            # depends on the size of the gradient of the image
            g[i][j] = 1 / (1 + beta * ((((o[i + 1][j] - o[i - 1][j]) / tx) * 2) + (((o[i][j + 1] - o[i][j - 1]) / ty) * 2)))
            o[i][j] = o[i][j] + omega * ((nu * ((0.5 * (((g[i + 1][j] + g[i][j]) * (o[i + 1][j] - o[i][j]) -
                                                         ((g[i][j] + g[i - 1][j]) * (o[i][j] - o[i - 1][j]))) / tx))) +
                                          (nu * ((0.5 * ((g[i][j + 1] + g[i][j]) * (o[i][j + 1] - o[i][j]) -
                                                         ((g[i][j - 1] + g[i][j]) * (o[i][j] - o[i][j - 1]))) / ty)))))

        # Comparing value from the original image after it went through w = div I
        err[i][j] = o[i][j] - r[i][j]

# Display the error in the image at each pixel
print err

# Display the resultant image
cv2.imshow('n', o)

# Creates a plotting area for the figure
plt.imshow(o, cmap='viridis')
plt.show()
