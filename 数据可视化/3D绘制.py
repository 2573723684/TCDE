import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
pa = 'F9result.txt'
if pa == 'F8result.txt':
    upper_limit = 10
    lower_limit = -10
elif pa == 'F9result.txt':
    upper_limit = 10
    lower_limit = 0.25
elif pa == 'F14result.txt' or pa == 'F15result.txt':
    upper_limit = 5
    lower_limit = -5

# Read the data from the text file
with open(pa, 'r') as file:
    text = file.read()

# Extract numbers using regular expression
numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
numbers_list = list(map(float, numbers))

# Reshape the list into a 2D array
data = np.array(numbers_list).reshape(-1, 3)

# Extract x, y, z coordinates
x_coords = data[:, 0]
y_coords = data[:, 1]
z_coords = data[:, 2]

# Set upper and lower limits

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')

# Set axis limits
ax.set_xlim([lower_limit, upper_limit])
ax.set_ylim([lower_limit, upper_limit])
ax.set_zlim([lower_limit, upper_limit])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Set plot title
plt.title("3D + " + pa)

plt.show()
