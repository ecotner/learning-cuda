import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read in the raw pixel data from the csv
img = pd.read_csv("image.csv")
img = img.values
# reshape the image from a vector of RGBA values to a matrix of RGBA values
print(img.shape)
img = img.reshape([512, 512, -1], order="F")
print(img.shape)
# generate the image and save it to a file
plt.imshow(img)
plt.savefig("image.png", bbox_inches="tight", dpi=150)
