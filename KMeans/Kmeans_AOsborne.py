#_________________________________________________________________________________________________________________________
# Author: Anastasiya Osborne
# Name: #Kmeans_AOsborne.py
# Program location:  ~\OneDrive\Documents\Kmeans_AOsborne.py


# Start: December 28, 2020
# Released: January 2, 2021
# Task: Implement a K-means clustering algorithm and apply it to compress an image in Python.
#_________________________________________________________________________________________________________________________

# Notes: 
# Please install the the following packages with 'pip install'
# If a program doesn't want to bring me to the home path, in a terminal, say cd ~\OneDrive\Documents\

from pathlib import Path
from skimage import io
from sklearn.cluster import KMeans
from termcolor import colored
import numpy as np

print(colored('==============================START OF PROGRAM=========================================================', 'yellow'))

# Note: Run this from a directory where image is located. 
file_to_open = "Palm.png"
	
#Read the image of a Palm in the home folder, size of 16,110 KB. 
Palm_image = io.imread('Palm.png')
io.imshow(Palm_image)
io.show() 
# Display pending images. Uh-huh. The image NEEDS to be manualy closed for the rest of the program to run. 
# Now I see the numbers for the rows and columns in the terminal. And the flattened image. 

#Dimensions of the original image
rows = Palm_image.shape[0]
cols = Palm_image.shape[1]
print(colored('ORIGINAL PALM IMAGE, NUMBER OF ROWS', 'red'))
print(rows)  
#4032
print(colored('ORIGINAL PALM IMAGE, NUMBER OF COLUMNS', 'red'))
print(cols) 
#3024 

#Flatten the image
Flattened_image = Palm_image.reshape(rows*cols, 3)
print(colored('FLATTENED OR RESHAPED IMAGE', 'red'))
print(Flattened_image)
#[ 71 101 103]
# [ 72 102 104]
 #[ 73 102 106]
 #...
 #[  4  17  10]
 #[  3  16   9]
 #[  0  13   6]]

# The program stops here, Don't worry, it will take some time as listed below. 

print(colored('Implement k-means clustering to form k clusters', 'red'))

#Implement k-means clustering to form k clusters. The size of the compressed image decreases as k decreases. 
# In theory, n = 32 is the smallest number of clusters without loosing without semblance of original image. 
# The recommended number of clusters is 64, with 128 being optimal. 
# However,I had to change the numb er of clusters to 8, as my personal HP Pavilion laptop could not handle procesing of the image with 64 clusters. No CUDA. 
# With 4 clusters, the image took 1.5 min to render. With 8 clusteres, it took over 4 minutes: however, the image was of better quality. Just prepare to wait. 

kmeans = KMeans(n_clusters=8)
kmeans.fit(Flattened_image)

print(colored('Replace each pixel value with its nearby centroid', 'red'))

#Replace each pixel value with its nearby centroid
compressed_image = kmeans.cluster_centers_[kmeans.labels_]
compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

print(compressed_image)
#[[ 96 103  79]
# [ 96 103  79]
# [ 96 103  79]
# ...
# [ 16  30   7]
# [ 16  30   7]
# [ 16  30   7]]

#Reshape the image to original dimensions
compressed_image = compressed_image.reshape(rows, cols, 3)

#Save and display output image
io.imsave('compressed_image_8.png', compressed_image)
io.imshow(compressed_image)
io.show()

#Dimension of the original image
rows = compressed_image.shape[0]
cols = compressed_image.shape[1]
print(colored('COMPRESSED PALM IMAGE, NUMBER OF ROWS', 'red'))
print(rows)  
#4032
print(colored('OCOMPRESSED PALM IMAGE, NUMBER OF COLUMNS', 'red'))
print(cols) 
#3024 
# Interestingly enough, the number of rows and columns stayed the SAME. however, the RGB (red, green, blue) values of the original and
# compressed image are vastly different. Only the first dimension of three stayed similar. The second and third dimensions were reduced. 
# Now we see CENTROIDS of the clusters, nor original values. The color is almost lost. 
# The size is reduced. Original image was 16, 110 KB in PNG format. With 8 clusters, it became 1,776 KB. With 4 clusters, 893 KB. 

print(colored('================================END OF PROGRAM=========================================================', 'yellow'))