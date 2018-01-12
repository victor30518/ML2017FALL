import os, sys
import numpy as np
from skimage import io,transform

img_folder_path = sys.argv[1]
target_img = sys.argv[2]

if(img_folder_path[0] == '/'):
	img_folder_path = img_folder_path[1:len(img_folder_path)]

# read image & construct X
for image_no in range(0,415):
	img = io.imread(img_folder_path + '/' + str(image_no) + '.jpg')
	img_flatten = img.flatten()
	if image_no == 0:
		X = img_flatten
	else:
		X = np.column_stack((X, img_flatten))

# SVD
X_mean = np.mean(X, axis=1)
U, s, V = np.linalg.svd(X - X_mean[:,None], full_matrices=False)

# reconstruct
split = target_img.split('.')
image_no = int(split[0])
k = 4

y = np.array(X[:,image_no])
y = y - X_mean

reconstruction = np.zeros(len(y))

for i in range(k):
	weight = np.dot(y,U[:,i])
	reconstruction += weight*U[:,i]

reconstruction = reconstruction + X_mean

reconstruction -= np.min(reconstruction)
reconstruction /= np.max(reconstruction)
reconstruction = (reconstruction * 255).astype(np.uint8)

reconstruction = reconstruction.reshape((600,600,3))

# io.imshow(reconstruction)
io.imsave('reconstruction.jpg',reconstruction)