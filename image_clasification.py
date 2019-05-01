# https://scikit-learn.org/stable/tutorial/basic/tutorial.html#introduction
# Basic image classification (sklearn)

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


#load dataset
# cat 12500
# dog 12500

cat_directory = "./finalproject_images/PetImages/Cat"
dog_directory = "./finalproject_images/PetImages/Dog"

list_cat_images = [cat_directory + "/"
                   + image for image in os.listdir(cat_directory)]
list_dog_images = [dog_directory + "/"
                   + image for image in os.listdir(dog_directory)]


### split into train and test data set/ Cat = 0 Dog = 1

#train would be first half of cat + second half of dog

train_images_names = list_cat_images[:len(list_cat_images)//2] \
                     + list_dog_images[len(list_dog_images)//2:]

# test would be second half of cat + first half of dog
test_images_names = list_cat_images[len(list_cat_images)//2:] \
                     + list_dog_images[:len(list_dog_images)//2]
# pictures are in diff sizes => need to scale this images into a standard size
# get size of image (stack over flow)
image_size = []

for i in train_images_names:
    img = Image.open(i)
    image_size.append(img.size)

# (mean_height, mean_width) = [sum(y) / len(y) for y in zip(*image_size)]
# print((mean_height, mean_width))
# this number wont work so take it out: Cat 10107, 12086, 1636, 9493, 9683, 666, 11565 Dog : 4761, 15, 26, 11702, 3516
# get the mean
mean = [sum(y) / len(y) for y in zip(*image_size)]
#[403.98647783645384, 360.83037285965753]
STANDARD_SIZE = [400, 350]

# rescal image and then transform image to matrix
data = []
y = []
for i, filename in enumerate(train_images_names):
    img = Image.open(filename)
    img = img.resize(STANDARD_SIZE)
    # img.getdata() Returns the contents of this image as a sequence object containing pixel values.
    # The sequence object is flattened,
    # so that values for line one follow directly after the values of line zero, and so on
    img = list(img.getdata())
    img = np.array(img)
    # now take (m, n) to flattens it to (1, m*n)
    print(img.shape, train_images_names[i])
    if len(img.shape) > 2:
        s = img.shape[0] * img.shape[1]
        img_wide = img.reshape(1, s)
        data.append(img_wide)

data = np.array(data)
data.shape



