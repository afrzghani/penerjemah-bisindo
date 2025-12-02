import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os

def pickle_images_labels():
	images_labels = []
	images = glob("gestures/*/*.jpg")
	images.sort()
	
	for image in images:
		label = image[image.find(os.sep)+1: image.rfind(os.sep)]
		img = cv2.imread(image, 0)
		if img is not None:
			images_labels.append((np.array(img, dtype=np.uint8), int(label)))
		else:
			print(f"Gagal load {image}")
	
	return images_labels

images_labels = pickle_images_labels()
images_labels = shuffle(shuffle(shuffle(shuffle(images_labels)), random_state=42))
images, labels = zip(*images_labels)
total_images = len(images)

train_split = int(0.7 * total_images)
val_split = int(0.9 * total_images)

train_images = images[:train_split]
train_labels = labels[:train_split]

val_images = images[train_split:val_split]
val_labels = labels[train_split:val_split]

test_images = images[val_split:]
test_labels = labels[val_split:]

with open("train_images", "wb") as f:
	pickle.dump(train_images, f)
print("train_images tersimpan")
with open("train_labels", "wb") as f:
	pickle.dump(train_labels, f)
print("train_labels tersimpan")
del train_images
del train_labels

with open("val_images", "wb") as f:
	pickle.dump(val_images, f)
print("val_images tersimpan")
with open("val_labels", "wb") as f:
	pickle.dump(val_labels, f)
print("val_labels tersimpan")
del val_images
del val_labels

with open("test_images", "wb") as f:
	pickle.dump(test_images, f)
print("test_images tersimpan")
with open("test_labels", "wb") as f:
	pickle.dump(test_labels, f)
print("test_labels tersimpan")
del test_images
del test_labels