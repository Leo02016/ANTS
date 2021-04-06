import os
import sys
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
ROOT_DIR = os.path.abspath("./Mask_RCNN")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import random
import cv2
import imutils
import scipy.io as sio
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
import coco

def preprocess_image(data_path):
    def noisy(image):
        mul = np.mean(image)
        sigma = np.std(image)
        gauss = np.random.normal(mul, sigma, image.shape)
        gauss = gauss.reshape(image.shape)
        masked_index = np.random.randint(0, 1, size=image.shape) < 0.1
        image[masked_index] = image[masked_index] + 0.01 * gauss[masked_index]
        return image

    def create_two_views(data_path, image_name, view_1, view_2):
        image = cv2.imread('{}/images/{}'.format(data_path, image_name))
        dir = image_name.split('/')[0]
        if not os.path.exists('{}/view_1/{}'.format(data_path, dir)):
            os.mkdir('{}/view_1/{}'.format(data_path, dir))
            os.mkdir('{}/view_2/{}'.format(data_path, dir))
        angle = random.randint(-45, 45)
        rotated_img = imutils.rotate(image, angle)
        cv2.imwrite('{}/view_1/{}'.format(data_path, image_name), rotated_img)
        view_1.append('{}/view_1/{}'.format(data_path, image_name))
        noisy_img = noisy(image)
        cv2.imwrite('{}/view_2/{}'.format(data_path, image_name), noisy_img)
        view_2.append('{}/view_2/{}'.format(data_path, image_name))
    view_1 = []
    view_2 = []
    labels = []
    if not os.path.exists('{}/view_1'.format(data_path)):
        os.mkdir('{}/view_1/'.format(data_path))
        os.mkdir('{}/view_2/'.format(data_path))
    with open('{}/lists/train.txt'.format(data_path)) as f:
        for i in range(600):
            a = f.readline()
            labels.append(int(a[:3]))
            create_two_views(data_path, a[:-1], view_1, view_2)
    with open('{}/lists/test.txt'.format(data_path)) as f:
        for i in range(588):
            a = f.readline()
            labels.append(int(a[:3]))
            create_two_views(data_path, a[:-1], view_1, view_2)
    sio.savemat('{}/birds_2_views.mat'.format(data_path), {'view_1': view_1, 'view_2': view_2, 'label': labels})
    print('Finish generating training and test data!')


def load_masked_image(image, mask, name):
    if len(mask) == 0:
        print('Image {} does not have mask:'.format(name))
    else:
        count = 0
        # pick the tok k components
        k = 2
        pixel_num = np.sum(np.sum(mask, axis=0), axis=0)
        indices = pixel_num.argsort()[-k:][::-1]
        for i in indices:
            # check if the masked region is large enough to contain useful features
            a = mask[:, :, i]
            index = np.array([a, a, a]).transpose(1, 2, 0)
            masked_image = np.zeros_like(image)
            masked_image[index] = image[index]
            # indices.append(True)
            skimage.io.imsave('{}_{seg}.jpg'.format(name[:-4], seg=count), masked_image)
            count += 1
        if len(indices) >= 1:
            a = np.sum(mask[:, :, indices], axis=2)
            a[a > 0] = 1
            a = 1 - a
            a = np.array(a, dtype=np.bool)
            index = np.array([a, a, a]).transpose(1, 2, 0)
            masked_image = np.zeros_like(image)
            masked_image[index] = image[index]
            skimage.io.imsave('{}_{seg}.jpg'.format(name[:-4], seg=count), masked_image)
            count += 1
            for i in range(len(indices), k):
                skimage.io.imsave('{}_{seg}.jpg'.format(name[:-4], seg=count), masked_image)
                count += 1


# generate two views
if not os.path.exists('./dataset/birds/birds_2_views.mat'):
    preprocess_image('./dataset/birds')
    print('Finish generating two views')
# matplotlib inline
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

# read the input RGB image
images = sio.loadmat('./dataset/birds/birds_2_views.mat')
count = 0
for image_name in images['view_1']:
    image_name = image_name.strip()
    image = skimage.io.imread(image_name)
    results = model.detect([image], verbose=0)
    load_masked_image(image, results[0]['masks'], image_name)
    count += 1
    print(count)
    # check if there are three segments for all images
    for i in range(3):
        if not os.path.exists('{}_{seg}.jpg'.format(image_name[:-4], seg=i)):
            skimage.io.imsave('{}_{seg}.jpg'.format(image_name[:-4], seg=i), image)
count = 0
for image_name in images['view_2']:
    image_name = image_name.strip()
    image = skimage.io.imread(image_name)
    results = model.detect([image], verbose=0)
    load_masked_image(image, results[0]['masks'], image_name)
    count += 1
    print(count)
    # check if there are three segments for all images
    for i in range(3):
        if not os.path.exists('{}_{seg}.jpg'.format(image_name[:-4], seg=i)):
            skimage.io.imsave('{}_{seg}.jpg'.format(image_name[:-4], seg=i), image)


def load_masked_image(image, mask, name):
    if len(mask) == 0:
        print('Image {} does not have mask:'.format(name))
    else:
        count = 0
        for i in range(mask.shape[2]):
            # check if the masked region is large enough to contain useful features
            a = mask[:, :, i]
            index = np.array([a, a, a]).transpose(1, 2, 0)
            masked_image = np.zeros_like(image)
            masked_image[index] = image[index]
            skimage.io.imsave('{}_{seg}.jpg'.format(name[:-4], seg=count), masked_image)
            count += 1
        a = np.sum(mask, axis=2)
        a[a > 0] = 1
        a = 1 - a
        a = np.array(a, dtype=np.bool)
        index = np.array([a, a, a]).transpose(1, 2, 0)
        masked_image = np.zeros_like(image)
        masked_image[index] = image[index]
        skimage.io.imsave('{}_{seg}.jpg'.format(name[:-4], seg=count), masked_image)
