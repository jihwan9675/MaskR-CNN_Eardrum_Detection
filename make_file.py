import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
from mrcnn import utils, visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import model as modellib, utils
from modules.CustomConfig import CustomConfig,InferenceConfig

def detect_roi(model, image_path=None):
    image = skimage.io.imread(image_path)
    r = model.detect([image], verbose=1)[0]
    print(r['rois'].shape[0]) # Count Detection Object
    y1, x1, y2, x2 = r['rois'][0] # Rectangle
    crop_image=image[y1:y2,x1:x2]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                        "Ear", r['scores'], 
                        title="Predictions")
    file_name = "{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())
    skimage.io.imsave(file_name, crop_image) # Save Crop_image


def main():
	weights_path="mask_rcnn_eardrum_0043.h5" # Change
	image_path="F1.jpg" # Change
	config = InferenceConfig()
	config.display()
	model = modellib.MaskRCNN(mode="inference", config=config,model_dir="./logs")
	model.load_weights(weights_path, by_name=True)
	detect_roi(model, image_path=image_path)


if __name__ == '__main__':
	try:
		main()
		sys.exit()
	except (EOFError, KeyboardInterrupt) as err:
		print("Keyboard Interupted or Error!!!")