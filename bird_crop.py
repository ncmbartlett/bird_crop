# https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

import numpy as np
import argparse
import cv2
import os
import glob

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
		   "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
		   "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CROP_DIR = 'cropped'

# Create save directory if needed
if not os.path.isdir(CROP_DIR):
	os.mkdir(CROP_DIR)

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="Path to directory of input images")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="Min. probability to filter weak detections")
args = vars(ap.parse_args())

# Load model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

for img_file in sorted(glob.glob(args["directory"] + '/*')):
	# Load input image, scale down by 2x, construct input blob for the image by resizing to a fixed 300x300 pixels, 
	# normalize it (note: normalization is done via the authors of the MobileNet SSD implementation)
	image = cv2.imread(img_file)
	image = cv2.pyrDown(image)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
	name = img_file.split('/')[1].split('.')[0]

	# Feed blob through network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()

	# Filter detections to only return birds
	detections = np.squeeze(detections, axis=(0, 1))
	detections = detections[detections[:, 1] == 3.]  # Note that "bird" is the 3rd class in CLASSES.

	# Main logic
	if detections.shape[0] == 0:  # If no detections, report it
		report = "{}: No bird detected".format(name)
		print(report)
		with open('failures.txt', 'a+') as f:
			f.write(report + '\n')
	else:
		# Only return the highest confidence detection
		confidence = detections[0, 2]

		# Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
		if confidence > args["confidence"]:
			report = "{}: Bird detected with {}% confidence".format(name, round(confidence * 100, 2))
			print(report)
			with open('successes.txt', 'a+') as f:
				f.write(report + '\n')
			box = detections[0, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype(np.int32)

			# Enlarge the bounding box a bit, and crop the object
			startX = max(0, startX - 20)
			endX = min(endX + 20, w)
			startY = max(0, startY - 20)
			endY = min(endY + 20, h)
			cropped = image[startY:endY, startX:endX]

			# Save cropped image
			save_name = CROP_DIR + '/' + img_file.split('/')[1] 
			save_name = save_name.split('.')[0] +'.png'
			cv2.imwrite(save_name, cropped)
		else:
			report = "{}: Confidence below threshold at {}%".format(name, round(confidence * 100, 2))
			print(report)
			with open('failures.txt', 'a+') as f:
				f.write(report + '\n')

