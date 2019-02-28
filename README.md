# bird_crop
Simple script for cropping birds from images using OpenCV, re-purposed from https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/ to deal only with birds and to do cropping instead of drawing bounding boxes.

# Usage
**Single image**; type into terminal: python bird_crop.py --i IMAGE_PATH

**Multiple images**; type into terminal: for i in IMAGE_DIR/\*; do python bird_crop.py --i $i; done

# Requirements
opencv 3.4+, numpy
