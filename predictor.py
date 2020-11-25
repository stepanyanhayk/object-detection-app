import tensorflow as tf
import cv2
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

from utils import plot_detections

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import colab_utils
from object_detection.builders import model_builder

from main import load_image_into_numpy_array
from main import detection_model

duck_class_id = 1
num_classes = 1
category_index = {duck_class_id: {'id': duck_class_id, 'name': 'rubber_ducky'}}


def load_test_images(directory):
# load test images and run inference with new model
    test_image_dir = 'models/research/object_detection/test_images/ducky/test/'
    test_images_np = []
    for i in range(1, 50):
  	    image_path = os.path.join(test_image_dir, 'out' + str(i) + '.jpg')
  	    test_images_np.append(np.expand_dims(load_image_into_numpy_array(image_path), axis=0))
    return test_images_np

# Again, uncomment this decorator if you want to run inference eagerly
@tf.function
def detect(input_tensor, detection_model):
    """
    Run detection on an input image.

    Args:
        input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.

    Returns:
        A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
        and `detection_scores`).
    """
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    return detection_model.postprocess(prediction_dict, shapes)

# Note that the first frame will trigger tracing of the tf.function, which will
# take some time, after which inference should be fast.
def predict(test_image_np, detection_model):
    label_id_offset = 1
    for i in range(len(test_images_np)):
        input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
        detections = detect(input_tensor, detection_model)

        plot_detections(
            test_images_np[i][0],
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.uint32)
            + label_id_offset,
            detections['detection_scores'][0].numpy(),
            category_index, figsize=(15, 20), image_name="gif_frame_" + ('%02d' % i) + ".jpg")

    imageio.plugins.freeimage.download()

    anim_file = 'duckies_test.gif'

    filenames = glob.glob('gif_frame_*.jpg')
    filenames = sorted(filenames)
    last = -1
    images = []
    for filename in filenames:
        image = imageio.imread(filename)
        images.append(image)

    imageio.mimsave(anim_file, images, 'GIF-FI', fps=5)

    display(IPyImage(open(anim_file, 'rb').read()))

test_image_dir = 'models/research/object_detection/test_images/ducky/test/'
test_images_np = load_test_images(test_image_dir)
predict(test_images_np, detection_model)

