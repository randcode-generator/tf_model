import tensorflow as tf
import sys
import os
from tensorflow.contrib import slim as contrib_slim
from imagenet_label import labels
from collections import OrderedDict 
from tensorflow.python.framework import ops
from inception_lib import inception_v1
from inception_lib import inception_utils

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

height = 224
width = 224

slim = contrib_slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# Put inception_v1.ckpt in the same directory
# Or change the 'checkpoints_dir' to where 'inception_v1.ckpt' is located
checkpoints_dir = os.getcwd()

def default_functionality():
  with tf.Graph().as_default():
    file = tf.read_file('schoolbus.jpg')

    image = tf.image.decode_jpeg(file, channels=3)
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # resize start
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image)
    # resize end

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    image  = tf.expand_dims(image, 0)
    with slim.arg_scope(inception_utils.inception_arg_scope()):
      logits, _ = inception_v1.inception_v1(image, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
      os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
      slim.get_model_variables('InceptionV1'))

    with tf.Session() as sess:
      init_fn(sess)
      tf.summary.FileWriter('./graphs', sess.graph)
      probabilities = sess.run(probabilities)
      probabilities = probabilities[0, 0:]
      sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
      names = labels.create_readable_names_for_imagenet_labels()
      
      for i in range(5):
        index = sorted_inds[i]
        # Shift the index of a class name by one. 
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))

def architecture():
  image = tf.random.uniform([1, height, width, 3])
  with slim.arg_scope(inception_utils.inception_arg_scope()):
    _, _ = inception_v1.inception_v1(image, num_classes=1000, is_training=False)

  endpoints = slim.utils.convert_collection_to_dict("InceptionV1_endpoints")
  print('{:67s} {:s}'.format("Input Layer", str(image.shape)))
  for x in endpoints:
    print('{:67s} {:s}'.format(str(endpoints[x].name), str(endpoints[x].shape)))

argLen = len(sys.argv)

if(argLen == 2 and sys.argv[1] == "arch"):
  architecture()
else:
  default_functionality()