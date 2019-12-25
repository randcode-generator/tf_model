# Based on the following sample
# https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb

import tensorflow as tf
import sys
import os
from tensorflow.contrib import slim as contrib_slim
from imagenet_label import labels
from vgg_lib import vgg

height = 224
width = 224

slim = contrib_slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

# Put vgg_16.ckpt in the same directory
# Or change the 'checkpoints_dir' to where 'vgg_16.ckpt' is located
checkpoints_dir = os.getcwd()

def _mean_image_subtraction(image, means):
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def default_functionality():
  with tf.Graph().as_default():
    file = tf.read_file('schoolbus.jpg')

    image = tf.image.decode_jpeg(file, channels=3)
    image = tf.dtypes.cast(image, tf.float32)

    # resize start
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image)
    image.set_shape([None, None, 3])
    # resize end

    image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    image  = tf.expand_dims(image, 0)
    with slim.arg_scope(vgg.vgg_arg_scope()):
      logits, _ = vgg.vgg_16(image, num_classes=1000, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
      os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
      slim.get_model_variables('vgg_16'))

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
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index+1]))

def architecture():
  image = tf.random.uniform([1, height, width, 3])
  with slim.arg_scope(vgg.vgg_arg_scope()):
    _, endpoints = vgg.vgg_16(image, num_classes=1000, is_training=False)

  print('{:30s} {:s}'.format("Input Layer", str(image.shape)))
  for x in endpoints:
    print('{:30s} {:s}'.format(str(endpoints[x].name), str(endpoints[x].shape)))

argLen = len(sys.argv)

if(argLen == 2 and sys.argv[1] == "arch"):
  architecture()
else:
  default_functionality()