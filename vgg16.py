# Based on the following sample
# https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb

import tensorflow as tf
import sys
import os
from tensorflow.contrib import slim as contrib_slim

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

def create_readable_names_for_imagenet_labels():
  """Create a dict mapping label id to human readable string.
  Returns:
      labels_to_names: dictionary where keys are integers from to 1000
      and values are human-readable names.
  We retrieve a synset file, which contains a list of valid synset labels used
  by ILSVRC competition. There is one synset one per line, eg.
          #   n01440764
          #   n01443537
  We also retrieve a synset_to_human_file, which contains a mapping from synsets
  to human-readable names for every synset in Imagenet. These are stored in a
  tsv format, as follows:
          #   n02119247    black fox
          #   n02119359    silver fox
  We assign each synset (in alphabetical order) an integer, starting from 1
  (since 0 is reserved for the background class).
  Code is based on
  https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py#L463
  """

  synset_list = [s.strip() for s in open('./imagenet_lsvrc_2015_synsets.txt').readlines()]
  num_synsets_in_ilsvrc = len(synset_list)
  assert num_synsets_in_ilsvrc == 1000

  synset_to_human_list = open('./imagenet_metadata.txt').readlines()
  num_synsets_in_all_imagenet = len(synset_to_human_list)
  assert num_synsets_in_all_imagenet == 21842

  synset_to_human = {}
  for s in synset_to_human_list:
    parts = s.strip().split('\t')
    assert len(parts) == 2
    synset = parts[0]
    human = parts[1]
    synset_to_human[synset] = human

  label_index = 1
  labels_to_names = {0: 'background'}
  for synset in synset_list:
    name = synset_to_human[synset]
    labels_to_names[label_index] = name
    label_index += 1

  return labels_to_names

def vgg_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

# Source
# https://github.com/tensorflow/models/blob/a19f2f8b1a4f855b89300333dc5d8ccad891e802/research/slim/nets/vgg.py#L148
def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           reuse=None,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points

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
    with slim.arg_scope(vgg_arg_scope()):
      logits, _ = vgg_16(image, num_classes=1000, is_training=False)
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
      names = create_readable_names_for_imagenet_labels()
      for i in range(5):
        index = sorted_inds[i]
        # Shift the index of a class name by one. 
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index+1]))

def architecture():
  image = tf.random.uniform([1, height, width, 3])
  with slim.arg_scope(vgg_arg_scope()):
    _, endpoints = vgg_16(image, num_classes=1000, is_training=False)

  print('{:30s} {:s}'.format("Input Layer", str(image.shape)))
  for x in endpoints:
    print('{:30s} {:s}'.format(str(endpoints[x].name), str(endpoints[x].shape)))

argLen = len(sys.argv)

if(argLen == 2 and sys.argv[1] == "arch"):
  architecture()
else:
  default_functionality()