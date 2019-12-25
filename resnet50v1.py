import tensorflow as tf
import collections
import os
import sys
from tensorflow.contrib import slim as contrib_slim
from imagenet_label import labels
from resnet_lib import resnet_v1

slim = contrib_slim
height = 224
width = 224

# Put resnet_v1_50.ckpt in the same directory
# Or change the 'checkpoints_dir' to where 'resnet_v1_50.ckpt' is located
checkpoints_dir = os.getcwd()

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

    image  = tf.expand_dims(image, 0)
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      logits, _ = resnet_v1.resnet_v1_50(image, num_classes=1000, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
      os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
      slim.get_model_variables('resnet_v1_50'))

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
  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    _, endpoints = resnet_v1.resnet_v1_50(image, num_classes=1000, is_training=False)

  print('{:79s} {:s}'.format("Input Layer", str(image.shape)))
  for x in endpoints:
    print('{:79s} {:s}'.format(str(endpoints[x].name), str(endpoints[x].shape)))

argLen = len(sys.argv)

if(argLen == 2 and sys.argv[1] == "arch"):
  architecture()
else:
  default_functionality()