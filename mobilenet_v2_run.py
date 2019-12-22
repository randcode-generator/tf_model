import tensorflow as tf
import sys
import os

sys.path.append(os.getcwd() + '/mobilenet_lib')

from mobilenet_lib import mobilenet_v2
from imagenet_label import labels

# For simplicity we just decode jpeg inside tensorflow.
# But one can provide any input obviously.
file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

# Note: arg_scope is optional for inference.
logits, endpoints = mobilenet_v2.mobilenet(images)

# Restore using exponential moving average since it produces (1.5-2%) higher 
# accuracy
ema = tf.train.ExponentialMovingAverage(0.999)
vars = ema.variables_to_restore()

saver = tf.train.Saver(vars)

checkpoint = "./mobilenet_v2_1.0_224.ckpt"

def default_functionality():
  with tf.Session() as sess:
    saver.restore(sess,  checkpoint)
    x = endpoints['Predictions'].eval(feed_dict={file_input: './schoolbus.jpg'})
    tf.summary.FileWriter('./graphs', sess.graph)
  label_map = labels.create_readable_names_for_imagenet_labels()  
  print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())

def architecture():
  with tf.Session() as sess:
    saver.restore(sess,  checkpoint)
    x = endpoints['Predictions'].eval(feed_dict={file_input: './schoolbus.jpg'})

    print('{:50s} {:s}'.format("Input Layer", str(image.eval(feed_dict={file_input: './schoolbus.jpg'}).shape)))
    for x in endpoints:
      print('{:50s} {:s}'.format(str(endpoints[x].name), str(endpoints[x].shape)))

argLen = len(sys.argv)

if(argLen == 2 and sys.argv[1] == "arch"):
  architecture()
else:
  default_functionality()