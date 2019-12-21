import tensorflow as tf
import mobilenet_v2
import sys

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

  synset_list = [s.strip() for s in open('../imagenet_lsvrc_2015_synsets.txt').readlines()]
  num_synsets_in_ilsvrc = len(synset_list)
  assert num_synsets_in_ilsvrc == 1000

  synset_to_human_list = open('../imagenet_metadata.txt').readlines()
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
    x = endpoints['Predictions'].eval(feed_dict={file_input: '../schoolbus.jpg'})
    tf.summary.FileWriter('./graphs', sess.graph)
  label_map = create_readable_names_for_imagenet_labels()  
  print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())

def architecture():
  with tf.Session() as sess:
    saver.restore(sess,  checkpoint)
    x = endpoints['Predictions'].eval(feed_dict={file_input: '../schoolbus.jpg'})

    print('{:50s} {:s}'.format("Input Layer", str(image.eval(feed_dict={file_input: '../schoolbus.jpg'}).shape)))
    for x in endpoints:
      print('{:50s} {:s}'.format(str(endpoints[x].name), str(endpoints[x].shape)))

argLen = len(sys.argv)

if(argLen == 2 and sys.argv[1] == "arch"):
  architecture()
else:
  default_functionality()