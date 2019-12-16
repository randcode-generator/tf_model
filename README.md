# Models in Tensorflow

### How to run
Follow the directions for the model you wish to run

### VGG16
1. Download VGG16 model from
[http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)
2. Unzip file and put the file 'vgg_16.ckpt' in the same directory. If you choose to place it in another directory, be sure to update the `checkpoints_dir` variable
3. Run `python vgg16.py`
4. (Optional) Run `python vgg16.py arch` to view VGG16 architecture