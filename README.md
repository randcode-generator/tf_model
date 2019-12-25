# Models in Tensorflow

### How to run
Follow the directions for the model you wish to run

### Tensorboard
After running one of the following model, run
`tensorboard --logdir=./graphs`

### VGG16
1. Download VGG16 model from
[http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)
2. Unzip file and put the file 'vgg_16.ckpt' in the same directory. If you choose to place it in another directory, be sure to update the `checkpoints_dir` variable
3. Run `python vgg16.py`
4. (Optional) Run `python vgg16.py arch` to view VGG16 architecture

### Resnet
1. Download Resnet model from
[http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)
2. Unzip file and put the file 'resnet_v1_50.ckpt' in the same directory. If you choose to place it in another directory, be sure to update the `checkpoints_dir` variable
3. Run `python resnet50v1.py`
4. (Optional) Run `python resnet50v1.py arch` to view Resnet architecture

### Mobilenet
1. Download Mobilenet model from
[https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz)
2. Unzip file and put the all the files in the same directory. If you choose to place it in another directory, be sure to update the `checkpoints_dir` variable
3. Run `python mobilenet_v2_run.py`
4. (Optional) Run `python mobilenet_v2_run.py arch` to view Mobilenet architecture