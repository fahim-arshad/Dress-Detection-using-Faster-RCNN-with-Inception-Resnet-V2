# Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2

### Summary

Step by Step Tutorial of Detecting dresses through the state of the art neural network Faster RCNN with Inception Resnet V2 in real-time.
This repository will not only guide you on how to Detect Clothing Items from an Image, Video or Webcam feed but
will also enable you to train your own Object Detection Classifier. The TensorFlow's Object Detection API is basically made for Linux
but this guide will show you how to use TensorFlow's Object Detection API to train an object detection classifier for multiple objects on Windows 10, 8, or 7..
Ofcourse, this will also work on Linus platforms with minor changes. This repository was done with tensorflow 2.1 but should also with 
older or newer tensorflow versions.

Below are the steps that we are going to follow if we want to train our own Classifier.

1. [Install Anaconda, CUDA and cuDNN](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2/new/master?readme=1#1-install-anaconda-cuda-and-cudnn)
2. [Set up Object Detection Directory](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2/new/master?readme=1#2-set-up-object-detection-directory)
3. [Set up the Virtual Environment](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2/new/master?readme=1#3-set-up-the-virtual-environment)
4. [Label pictures](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2/new/master?readme=1#4-label-pictures)
5. [Generate *.csv* and *tfrecord files*](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2/new/master?readme=1#5-generate-csv-and-tfrecord-files)
6. [Create Label Map](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2/new/master?readme=1#6-create-label-map)
7. [Training](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2/new/master?readme=1#7-training)
8. [Export the Inference Graph (Object Detection CLassifier)](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2/new/master?readme=1#8-export-the-inference-graph-object-detection-classifier)
9. [Test your Classifier](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2/new/master?readme=1#9-test-your-classifier)

### 1. Install Anaconda, CUDA and cuDNN

Download and install [Anaconda](https://www.anaconda.com/products/individual#download-section) as mentioned on the website. The Faster RCNN requires Tensorflow GPU to detect objects in video and live feed but it can detect objects in pictures without Tensorflow GPU. If you want to train your own object detector it is recommended to install a GPU on your system as it reduces the training time by about a factor of 8 (3 hours of training instead of 24). The CPU only version can also be used but it will take longer to train and also to detect object (dresses). We used python 3.5 for our project so in the next step, when creating the anaconda virtual environment we will tell it to use python 3.5. Visit [TensorFlow’s website](https://www.tensorflow.org/install) for further up-to-date installation details, including how to install it on Linux and macOs. [Instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) are also mentioned in the [Object-detection repository](https://github.com/tensorflow/models/tree/master/research/object_detection) that we are going  to need for our task.

### 2. Set up Object Detection Directory

For this step create a folder directly in C drive and name it *“tensorflow1”*. This directory will include all the code form the github repository, the Tensorflow Object Detection API, the images from the dataset, configuration files and everything else needed for the object detection classifier. Download the full [Tensorflow Object Detection repository](https://github.com/tensorflow/models) extract the “model-master” folder directly into *“C:\tensorflow1”* directory you created in the previous step. Rename the *“model-master”* to *“models”*.

**IMP: The Tensorflow Object Detection repository in continually updated by its developers so this table is given that shows which Object detection repository is suited with which Tensorflow version. It is always better to use the latest Tensorflow version with the latest Object Detection repository.**

| Object Detection Repository | Tensorflow Version |
| --------------------------- | ------------------ |
| [Link](https://github.com/tensorflow/models) | Latest |
| [Link](https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b) | TF 1.9 |
| [Link](https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc) | TF 1.8 |
| [Link](https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f) | TF 1.7 |
| [Link](https://github.com/tensorflow/models/tree/r1.13.0) | TF 1.13 |
| [Link](https://github.com/tensorflow/models/tree/r1.12.0) | TF 1.12 |
| [Link](https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df) | TF 1.10 |

Now download the Faster RCNN Inception V2 COCO model from the TensorFlow’s [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). You can use you own model if you are planning to run this project on a low computational power device such as SDD-MobileNet Model or Rasberry Pi but they will give you low accuracy on the expense of faster detection and low required computational power. Extract the faster_rcnn_inception_v2_coco_2018_01_28 folder to the
*C:\tensorflow1\models\research\object_detection* folder.

Now download the full repository from this page(go to the top and clone download the repository) and extract it into *C:\tensorflow1\models\research\object_detection* folder. This will now provide you with the specific directory structure that is required. This repository contains the images, annotation data, tfrecord files and the .csv files required for this project.

### 3. Set up the Virtual Environment

### 4. Label pictures

### 5. Generate *.csv* and *tfrecord files*

### 6. Create Label Map

### 7. Training

### 8. Export the Inference Graph (Object Detection CLassifier)

### 9. Test your Classifier
