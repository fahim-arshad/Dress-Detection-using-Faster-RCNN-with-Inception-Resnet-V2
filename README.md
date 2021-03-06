# Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2

### Summary

Step by Step Tutorial of Detecting dresses through the state of the art neural network Faster RCNN with Inception Resnet V2 in real-time.
This repository will not only guide you on how to Detect Clothing Items from an Image, Video or Webcam feed but
will also enable you to train your own Object Detection Classifier. The TensorFlow's Object Detection API is basically made for Linux
but this guide will show you how to use TensorFlow's Object Detection API to train an object detection classifier for multiple objects on Windows 10, 8, or 7..
Ofcourse, this will also work on Linus platforms with minor changes. This repository was done with tensorflow 1.5 but should also with 
older or newer tensorflow versions.

Below are the steps that we are going to follow if we want to train our own Classifier.

1. [Install Anaconda, CUDA and cuDNN](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2#1-install-anaconda-cuda-and-cudnn)
2. [Set up Object Detection Directory](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2#2-set-up-object-detection-directory)
3. [Set up the Virtual Environment](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2#3-set-up-the-virtual-environment)
4. [Label pictures](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2#4-label-pictures)
5. [Generate *.csv* and *tfrecord files*](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2#5-generate-csv-and-tfrecordfiles)
6. [Create Label Map](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2#6-create-label-map)
7. [Training](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2#7-training)
8. [Export the Inference Graph (Object Detection CLassifier)](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2#8-export-the-inference-graph-object-detection-classifier)
9. [Test your Classifier](https://github.com/fahim-arshad/Dress-Detection-using-Faster-RCNN-with-Inception-Resnet-V2#9-test-your-classifier)

### 1. Install Anaconda, CUDA and cuDNN

Download and install [Anaconda](https://www.anaconda.com/products/individual#download-section) as mentioned on the website. The Faster RCNN requires Tensorflow GPU to detect objects in video and live feed but it can detect objects in pictures without Tensorflow GPU. If you want to train your own object detector it is recommended to install a GPU on your system as it reduces the training time by about a factor of 8 (3 hours of training instead of 24). The CPU only version can also be used but it will take longer to train and also to detect object (dresses). We used python 3.5 for our project so in the next step, when creating the anaconda virtual environment we will tell it to use python 3.5. Visit [TensorFlow’s website](https://www.tensorflow.org/install) for further up-to-date installation details, including how to install it on Linux and macOs. [Instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) are also mentioned in the [Object-detection repository](https://github.com/tensorflow/models/tree/master/research/object_detection) that we are going  to need for our task.

### 2. Set up Object Detection Directory

For this step create a folder directly in C drive and name it *“tensorflow1”*. This directory will include all the code form the github repository, the Tensorflow Object Detection API, the images from the dataset, configuration files and everything else needed for the object detection classifier. Download the full [Tensorflow Object Detection repository](https://github.com/tensorflow/models) extract the “model-master” folder directly into *“C:\tensorflow1”* directory you created in the previous step. Rename the *“model-master”* to *“models”*.

**IMP: The Tensorflow Object Detection repository in continually updated by its developers so this table is given that shows which Object detection repository is suited with which Tensorflow version. It is always better to use the latest Tensorflow version with the latest Object Detection repository. If you encounter any errors while using the latest versions of libraries and APIs it may be necessary to use the exact same [object detection repository](https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99) and Tensorflow v1.5. As our pc did not support the AVX instructions released in 2011(AVX instructions were introduced after tensorflow v1.6), we opted to use Tensorflow v1.5 instead of building tensorflow from source or changing the cpu flags.**

| Object Detection Repository | Tensorflow Version |
| --------------------------- | ------------------ |
| [Link](https://github.com/tensorflow/models) | Latest |
| [Link](https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b) | TF 1.9 |
| [Link](https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc) | TF 1.8 |
| [Link](https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f) | TF 1.7 |
| [Link](https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99) | TF 1.5 |
| [Link](https://github.com/tensorflow/models/tree/r1.13.0) | TF 1.13 |
| [Link](https://github.com/tensorflow/models/tree/r1.12.0) | TF 1.12 |
| [Link](https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df) | TF 1.10 |

Now download the Faster RCNN Inception V2 COCO model from the TensorFlow’s [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). You can use you own model if you are planning to run this project on a low computational power device such as SDD-MobileNet Model or Rasberry Pi but they will give you low accuracy on the expense of faster detection and low required computational power. Extract the faster_rcnn_inception_v2_coco_2018_01_28 folder to the
*C:\tensorflow1\models\research\object_detection* folder.

Now download the full repository from this page(go to the top and clone download the repository) and extract it into *C:\tensorflow1\models\research\object_detection* folder. This will now provide you with the specific directory structure that is required. This repository contains the images, annotation data, tfrecord files and the .csv files required for this project.

If you want to train your own object detection classifier with your own dataset you need to delete the following files:
- files in *\object_detection\training*
- files in *\object_detection\inference_graph*
- files in *\object_detection\images* "train/test_labels.csv" **not the test/train folders**
- files in *\object_detection\images\train* and *\object_detection\images\test* **not the test/train folders**

### 3. Set up the Virtual Environment

Open the Anaconda Command Prompt with administrative privileges and create a new virtual Tensorflow environment and activate it with the following commands.

```conda create –n tensorflow1 pip python-3.5```
```activate tensorlow1```
```python –m pip install –upgrade pip```

Install tensorflow-gpu with:

`pip install --ignore-installed --upgrade tensorflow-gpu`

**IMP: If you want to use the Tensorflow CPU version then just use “tensorflow” instead of “tensorflow-gpu” in the previous command.**

Other libraries and packages required to run this project are:

-	Opencv ```pip install opencv-python```
-	Protobuf ```conda install -c anaconda protobuf```
-	Matplotlib ```pip install matplotlib```
-	Lxml ```pip install lxml```
-	Contextlib2 ```pip install contextlib2```
-	Xython ```pip install Cython```
-	Pandas ```pip install pandas```
-	Pillow ```pip install pillow```

Opencv and pandas are not required by Tensorflow but are used in this project to generate tfrecord files and enable the of image, video and webcam scripts that are used in this project and are mentioned in the User Manual.

Now we need to configure a PYTHONPATH variable and it will be configured every time the “tensorfow1” environment is exited. This variable must point to *“\models”*, *“\models\research”*, and *“\models\research\slim”* directories:

```set PYTHONPATH=C:\tensorflow1\models; C:\tensorflow1\models\research; C:\tensorflow1\models\research\slim```

**You can test the PYTHONPATH variable by executing "echo %PYTHONPATH%" command to check whether it has been set or not**

Now we need to compile the protobuf files. Because the command given on the TensorFlow’s Object Detection API installation page does not work on windows so we will have to call out each .proto file individually. First we will need to change the directory in in Anaconda Command Prompt:

```cd C:\tensorflow1\models\research```

Then:

```protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto```

This command will create a xxxx_pb2.py file from every xxxx.proto file in the “\object_detection\protos” folder.
Now execute the following commands from “C:\tensorflow1\models\research” folder.

**If you ever get an error saying "Import Error: Cannot import xxxx_pb2, you will need to update the protoc command and include *.\object_detection\protos\xxxx.proto* in the protoc command" **

```python setup.py build```

```python setup.py install```

The environment needed for testing the Dress Detection Classifier is all set up now.

### 4. Label pictures

The size of the pictures should not be greater than 200kb and the resolution should not be greater than 720x1280(or it will take much longer to train the model on the dataset). You can use your mobile camera to take the pictures and if your size or resoultion is too much there are many scripts available online that help can reduce the size of your pictures.

Make sure to capture the images with different backgrounds, different lightning conditions, multiple desired and random objects in the picture so you can make a good dataset that will train a robust model. We have collected 1500 images of 6 desired objects atm i.e. skirts, t-shirts, jeans, trousers, sleeveless and shorts. you can collect your own dataset or use the dataset that we have provided in the repository.

Move 80% of the image into the *object_detection/images/train* directory and 20% of the images into the *object_detection/images/test* directory.

To Label the images we have used [LabelImg](https://github.com/tzutalin/labelImg). It is available on github and instructions to install and use it are mentioned clearly on the github link. Each image will have a corresponding .xml file after you have labelled the images by LabelImg in the same directory that the images are present in.

### 5. Generate *.csv* and *tfrecord files*

TFRecord files serve as an input to the tensorflow training model. We will first convert the .xml files into csv files and then the csv files into the TFRecord files. The scripts used for these conversions are from [Dat Tran’s Raccoon Detector dataset](https://github.com/datitran/raccoon_dataset) but we have a different folder directory architecture so the sciprts are changed accordingly.

From the \object_detection folder issue the following command:-

```python xml_to_csv.py```

This will create a train_label.csv and test_label.csv in the \object_detection\images folder.

Now open edit the generate_tfrecord.py file from the \object_detection folder and at line 31 replace the label map with name of your own objects

```
def class_text_to_int(row_label):
    if row_label == 'skirt':
        return 1
    elif row_label == 'trouser':
        return 2
    elif row_label == 't-shirt':
        return 3
    elif row_label == 'jeans':
        return 4
    elif row_label == 'sleeveless':
        return 5
    elif row_label == 'shorts':
        return 6
    else:
        0
```

Save the generate_tfrecord.py and execute the following two command from the \object_detection folder.

```python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record```

```python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record```

Now you will have a train.record and test.record file in your \object_detection folder

### 6. Create Label Map

Use text editor to create a file and save it labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder **.pbtx should be the format and not .txt**. Edit the labelmap.pbtx with the text editor and make a label map with the following format:-

``` 
item {
  id: 1
  name: 'skirt'
}

item {
  id: 2
  name: 'trouser'
}

item {
  id: 3
  name: 't-shirt'
}

item {
  id: 4
  name: 'jeans'
}

item {
  id: 5
  name: 'sleeveless' 
}
item {
  id: 6
  name: 'shorts'
}  
```
The id numbers should be in the order as in the generate_tfrecord.py file.

### 7. Training

In this step we will configure the training pipeline, define which model is to be used and what will be the parameters.

From the *C:\tensorflow1\models\research\object_detection\samples\configs* folder copy and paste the faster_rcnn_inception_v2_pets.config file into the *C:\tensorflow1\models\research\object_detection\training* directory. Open and edit the faster_rcnn_inception_v2_pets.config file with a text editor. 

**IMP: make sure to use forward slashes and double quotation markes when giving the directory paths or you will encounter errors**

- **Line 9:** set the num_classes variable equal to the number of objects you want to  detect(in our case it was 6).

- **Line 106:** set the fine_tune_checkpoint:-

```"C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"```

- **Line 123:** set the input_path:-

```"C:/tensorflow1/models/research/object_detection/train.record"```

- **Line 125:** set the label_map_path:-

```"C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"```

- **Line 130:** set the num_examples variable equal to the number of images in your *\object_detection\images\train* folder

- **Line 135:** set the eval input_path:-

```"C:/tensorflow1/models/research/object_detection/test.record"```

- **Line 135:** set the eval label_map_path:-

```"C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"```

Your pipeline has now been configured!

## Start the Training

Now it is time to start the training of your model. From the *\object_detection* directory run the following command:-

```python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config```

*We have used tensorflow v1.5, but from tensorflow v1.9 train.py file has been deprecated and model_main.py is being used instead. Although the train.py file is available in the \object_detection\legacy folder. You can move the train.py into the \object_detection folder and run it simply or you an use either of the following files to train your model. Just replace the 'train.py' with 'model_main.py' in the above command to use the model_main.py file for training.*

The training procedure will be initialized in about half a minute. Each step would show the loss and time per step. In our opinion the loss should be as low as 0.05 so that the model is trained enough. The model will take time according the capacity of your CPU and/or GPU.

Checkpoints are saved after every few minutes. You can stop the training by issuing ctrl+c from the command prompt and resume it later as you want(It will automatically start from the last saved checkpoint)

### 8. Export the Inference Graph (Object Detection CLassifier)

After the training, we now have to export the frozen inference graph i.e. the object detection classifier.

```python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph```

Run the following command from *\object_detection* folder and instead of XXXX in the command enter the highest numbered .ckpt file that is present in the *\object_detection\training* folder. Now your frozen inference graph is exported and ready for testing.

### 9. Test your Classifier

If you want to test the Dress Detection System you can use “inference graph.pb” file in the *“object_detection”* folder but if you want to train the Dress Detection System again or you want to train your own Object Detection Classifier the relevant instructions are mentioned on the github tutuorial. For now the frozen_inference_graph.pb file in the *“\Object_detection\inference_graph”* folder contains the object (Dress) Detection Classifier.

The repository contains python scripts required to test it out on image, video or webcam live feed. You need to change NUM_CLASSES variable in the scripts equal to the number of classes you want to detect. In our Dress Detection System we have a total of 6 classes so we are using “NUM_CLASSES = 6”. Move and rename a picture or video in the *\object_detection* folder change IMAGE/VIDEO_NAME variable in the *Obejct_detection_image/video.py* according to the image/video name in the folder. You can also plug in a USB webcam by using the Object_detection_webcam.py file.

Now execute the following command in the Anaconda Command Prompt:

```Idle```

Note that the tensorflow1 environment should be activated as mentioned above and the PYTHONPATH variable is also set. Now IDLE window will be opened and you can run any one of the following scripts from there.

A window will be opened after about 20 seconds and the object (Dress) Detector will be initialized. In the window the desired objects will be shown detected according to the accuracy of the system.

