import os
import cv2
import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()

# Path to frozen detection graph
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map(as in the google tutorial).
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Extract image tensor
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Extract detection boxes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Extract detection scores
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# Extract detection classes
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

while(True):
 
    # Read frame from camera
    ret, frame = video.read()
    # each pixel in the frame has the RGB value
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # expand frame dimensions to have shape: [1, None, None, 3]
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Perform the detection
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    #visualize the results
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.40)

    # All the results have been drawn on the frame.
    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

