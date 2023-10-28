import cv2
import os

root_dir = os.path.join(os.path.dirname(__file__), 'models/face_detection_models/')

def create_face_detector():
    print(root_dir)
    face_detector_model = cv2.dnn.readNetFromCaffe(root_dir + 'face_detector_caffe/deploy.prototxt',
                             root_dir + '/face_detector_caffe/res10_300x300_ssd_iter_140000_fp16.caffemodel')
    return face_detector_model