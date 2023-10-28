import cv2


def create_face_detector():
    face_detector_model = cv2.dnn.readNetFromCaffe('models/face_detection_models/face_detector_caffe/deploy.prototxt',
                             'models/face_detection_models/face_detector_caffe/res10_300x300_ssd_iter_140000_fp16.caffemodel')
    return face_detector_model