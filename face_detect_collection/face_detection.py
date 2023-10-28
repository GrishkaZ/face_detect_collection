from abc import abstractmethod

import cv2
import os
import numpy as np

root_dir = os.path.join(os.path.dirname(__file__), 'models/face_detection_models/')


def draw_bboxes(image, bboxes, inplace=False):
    if not inplace:
        image = image.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if not inplace:
        return image


def crop_faces(self, image, bboxes):
    return [image[y: y + h, x: x + w] for x, y, w, h in bboxes]


class _BaseFaceDetector:

    '''
    return rectangles (x, y, w, h)
    '''
    @abstractmethod
    def get_bboxes(self, image, *args, **kwargs):
        pass


class CaffeFaceDetector(_BaseFaceDetector):

    def __init__(self):
        self.face_detector_model = cv2.dnn.readNetFromCaffe(root_dir + 'face_detector_caffe/deploy.prototxt',
                                                       root_dir + '/face_detector_caffe/res10_300x300_ssd_iter_140000_fp16.caffemodel')

    def get_bboxes(self, image, confidence_threshold):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (123.0, 177.0, 104.0))

        self.face_detector_model.setInput(blob)
        detections = self.face_detector_model.forward()

        detections = np.squeeze(detections)

        conf_boxes = detections[detections[:, 2] > 0.5][:, 2:]
        conf_boxes = conf_boxes[conf_boxes[:,0] > confidence_threshold]
        if len(conf_boxes) == 0:
            return []

        boxes = []
        for conf_box in conf_boxes:
            box = conf_box[1:] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            x_pad = int((endX - startX) * 0.2)
            y_pad_top = int((endY - startY) * 0.2)
            y_pad_bot = int((endY - startY) * 0.05)
            startX = max(startX - x_pad, 0)
            startY = max(startY - y_pad_top, 0)

            endX = min(endX + x_pad, image.shape[1])
            endY = min(endY + y_pad_bot, image.shape[0])

            boxes.append((startX, startY, endX - startX, endY - startY))
        return np.array(boxes)


class HaarCascadeFaceDetector(_BaseFaceDetector):

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(root_dir + '/haarcascade_frontalface_default.xml')

    def get_bboxes(self, image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.face_cascade.detectMultiScale(gray_img, 1.1, 4)


