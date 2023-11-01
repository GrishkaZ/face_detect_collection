from abc import abstractmethod
from collections import defaultdict
from typing import List

import cv2
import os
import itertools
import numpy as np
import mediapipe as mp
import time

import torch
from kornia import contrib as kc

root_dir = os.path.join(os.path.dirname(__file__), 'models/face_detection_models/')


def draw_bboxes(image, bboxes, inplace=False):
    if not inplace:
        image = image.copy()
    for (x, y, w, h) in bboxes:
        try:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        except:
            print('DRAW RECTANGLE ERROR')
            print((x, y), (x + w, y + h))
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


class MediapipeSolutionFaceDetector(_BaseFaceDetector):
    Keypoint_types = mp.solutions.face_detection.FaceKeyPoint

    SHORT_RANGE = 0
    FULL_RANGE = 1

    def __init__(self, min_detection_con=0.5, model_selection=SHORT_RANGE):
        mp.solutions.face_detection.FaceDetection()
        self.face_detection_model = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_detection_con,
                                                                              model_selection=model_selection)

    def get_bboxes(self, image, *args, **kwargs):

        self.results = self.face_detection_model.process(image)
        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # if detection.score[0] > self.minDetectionCon:
                conf_box = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                bbox = [int(conf_box.xmin * w), int(conf_box.ymin * h),
                        int(conf_box.width * w), int(conf_box.height * h)]
                bboxes.append(bbox)

        return bboxes


    def get_keypoints(self, image, keypoints_types = None):
        self.results = self.face_detection_model.process(image)
        if self.results.detections:
            keypoints_list = []
            for id, detection in enumerate(self.results.detections):

                if keypoints_types:
                    keypoints = [mp.solutions.face_detection.get_key_point(detection,type) for type in keypoints_types]
                else:
                    keypoints = detection.location_data.relative_keypoints

                for keypoint in keypoints:
                    if keypoint.x and keypoint.y:
                        x = int(keypoint.x * image.shape[1])
                        y = int(keypoint.y * image.shape[0])
                        keypoints_list.append((keypoint.keypoint_label, (x, y)))

            return keypoints_list

    def __del__(self):
        self.face_detection_model.close()



class MediapipeTaskFaceDetector(_BaseFaceDetector):
    VisionRunningMode = mp.tasks.vision.RunningMode

    def __init__(self, vision_running_mode,
                 min_detection_confidence = 0.5, min_suppression_threshold = 0.3,
                 result_callback = None):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        FaceDetectorResult = mp.tasks.vision.FaceDetectorResult

        # Create a face detector instance with the live stream mode:
        def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
            print('face detector result: {}'.format(result))

        self.vision_running_mode = vision_running_mode

        if self.vision_running_mode == self.VisionRunningMode.LIVE_STREAM and result_callback is None:
            print('result_callback hasn\'t been selected. Set default print_result callback.')
            result_callback = print_result

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(root_dir, 'blaze_face_short_range.tflite')),
            running_mode= self.vision_running_mode,
            min_detection_confidence=min_detection_confidence,
            min_suppression_threshold=min_suppression_threshold,
            result_callback=result_callback
        )

        self.face_detector_model = FaceDetector.create_from_options(options)


    def get_bboxes(self, image, *args, **kwargs):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        if self.vision_running_mode == self.VisionRunningMode.IMAGE:
            face_detector_result = self.face_detector_model.detect(mp_image)
        elif self.vision_running_mode == self.VisionRunningMode.VIDEO:
            face_detector_result = self.face_detector_model.detect_for_video(mp_image, round(time.time() * 1000))
        elif self.vision_running_mode == self.VisionRunningMode.LIVE_STREAM:
            face_detector_result = self.face_detector_model.detect_async(mp_image, round(time.time() * 1000))
        else:
            raise ValueError('Incorrect vision_running_mode')

        if face_detector_result:
            bboxes = []
            for detection in face_detector_result.detections:
                bbox = detection.bounding_box
                bboxes.append((bbox.origin_x, bbox.origin_y, bbox.width, bbox.height))
            return bboxes


    def __del__(self):
        self.face_detector_model.close()


class KorniaFaceDetector(_BaseFaceDetector):

    FaceKeypoint = kc.FaceKeypoint

    def __init__(self, confidence_threshold = 0.5, nms_threshold=0.3, device = 'cpu'):
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                print("CUDA in not available, run on CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.face_detection_model = kc.FaceDetector(confidence_threshold=confidence_threshold,
                                                    nms_threshold=nms_threshold).to(self.device)

    def detect(self, images: List[np.ndarray], batch_size = 1) -> List[List[kc.FaceDetectorResult]]:
        img_shapes = np.array([im.shape for im in images]).T
        if len(np.unique(img_shapes[0])) > 1 or len(np.unique(img_shapes[1])) > 1 or len(np.unique(img_shapes[2])) > 1:
            raise ValueError('All images shapes must be the same.')
        images = np.array(images)
        tensor_images = torch.FloatTensor(images).permute(0,3,1,2)
        with torch.no_grad():
            images_detections_list = []
            for i in range(0, len(images), batch_size):
                batch = tensor_images[i : i + batch_size].to(self.device)
                detections_list = self.face_detection_model(batch)
                for detections in detections_list:
                    images_detections_list.append([kc.FaceDetectorResult(detection).to(torch.device('cpu'))
                                                 for detection in detections])
            return images_detections_list



    def get_bboxes(self, image, *args, **kwargs):
        detections = self.detect([image])[0]
        bboxes = []
        for detection in detections:
            x, y = detection.top_left.numpy().astype(int)
            w, h = int(detection.width.item()), int(detection.height.item())
            bboxes.append((x, y, w, h))
        return bboxes


    def get_keypoints(self, image, keypoints_types=None) -> dict:
        if keypoints_types is None:
            keypoints_types = [self.FaceKeypoint.NOSE,
                               self.FaceKeypoint.EYE_LEFT,
                               self.FaceKeypoint.EYE_RIGHT,
                               self.FaceKeypoint.NOSE,
                               self.FaceKeypoint.MOUTH_LEFT,
                               self.FaceKeypoint.MOUTH_RIGHT]

        detections = self.detect([image])[0]
        keypoints_dict = defaultdict(list)
        for detection in detections:
            for key_point in keypoints_types:
                keypoints_dict[key_point].append(detection.get_keypoint(key_point).cpu().numpy().astype(int))
        return keypoints_dict


