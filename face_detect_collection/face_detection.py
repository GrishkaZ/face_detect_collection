from abc import abstractmethod

import cv2
import os
import numpy as np
import mediapipe as mp
import time

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

    # @staticmethod
    # def visualize(image,detection_result) -> np.ndarray:
    #
    #     annotated_image = image.copy()
    #     height, width, _ = image.shape
    #
    #     for detection in detection_result.detections:
    #         # Draw bounding_box
    #         bbox = detection.bounding_box
    #         start_point = bbox.origin_x, bbox.origin_y
    #         end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    #         cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
    #
    #         # Draw keypoints
    #         for keypoint in detection.keypoints:
    #             keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
    #                                                            width, height)
    #             color, thickness, radius = (0, 255, 0), 2, 2
    #             cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
    #
    #         # Draw label and score
    #         category = detection.categories[0]
    #         category_name = category.category_name
    #         category_name = '' if category_name is None else category_name
    #         probability = round(category.score, 2)
    #         result_text = category_name + ' (' + str(probability) + ')'
    #         text_location = (MARGIN + bbox.origin_x,
    #                          MARGIN + ROW_SIZE + bbox.origin_y)
    #         cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
    #                     FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    #
    #     return annotated_image


    def __del__(self):
        self.face_detector_model.close()


