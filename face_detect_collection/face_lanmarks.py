import numpy as np
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class MediapipeLandmarksDrawer:

    #todo set drawing params
    def __init__(self, color=(0, 0, 255), thickness=1, circle_radius=1):
        self.drawing_spec = mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def draw(self, image, face_landmarks, inplace=False):

        annotated_image = image if  inplace else image.copy()

        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

        if not inplace:
            return annotated_image


#todo return modes: images, landmarks, show image/video
class MediapipeFaceMeshDetector:
    """
        drawer: draw landmarks if not None
    """
    def __init__(self, drawer: MediapipeLandmarksDrawer = None):
        self.drawer = drawer

    '''
      For static images.
      params: see mediapipe FaceMesh docks
    '''
    def detect_on_images(self, images, return_landmarks = False,  max_num_faces = 1, refine_landmarks = False, min_detection_confidence=0.5):

        if not self.drawer and not return_landmarks:
            raise ValueError('if return_landmarks == False, the drawer must be')

        if return_landmarks:
            landmarks_list = []
        if self.drawer:
            annotated_images = []

        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence) as face_mesh:

            for idx, image in enumerate(images):
                results = face_mesh.process(image)
                if return_landmarks:
                    landmarks_list.append(results.multi_face_landmarks)
                if not results.multi_face_landmarks:
                    continue

                if self.drawer:
                    for face_landmarks in results.multi_face_landmarks:
                        annotated_images.append(self.drawer.draw(image, face_landmarks, False))

        if self.drawer and return_landmarks:
            return annotated_images, landmarks_list
        if return_landmarks:
            return landmarks_list
        return annotated_images

    '''
      For video input.
      other params: see mediapipe FaceMesh docks
      :return (frame number, landmarks)
    '''
    def detect_on_video(self, cap: cv2.VideoCapture, return_landmarks = False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        if not self.drawer and not return_landmarks:
            raise ValueError('if return_landmarks == False, the drawer must be')

        if return_landmarks:
            landmarks_list = []
        with mp_face_mesh.FaceMesh(
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence) as face_mesh:

            print('PRESS ESCAPE TO STOP')

            frame_counter = 0
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                if results.multi_face_landmarks:
                    if return_landmarks:
                        landmarks_list.append((frame_counter, results.multi_face_landmarks))
                    if self.drawer:
                        image.flags.writeable = True
                        for face_landmarks in results.multi_face_landmarks:
                            self.drawer.draw(image, face_landmarks, True)
                            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1)[...,::-1])

                frame_counter +=1
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        if return_landmarks:
            return landmarks_list

# def _trans_f(landmarks):
#     landmarks = np.array(detector.findFaceMesh(img, False)[1][0])
#     return torch.Tensor((landmarks - [133.00270142, 201.18676369]) / [46.83234683, 49.95552163]).transpose(1,0).unsqueeze(0)