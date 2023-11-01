import numpy as np
import mediapipe as mp
import cv2
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

root_dir = os.path.join(os.path.dirname(__file__), 'models/')

#todo features by triangulation

class MediapipeFaceAligner():

    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceAligner = mp.tasks.vision.FaceAligner
        FaceAlignerOptions = mp.tasks.vision.FaceAlignerOptions

        options = FaceAlignerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(root_dir, 'face_landmarker.task')),
        )

        self.face_aligner_model = FaceAligner.create_from_options(options)


    def align(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        image = self.face_aligner_model.align(mp_image)
        return image.numpy_view()

    def __del__(self):
        self.face_aligner_model.close()

class MediapipeLandmarksDrawer:

    #todo set drawing params
    def __init__(self, color=(0, 0, 255), thickness=1, circle_radius=1):
        self.drawing_spec = mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)

    def draw(self, image, face_landmarks, inplace=False):

        annotated_image = image if  inplace else image.copy()
        try:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        except:
            pass
        try:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
        except:
            pass
        try:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        except:
            pass

        if not inplace:
            return annotated_image


#todo return modes: images, landmarks, show image/video
class MediapipeFaceMeshDetector:

    """
        drawer: draw landmarks if not None
    """
    def __init__(self, drawer: MediapipeLandmarksDrawer = None):
        self.drawer = drawer

    @staticmethod
    def get_mesh_coordinates(multi_face_landmarks, img_w, img_h):
        faces = []
        if multi_face_landmarks:
            for faceLms in multi_face_landmarks:
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face.append([x, y])
                faces.append(face)
        return faces


    '''
      For static images.
      params: see mediapipe FaceMesh docks
    '''
    def detect_on_images(self, images, return_landmarks = True,
                         max_num_faces = 1, refine_landmarks = False, min_detection_confidence=0.5):

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
                    landmarks_list.append(self.get_mesh_coordinates(results.multi_face_landmarks,
                                                                    image.shape[1], image.shape[0]))
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
                        landmarks_list.append((frame_counter, self.get_mesh_coordinates(results.multi_face_landmarks,
                                                                        image.shape[1], image.shape[0])))
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


    @staticmethod
    def get_simple_features_by_coordinates(landmarks_coordinates):
        # овал лица
        face_circle = [162, 21, 54, 103, 67, 109, 10, 338,
                       297, 332, 284, 251, 389, 356, 454, 366, 361, 435, 367, 397, 365, 379,
                       378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132,
                       137, 227, 127]

        # ширина высота лица
        face_height = [10, 152]
        face_width = [227, 447]

        left_eye = [226, 130, 246, 161, 160, 159, 158, 157, 173, 155,
                    154, 153, 145, 144, 163, 25]
        right_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249,
                     390, 373, 374, 380, 381, 382]

        # ширина высота глаз
        left_eye_height = [159, 145]
        left_eye_width = [226, 155]

        right_eye_height = [386, 374]
        right_eye_width = [263, 382]

        # расстояние между глазами (делить на высоту лица)
        eye_dist = [155, 382]

        # нос
        nose = [55, 193, 122, 196, 3, 236, 198, 209, 49, 48, 64, 240, 60, 2,
                328, 460, 439, 278, 279, 429, 420, 456, 248, 419, 351, 417, 285, 8]

        # ширины носа
        nose_width_1 = [55, 285]
        nose_width_2 = [3, 248]
        nose_width_3 = [64, 439]

        # высота носа
        nose_height = [8, 2]

        # губы
        lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405,
                314, 17, 84, 181, 91, 146]

        # брови
        left_brow = [70, 53, 52, 65, 55]
        right_brow = [285, 295, 282, 283, 300]

        landmarks = np.array(landmarks_coordinates).squeeze()
        face_area = cv2.contourArea(landmarks[face_circle])

        ret_dict = dict()

        # ширина/высота лица
        visota_litsa = np.linalg.norm(landmarks[face_height[0]] - landmarks[face_height[1]])
        shirina_litsa = np.linalg.norm(landmarks[face_width[0]] - landmarks[face_width[1]])
        ret_dict['face_wh'] = shirina_litsa / visota_litsa

        # скругленность лица
        circ_center, circ_radius = cv2.minEnclosingCircle(landmarks[face_circle])
        ret_dict['face_round'] = face_area / (circ_radius ** 2 * np.pi)

        # ширина/высота глаз
        left_eye_h = np.linalg.norm(landmarks[left_eye_height[0]] - landmarks[left_eye_height[1]])
        right_eye_h = np.linalg.norm(landmarks[right_eye_height[0]] - landmarks[right_eye_height[1]])

        left_eye_w = np.linalg.norm(landmarks[left_eye_width[0]] - landmarks[left_eye_width[1]])
        right_eye_w = np.linalg.norm(landmarks[right_eye_width[0]] - landmarks[right_eye_width[1]])

        ret_dict['eyes_wh'] = (left_eye_h / left_eye_w + right_eye_h / right_eye_w) / 2

        # расстояние между глазами eye_dist
        ret_dict['eyes_dist'] = np.linalg.norm(landmarks[eye_dist[0]] - landmarks[eye_dist[1]]) / shirina_litsa

        lambda_ = 10

        # площадь глаз
        ret_dict['eyes_area'] = (cv2.contourArea(landmarks[left_eye]) + cv2.contourArea(
            landmarks[right_eye])) * lambda_ / face_area

        # ширина/высота носа
        visota_nosa = np.linalg.norm(landmarks[nose_height[0]] - landmarks[nose_height[1]])
        avg_shirina_nosa = (np.linalg.norm(landmarks[nose_width_1[0]] - landmarks[nose_width_1[1]]) + \
                            np.linalg.norm(landmarks[nose_width_2[0]] - landmarks[nose_width_2[1]]) + \
                            np.linalg.norm(landmarks[nose_width_3[0]] - landmarks[nose_width_3[1]])) / 3
        ret_dict['nose_wh'] = avg_shirina_nosa / shirina_litsa  # visota_nosa

        # высота носа
        ret_dict['nose_h'] = visota_nosa / (visota_litsa * shirina_litsa) ** 0.5

        # ширина/высота губ
        rect = cv2.minAreaRect(landmarks[lips])
        w = min(rect[1])
        h = max(rect[1])
        ret_dict['lips_wh'] = w / h

        # степень округлости губ
        lips_area = cv2.contourArea(landmarks[lips])
        circ_center, circ_radius = cv2.minEnclosingCircle(landmarks[lips])
        ret_dict['lips_round'] = lips_area / (circ_radius**2 * np.pi)

        # площадь губ / площадь лица
        ret_dict['lips_area'] = lips_area * lambda_ / face_area

        # спрямленность бровей
        left_brow_len = np.linalg.norm(landmarks[left_brow][0] - landmarks[left_brow][-1])
        right_brow_len = np.linalg.norm(landmarks[right_brow][0] - landmarks[right_brow][-1])
        ret_dict['brows_strict'] = ((cv2.arcLength(landmarks[left_brow],
                                                   False) - left_brow_len) * 10 / left_brow_len + \
                                    (cv2.arcLength(landmarks[right_brow],
                                                   False) - right_brow_len) * 10 / right_brow_len) / 2

        return ret_dict

        # def _trans_f(landmarks):
#     landmarks = np.array(detector.findFaceMesh(img, False)[1][0])
#     return torch.Tensor((landmarks - [133.00270142, 201.18676369]) / [46.83234683, 49.95552163]).transpose(1,0).unsqueeze(0)


class FaceMeshTriangulator:

    def __init__(self, face_mesh_coordinates, img_w, img_h):
        r = (0, 0, img_w, img_h)
        self.subdiv = cv2.Subdiv2D(r)

        for p in face_mesh_coordinates:
            self.subdiv.insert(p)

    def draw_delaunay(self, img, color, inplace=False):
        if not inplace:
            img = img.copy()
        trangleList = self.subdiv.getTriangleList()
        for t in trangleList:
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))
            cv2.line(img, pt1, pt2, color, 1)
            cv2.line(img, pt2, pt3, color, 1)
            cv2.line(img, pt3, pt1, color, 1)

        if not inplace:
            return img

    def draw_voronoi(self,img, color, inplace = False):
        if not inplace:
            img = img.copy()
        (facets, centers) = self.subdiv.getVoronoiFacetList([])
        facets = [np.array(facet, dtype = int) for facet in facets]
        cv2.polylines(img, facets, True, color, 1)
        if not inplace:
            return img




