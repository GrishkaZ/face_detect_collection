from abc import abstractmethod
from typing import Iterable
import numpy as np
import os
import cv2

#to add ?:
#https://www.robots.ox.ac.uk/~vgg/software/vgg_face/


root_dir = os.path.join(os.path.dirname(__file__), 'models/face_embeddings_models/')

def normalize(arr):
    return arr / np.linalg.norm(arr)

#todo add face warper by lanmarks

class _BasicFaceEmbeddingsModel:

    @abstractmethod
    def images2embeddings(self, faces: Iterable, *args, **kwargs):
        pass


class IResnet(_BasicFaceEmbeddingsModel):

    __image_size = (112, 112,)

    def __init__(self, backbone = 'resnet18', framework = 'opencv'):
        if backbone == 'resnet18':
            model_path = os.path.join(root_dir, 'iresnet18.onnx')
        else:
            raise ValueError('backbone in not one of [resnet18]')

        if framework == 'opencv':
            self._emb_model = cv2.dnn.readNetFromONNX(model_path)
            def get_emb(blob):
                self._emb_model.setInput(blob)
                return self._emb_model.forward()
        elif framework == 'ort':
            import onnxruntime as ort
            self._inf_sess = ort.InferenceSession(model_path)
            def get_emb(blob):
                res_ort = self._inf_sess.run(None, {'input.1': blob})
                return np.array(res_ort).squeeze()
        else:
            raise ValueError('framework in not one of [opencv, ort]')

        self._get_emb = get_emb


    def images2embeddings(self, faces: Iterable, batch_size = 1):

        blob = cv2.dnn.blobFromImages(faces, 1. / 127.5, (112, 112,), (127.5, 127.5, 127.5), swapRB=False).astype(np.float32)

        res = []
        for batch in np.array_split(blob, np.ceil(len(blob) / 16)):
            res_batch = self._get_emb(batch)
            res_batch = normalize(res_batch)
            res.extend(res_batch.tolist())

        return res

