import os
import numpy as np
import dlib
import cv2

from data import FaceData

class FaceDetector:
    _detector = None
    _predictor = None
  
# 주어진 이미지에서 얼굴을 자동으로 탐지 
    def detect(self, image, downSampleRatio = None):
        if FaceDetector._detector is None or FaceDetector._predictor is None:
            FaceDetector._detector = dlib.get_frontal_face_detector()

            faceModel = os.path.abspath('D:\\workspace\\emotional-analysis\\models\\face_model.dat' \
                            .format(os.path.dirname(__file__)))
            FaceDetector._predictor = dlib.shape_predictor(faceModel)

        if downSampleRatio is not None:
            detImage = cv2.resize(image, (0, 0), fx=1.0 / downSampleRatio,
                                                 fy=1.0 / downSampleRatio)
        else:
            detImage = image

# 이미지에서 얼굴을 감지 
        detectedFaces = FaceDetector._detector(detImage, 1)
        if len(detectedFaces) == 0:
            return False, None

# 많은 얼굴이 발견 되더라도 첫 번째 얼굴만 고려
        region = detectedFaces[0]

        if downSampleRatio is not None:
            region = dlib.rectangle(region.left() * downSampleRatio,
                                    region.top() * downSampleRatio,
                                    region.right() * downSampleRatio,
                                    region.bottom() * downSampleRatio)

# 얼굴 영역에 모양 모델을 맞추어 위치를 예측 
        faceShape = FaceDetector._predictor(image, region)
        face = FaceData()

        face.landmarks = np.array([[p.x, p.y] for p in faceShape.parts()])

        margin = 10
        x, y, w, h = cv2.boundingRect(face.landmarks)
        face.region = (
                       max(x - margin, 0),
                       max(y - margin, 0),
                       min(x + w + margin, image.shape[1] - 1),
                       min(y + h + margin, image.shape[0] - 1)
                      )

        return True, face
