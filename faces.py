import os
import numpy as np
import dlib
import cv2

from data import FaceData

# 이미지의 얼굴 탐지기 구현 
class FaceDetector:

# 이미지에서 얼굴을 감지하는데 사용되는 dilb 오브젝트의 인스턴스 
    _detector = None

# 얼굴의 위치를 예측하는데 사용되는 dilb 오브젝트의 인스턴스 
    _predictor = None
  
# 주어진 이미지에서 얼굴을 자동으로 탐지 
# dilb 패키지의 얼굴 검출기 / 예측자 를 사용 
    def detect(self, image, downSampleRatio = None):
   
# 얼굴을 검색할 이미지 데이터 
        if FaceDetector._detector is None or FaceDetector._predictor is None:
            FaceDetector._detector = dlib.get_frontal_face_detector()

            faceModel = os.path.abspath('D:\\workspace\\emotional-analysis\\models\\face_model.dat' \
                            .format(os.path.dirname(__file__)))
            FaceDetector._predictor = dlib.shape_predictor(faceModel)

# 요청이 있는 경우 개선을 위해 원본 이미지의 크기를 줄임 
# 초기 얼굴 검출 서능
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

# 다운 스케일링이 요청된 경우 감지된 영역을 축소
# 해상도가 약간 낮아도 랜드마크는 이미지를 찾을 수 있음
        if downSampleRatio is not None:
            region = dlib.rectangle(region.left() * downSampleRatio,
                                    region.top() * downSampleRatio,
                                    region.right() * downSampleRatio,
                                    region.bottom() * downSampleRatio)

# 얼굴 영역에 모양 모델을 맞추어 위치를 예측 
        faceShape = FaceDetector._predictor(image, region)

# 데이터 반환 
        face = FaceData()

# 예측 된 랜드마크 위치로 오브젝트 데이터를 업데이트 
# 경계 상자 그리기 
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
