import os
from collections import OrderedDict
import numpy as np

from gabor import GaborBank
from data import FaceData
from faces import FaceDetector

from sklearn import svm
from sklearn.externals import joblib

# 모델을 로드 할 수 없음을 나타내는 예외 코드 
class InvalidModelException(Exception):
    pass
  
class EmotionsDetector:
    def __init__(self):
    
        self._clf = svm.SVC(kernel='rbf', gamma=0.001, C=10,
                            decision_function_shape='ovr',
                            probability=True, class_weight='balanced')

# 감정은 7가지 라벨로 분류 
# 중립 , 행복 , 슬픔 , 분노 , 공포 , 놀라움 , 혐오감 
        self._emotions = OrderedDict([
                             (0, 'neutral'), (1, 'happiness'), (2, 'sadness'),
                             (3, 'anger'), (4, 'fear'),  (5, 'surprise'),
                             (6, 'disgust')
                         ])

# 모델 파일 로드 
        modulePath = os.path.dirname(__file__)
        self._modelFile = os.path.abspath('D:\\workspace\\emotional-analysis\\models\\emotions_model.dat'\
                            .format(modulePath))
      
# 파일이 있는 경우 디스크에서 모델 로드 
        if not os.path.isfile(self._modelFile):
            raise InvalidModelException('모델이 없습니다 ')

        if not self.load():
            raise InvalidModelException('모델을 로드 할 수 없습니다')
            
# SVM 모델을 로드
    def load(self):
       
        try:
            clf = joblib.load(self._modelFile)
        except:
            return False

        self._clf = clf
        return True

    def _relevantFeatures(self, gaborResponses, facialLandmarks):
        points = np.array(facialLandmarks)

        try:

# 얼굴이 절반밖에 없어도 탐지 가능 . 
# 이 경우 이미지 외부 랜드 마크 응답에 대해 0.0으로 가정
            responses = gaborResponses[:, points[:, 1], points[:, 0]]
        except:
            w = gaborResponses.shape[2]
            h = gaborResponses.shape[1]

            responses = np.zeros((32, 68), dtype=float)
            for i in range(len(points)):
                x = points[i][0]
                y = points[i][1]
                if x < w and y < h:
                    responses[:, i] = gaborResponses[:, y, x]
                else:
                    responses[:, i] = 0.0

# 2차원 행렬을 단일 차원으로 재구성
        featureVector = responses.reshape(-1).tolist()

        return featureVector

# 주어진 특징에 따라 감정을 감지
    def detect(self, face, gaborResponses):
        
        features = self._relevantFeatures(gaborResponses, face.landmarks)

        return self.predict(features)

# 주어진 특징 벡터에 대한 감정을 예측 
    def predict(self, features):
   
        probas = self._clf.predict_proba([features])[0]

        ret = OrderedDict()
        for i in range(len(self._emotions)):
            label = self._emotions[i]
            ret[label] = probas[i]
        return ret
