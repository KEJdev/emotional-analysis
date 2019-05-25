import cv2
from collections import OrderedDict
import numpy as np
import sys

# ------------- 눈 코 입 턱 인식 부위 -------------

# 턱선 , 오른쪽 눈썹, 왼쪽 눈썹 , 코 윗부분 , 코 아랫 부분 ,
# 오른쪽 눈 , 왼쪽 눈 , 바깥 립 , 안쪽 립 

class FaceData:
     
    _jawLine = [i for i in range(17)]    

    _rightEyebrow = [i for i in range(17,22)]

    _leftEyebrow = [i for i in range(22,27)]
 
    _noseBridge = [i for i in range(27,31)]
 
    _lowerNose = [i for i in range(30,36)]
 
    _rightEye = [i for i in range(36,42)]
  
    _leftEye = [i for i in range(42,48)]
  
    _outerLip = [i for i in range(48,60)]
 
    _innerLip = [i for i in range(60,68)]
 
    
 # 이미지에서 얼굴의 왼쪽 , 위 , 오른쪽, 아래쪽 좌표 가져오기 
 # 기본 값은 모두 0 으로 
 
    def __init__(self, region = (0, 0, 0, 0),
                 landmarks = [0 for i in range(136)]):   
  
# 이미지에서 얼굴이 발견
        self.region = region

# 얼굴에서 좌표 구하기 
        self.landmarks = landmarks

# 얼굴 데이터를 복사     
    def copy(self):
 
        return FaceData(self.region, self.landmarks.copy())

# FaceData 객체가 비어 있는지 확인하기     
    def isEmpty(self):
    
        return all(v == 0 for v in self.region) or \
               all(vx == 0 and vy == 0 for vx, vy in self.landmarks)

# 경계표에 따라 지정된 이미지 자르기     
    def crop(self, image):
     
        left = self.region[0]
        top = self.region[1]
        right = self.region[2]
        bottom = self.region[3]
 
        croppedImage = image[top:bottom+1, left:right+1]

# 잘린 이미지를 복사하기 
        croppedFace = self.copy()
        croppedFace.region = (0, 0, right - left, bottom - top)
        croppedFace.landmarks = [[p[0]-left, p[1]-top] for p in self.landmarks]

        return croppedImage, croppedFace

# 이미지에 얼굴 표식 그리기
# 얼굴 데이터를 그리는 이미지 데이터 , 기본 값 True
        
    def draw(self, image, drawRegion = None, drawFaceModel = None):
       
        if self.isEmpty():
            raise RuntimeError('얼굴을 찾을 수가 없습니다 .')

        if drawRegion is None:
            drawRegion = True
        if drawFaceModel is None:
            drawFaceModel = True

# 이미지가 있으면 이미지 표식을 그린다 .
        if drawRegion:
            cv2.rectangle(image, (self.region[0], self.region[1]),
                                 (self.region[2], self.region[3]),
                                 (0, 0, 255), 2)

# 위치 표시 
        color = (0, 255, 255)
        for i in range(68):
            cv2.circle(image, tuple(self.landmarks[i]), 1, color, 2)

        if drawFaceModel:
            c = (0, 255, 255)
            p = np.array(self.landmarks)

            cv2.polylines(image, [p[FaceData._jawLine]], False, c, 2)
            cv2.polylines(image, [p[FaceData._leftEyebrow]], False, c, 2)
            cv2.polylines(image, [p[FaceData._rightEyebrow]], False, c, 2)
            cv2.polylines(image, [p[FaceData._noseBridge]], False, c, 2)
            cv2.polylines(image, [p[FaceData._lowerNose]], True, c, 2)
            cv2.polylines(image, [p[FaceData._leftEye]], True, c, 2)
            cv2.polylines(image, [p[FaceData._rightEye]], True, c, 2)
            cv2.polylines(image, [p[FaceData._outerLip]], True, c, 2)
            cv2.polylines(image, [p[FaceData._innerLip]], True, c, 2)

        return image

class GaborData:
   
# 그려진 얼굴에 대한 gabor 응답 
    def __init__(self, features = [0.0 for i in range(2176)]):
      
        self.features = features
      
# 데이터를 복사해서 gabordata 반환 
    def copy(self):
     
        return GaborData(self.features.copy())

# 객체가 비어 있는지 확인하기 
    def isEmpty(self):
       
        return all(v == 0 for v in self.features) 
