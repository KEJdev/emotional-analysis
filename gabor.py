import numpy as np
from skimage.filters import gabor_kernel
import cv2

class KernelParams:

# 주어진 gabor 커널를 나타내는 클래스
    def __init__(self, wavelength, orientation):

        self.wavelength = wavelength
       
        self.orientation = orientation

# 해시 값을 생성 
    def __hash__(self):

        return hash((self.wavelength, self.orientation))

# 다른 인스턴스와 이 객체 인스턴스가 동일한지 확인 
    def __eq__(self, other):

        return (self.wavelength, self.orientation) == \
               (other.wavelength, other.orientation)

# 이 객체와 다른 인스턴스와 다른지 확인 
    def __ne__(self, other):

        return not(self == other)

class GaborBank:

    def __init__(self, w = [4, 7, 10, 13],
                       o = [i for i in np.arange(0, np.pi, np.pi / 8)]):

        self._wavelengths = w
     
        self._orientations = o
 
        self._kernels = {}


        for wavelength in self._wavelengths:
            for orientation in self._orientations:

                # 주파수 = 1 / 파장
                frequency = 1 / wavelength

                # 파장 = 빈도, 방향    
                kernel = gabor_kernel(frequency, orientation)
                par = KernelParams(wavelength, orientation)
                self._kernels[par] = kernel

    def filter(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        responses = []
        for wavelength in self._wavelengths:
            for orientation in self._orientations:

                frequency = 1 / wavelength
                par = KernelParams(wavelength, orientation)
                kernel = self._kernels[par]
                
                # cv2.filter2D 2D 이미지 
                # cv2.CV_32F 픽셀을 0 - 1.0 사이의 값으로 변환 
                real = cv2.filter2D(image, cv2.CV_32F, kernel.real)
                imag = cv2.filter2D(image, cv2.CV_32F, kernel.imag)
                
                # 주파수 계산 함수 cv2.magnitude
                mag = cv2.magnitude(real, imag)
                cv2.normalize(mag, mag, -1, 1, cv2.NORM_MINMAX)

                responses.append(mag)

        return np.array(responses)
