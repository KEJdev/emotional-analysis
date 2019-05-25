import sys
import argparse
import cv2
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta
import matplotlib.pylab as plt
from faces import FaceDetector
from data import FaceData
from gabor import GaborBank
from emotions import EmotionsDetector

# 감정 별 cnt 
cnt = 0

# 중립 
neutral_cnt = 0

# 행복 
happiness_cnt = 0

# 슬픔
sadness_cnt = 0

# 분노 
ange_cnt = 0

# 공포 
fear_cnt = 0

# 놀라움 
surprise_cnt = 0

# 혐오감 
disgust_cnt = 0



# 감지 된 얼굴 영역 , 경계표 및 감정을 나타내는 클래스
class VideoData:

    def __init__(self):

        # 얼굴 검출기 인스턴스 
        self._faceDet = FaceDetector()
       
        # Gabor 인스턴스 
        self._bank = GaborBank()
        
        # 감정 탐지기 
        self._emotionsDet = EmotionsDetector()
        
        # 검출 된 마지막 얼굴 데이터 
        self._face = FaceData()
        
        # 감지 된 마지막 검정 데이터
        self._emotions = OrderedDict()

# 주어진 프레임 이미지에서 얼굴과 프로토 타입 감정을 감지 
    def detect(self, frame):
       
# 탐지를 수행할 이밎 
        ret, face = self._faceDet.detect(frame)
        if ret:
            self._face = face

# 얼굴 영역 잘라내기 
            frame, face = face.crop(frame)
     
# Gabor 로 필터링
            responses = self._bank.filter(frame)

# 필터 응답을 기반으로 프로토 타입 감정을 감지 
            self._emotions = self._emotionsDet.detect(face, responses)

            return True
        else:
            self._face = None
            return False


# 지정된 프레임 이미지의 감지 된 데이터를 그림
    def draw(self, frame):

# 글꼴 설정 , 스케일 , 두께 , 글로우 , count 
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        glow = 3 * thick

        global cnt 
        global neutral_cnt
        global happiness_cnt 
        global sadness_cnt 
        global ange_cnt
        global fear_cnt 
        global surprise_cnt
        global disgust_cnt 

        
        black = (0, 0, 0)
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        red = (0, 0, 255)

        empty = True

# 얼굴 표식과 얼굴 거리를 그림
        x = 5
        y = 0
        w = int(frame.shape[1]* 0.2)
        try:
            face = self._face
            empty = face.isEmpty()
            face.draw(frame)
        except:
            pass

        try:
            emotions = self._emotions
            if empty:
                labels = []
                values = []
            else:
                labels = list(emotions.keys())
                values = list(emotions.values())
                bigger = labels[values.index(max(values))]

                text = 'emotions'  # 감정 
                size, _ = cv2.getTextSize(text, font, scale, thick)
                y += size[1] + 20
                
                # 프레임, 텍스트, (x,y), 글꼴, 크기 , 검정 , 광선 
                cv2.putText(frame, text, (x, y), font, scale, black, glow)

                # 프레임 ,텍스트 ,(x,y), 글꼴 , 크기, 노란색 , 두께
                cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

                y += 5
                cv2.line(frame, (x,y), (x+w,y), black, 1)
            
            # 행복 , 글꼴 ,크기 , 두께 
            size, _ = cv2.getTextSize('happiness', font, scale, thick)
            t = size[0] + 20
            w = 150
            h = size[1]

                
            for l, v in zip(labels, values):
                lab = '{}:'.format(l)
                val = '{:.2f}'.format(v)
                size, _ = cv2.getTextSize(l, font, scale, thick)
                

                # 가장 큰 확률은 red 로 나머진 yellow 
                color = red  if l == bigger else yellow
                    
                 
                if color == red:
                    if l == 'neutral':
                        neutral_cnt += 1 

                    if l == 'happiness':
                        happiness_cnt +=1
                    
                    if l == 'sadness':
                        sadness_cnt += 1
                    
                    if l == 'anger' :
                        ange_cnt += 1

                    if l == 'fear' :
                        fear_cnt += 1
                        
                    if l == 'surprise':
                        surprise_cnt += 1
                        
                    if l == 'disgust':
                        disgust_cnt += 1
                        
                        
                    
                    cnt =[neutral_cnt ,happiness_cnt,sadness_cnt,ange_cnt ,fear_cnt ,surprise_cnt,disgust_cnt]
                    
                        
                    text1 = ' count '
                    size, _ = cv2.getTextSize(text1, font, scale, thick)
                    a = 300   # →
                    b = 33  # ↓
                    cv2.putText(frame, text1, (a, b), font, scale, black, glow)
                    cv2.putText(frame, text1, (a, b), font, scale, yellow, thick)    
                        
                    text2 = '{:d}'.format(cnt[0]) # 무표정    
                    a = 300   # →
                    b = 66  # ↓
                    cv2.putText(frame, text2, (a, b), font, scale, black, glow)
                    cv2.putText(frame, text2, (a, b), font, scale, yellow, thick)   
                    
                    text3 = '{:d}'.format(cnt[1]) # 행복
                    a = 300   # →
                    b = 90  # ↓
                    cv2.putText(frame, text3, (a, b), font, scale, black, glow)
                    cv2.putText(frame, text3, (a, b), font, scale, yellow, thick) 
                    
                    text4 = '{:d}'.format(cnt[2])   # 슬픔 
                    a = 300   # →
                    b = 111  # ↓
                    cv2.putText(frame, text4, (a, b), font, scale, black, glow)
                    cv2.putText(frame, text4, (a, b), font, scale, yellow, thick) 
                    
                    text5 = '{:d}'.format(cnt[3])   # 분노 
                    a = 300   # →
                    b = 140  # ↓
                    cv2.putText(frame, text5, (a, b), font, scale, black, glow)
                    cv2.putText(frame, text5, (a, b), font, scale, yellow, thick) 
                    
                    text6 = '{:d}'.format(cnt[4])   # 공포
                    a = 300   # →
                    b = 170  # ↓
                    cv2.putText(frame, text6, (a, b), font, scale, black, glow)
                    cv2.putText(frame, text6, (a, b), font, scale, yellow, thick) 
                    
                    text7 = '{:d}'.format(cnt[5])   # 놀라움 
                    a = 300   # →
                    b = 200  # ↓
                    cv2.putText(frame, text7, (a, b), font, scale, black, glow)
                    cv2.putText(frame, text7, (a, b), font, scale, yellow, thick) 
                                        
                    text8 = '{:d}'.format(cnt[6])   # 혐오감 
                    a = 300   # →
                    b = 230  # ↓
                    cv2.putText(frame, text8, (a, b), font, scale, black, glow)
                    cv2.putText(frame, text8, (a, b), font, scale, yellow, thick) 
                             

                y += size[1] + 15

                p1 = (x+t, y-size[1]-5)
                p2 = (x+t+w, y-size[1]+h+5)
                cv2.rectangle(frame, p1, p2, black, 1)

                # 확률에 비례하는 채워진 사각형을 그립니다.
                p2 = (p1[0] + int((p2[0] - p1[0]) * v), p2[1])
                cv2.rectangle(frame, p1, p2, color, -1)
                cv2.rectangle(frame, p1, p2, black, 1)
                
                # 감정 레이블을 그립니다
                # 프레임 , 레이블 , 폰트 , 블랙 , 글로우
                # 프레임 , 실습 , 글꼴 , 크기 , 색상 , 두께 
                cv2.putText(frame, lab, (x, y), font, scale, black, glow)
                cv2.putText(frame, lab, (x, y), font, scale, color, thick)

                # 감정 확률의 값을 그립니다 
                # 프레임, 값 , 폰트 , 스케일 , 블랙 , 글로우 
                # 프레임, 값 , 글꼴 , 크기 , 흰색 , 두꺼운 
                cv2.putText(frame, val, (x+t+5, y), font, scale, black, glow)
                cv2.putText(frame, val, (x+t+5, y), font, scale, white, thick)

        except Exception as e:
            print(e)
            pass

def main(argv):

    args = parseCommandLine(argv)
    
# 비디오를 로드하거나 웹캠을 시작
    if args.source == ' cam ':
        video = cv2.VideoCapture(args.id)
        if not video.isOpened():
            print('웹캠을 여는 중 오류가 발생했습니다 '.format(args.id))
            sys.exit(-1)

        fps = 0
        frameCount = 0
        sourceName = 'Webcam #{}'.format(args.id)
    else:
        video = cv2.VideoCapture('D:\\workspace\\emotional-analysis\\오바마_대학 연설.mp4') 
        if not video.isOpened():
            print('비디오 파일을 여는 중 오류가 발생했습니다 '.format(args.file))
            sys.exit(-1)

        fps = int(video.get(cv2.CAP_PROP_FPS))
        frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        sourceName = args.file
            
        fourcc = cv2.VideoWriter_fourcc ( *'DIVX' )
        out = cv2.VideoWriter( 'D:\\workspace\\emotional-analysis\\오바마_cnt.avi' , fourcc, fps, ( 720 , 720 ))
        
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);

    data = VideoData()

    paused = False
    frameNum = 0
    
# 비디오 처리 
    while True:

        if not paused:
            start = datetime.now()

        ret, img = video.read()
        if ret:
            frame = img.copy()
        else:
            paused = True

        drawInfo(frame, frameNum, frameCount, paused, fps, args.source)

        data.detect(frame)
        data.draw(frame)
       
        # 얼굴 인식된 이미지 화면 표시

        cv2.imshow(sourceName, frame)
        out.write(frame)
        
        if paused:
            key = cv2.waitKey(0)
        else:
            end = datetime.now()
            delta = (end - start)
            if fps != 0:
                delay = int(max(1, ((1 / fps) - delta.total_seconds()) * 1000))
            else:
                delay = 1
                
            key = cv2.waitKey(delay)

                
# ESC 또는 q , Q 를 누르면 종료 
        if key == ord('q') or key == ord('Q') or key == 27:
            break
     
        if not paused:
            frameNum += 1

    video.release()
    cv2.destroyAllWindows()

# 지정된 프레임 번호와 관련된 텍스트 정보를 프레임 이미지에 그림
def drawInfo(frame, frameNum, frameCount, paused, fps, source):

# 글꼴 , 스케일 , 두께 , 글로우 
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    glow = 3 * thick

    black = (0, 0, 0)
    yellow = (0, 255, 255)

    if source == 'video':
        text = 'Frame: {:d}/{:d} {}'.format(frameNum, frameCount - 1,
                                            '(paused)' if paused else '')
    else:
        text = 'Frame: {:d} {}'.format(frameNum, '(paused)' if paused else '')
    size, _ = cv2.getTextSize(text, font, scale, thick)
    x = 5
    y = frame.shape[0] - 2 * size[1]
    cv2.putText(frame, text, (x, y), font, scale, black, glow)
    cv2.putText(frame, text, (x, y), font, scale, yellow, thick)


def parseCommandLine(argv):
    parser = argparse.ArgumentParser(description='얼굴과 감정을 테스트 합니다')

    parser.add_argument('source', nargs='?', const='Yes',
                        choices=['video', 'cam'], default='cam')


    parser.add_argument('-f', '--file', metavar='<name>')


    parser.add_argument('-i', '--id', metavar='<number>', default=0, type=int)


    args = parser.parse_args()

    if args.source == 'video' and args.file is None:
        parser.error('-f is required when source is "video"')

    return args

if __name__ == '__main__':
    main(sys.argv[1:])
    
