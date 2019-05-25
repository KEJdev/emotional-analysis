# How did it start?

"재미"를 측정 할 수 있을까 ?  라는 생각을 문뜩하게 되었습니다.  
게임을 할때, 우리는 "재미"를 느낍니다.  

그런데, 어느부분에서 ? , 왜 ? 이런것을 생각해 본적이 있나요 ?  
혹시 게임을 만들 때,  재미 없는 부분을 좀 더 줄여서 재밌는 부분을 늘린다면 ? 좋지 않을까요 ? 

최근에 재밌는 논문들도 보았어요.  
충동 구매 후에 느낀 감정이 긍정적이라면 다시 그 쇼핑했던 곳으로 가서 쇼핑을 하다는 연구결과도 있더라구요.  

즉, 어떤 매장, 어떤 게임, 어떤 음식이던간에 긍정적인 감정을 느끼게 된다면 ? 마케팅에 좋지 않을까 ? 

그래!  
사람의 얼굴에서 나오는 순간순간의 감정을 데이터로 만들어보자 ! 하고 시작하게 되었습니다.  

# Detection of Emotions  
비디오로 찍은 얼굴과 감정을 탐지합니다.  

감정 라벨은 총 7가지입니다.  

- Neutral   
- Happiness   
- Sadness   
- Anger  
- Fear  
- Surprise  
- Disgust  

![](./img/Trump.png)
**<center>비디오 감정 분석 결과</center>**  

# Fun Facts  
프로젝트를 진행하면서 재밌던 사실은,  
트럼프와 오바마 두 대통령의 연설 장면을 감정 분석하였을때 입니다.

연설 내용 중 비슷한 이야기 하고 있는 두 부분을 비디오로 담아 감정 분석을 했는데, 트럼프 대통령의 경우 혐오감, 분노 등의 감정 카운트가 많이 되었으며, 오바마 대통령의 경우 행복, 놀라움 등의 감정 카운트가 되었습니다.

이 결과는 여러가지 가정을 세울 수 있는데,   
이야기는 더 길어 질 것 같으므로 자세한 이야기는 블로그에서 다룰 예정입니다.   

# Details on the Implementation
얼굴 검출은 Cascade Detector(Viola-Jones algorithm) 을 기반으로 하며, 감정 감지는 RBF 커널을 사용하는 SVM(Support Vector Machine)을 기반으로 합니다.

# Usage
tsst.py 는 비디오를 탐지 및 테스트 하는데 사용됩니다.  
실제 사용하려면 코드 중간에 있는 경로를 수정하여 사용해야합니다.  

# License
Some of the codes follow MIT License.  
The code is licensed under MIT license and can be used as desired. Only the trained models (the files in model folder) can not be used for commercial purposes due to the limitations on the licenses of the datasets used to train them.

If you want to use the code for commercial purposes, you must train your own model from your own image datasets labelled with the prototypic emotions.
