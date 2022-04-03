# 차선인식 프로젝트

## 전체 구성
1. yolo 사용해서 차선 탐지 [\*1 \*2 \*3] [\*4]
2. opencv로 생성
    - 먼저 python으로 제작하고, c++로도 변환해보기
3. 동영상이니까 1,2초전의 데이터까지 불러와서 학습하면서 탐지
4. 물체도 탐지

<br>

yolo를 써서 물체 인식 함수 따로, 차선 인식 함수 따로 헤서 두개다 불러오면 되지 않을까 생각함. 참고 논문을 읽으면서 파악좀 해보자. 2번은 c언어로 되어 있는 듯하다.


## 단계 및 설명
1. yolo_clone.sh
2. yolo_train.sh
3. 1,2 초전 frame을 저장해놓고 학습데이터로 사용
- while true 안에서 원래 파일 불러와서 훈련 후 탐지해서 모델 저장, 다음 프레임에서 불러와서 다시 학습 후 탐지
- 이것이 realtime??
4. make_dataset.sh (KITTI dataset) 
- git clone 나의 깃허브 KITTI 있는 곳
5. opencv.py
- cam을 통해 윈도우 오픈
6. 물체 탐지 기능 추가

https://github.com/ultralytics/yolov5

https://blog.naver.com/PostView.naver?blogId=ehdrndd&logNo=222462355643&parentCategoryNo=&categoryNo=48&viewDate=&isShowPopularPosts=true&from=search




<detail open>
    <title> 참고 사이트 </title>

[*1.참고 논문](https://www.koreascience.or.kr/article/JAKO202111236685883.pdf)
[*2.참고 논문](http://mie.pcu.ac.kr/research_file/J3NeCelpGcH2L7XnL88lLJsBvotImZMx.pdf)
[*3.참고 논문](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002787699)

[*4.yolo 사용법](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

</detail>


<detail open>
    <title> 기술 스택 </title>

<code><img alt = "3.1 Python" height="20" src="https://cdn.icon-icons.com/icons2/2699/PNG/512/pytorch_logo_icon_170820.png"> pytorch</code> <code><img alt = "3.1 Python" height="20" src="https://cdn.icon-icons.com/icons2/2699/PNG/512/opencv_logo_icon_170887.png"> OpenCV</code> <code><img alt = "3.1 Python" height="20" src="https://cdn.icon-icons.com/icons2/2989/PNG/512/address_delivery_map_tracking_distribution_icon_187242.png"> Yolov5</code>

</detail>