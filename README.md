# kwater_fish_detection

https://aifactory.space/task/2600/overview


 ![20231023103425_hanM](https://github.com/AIParkDH/kwater_fish_detection/assets/130961538/0fde8eb4-d73e-4a20-bb14-47a80eb31731)




1_kwater_yolo8_train_format_setting
2_kwater_yolo8_train
3_kwater_yolo8_test
4_kwater_yolo8_test_submit_setting

이렇게 4개 파일로 구성

1_kwater_yolo8_train_format_setting : 대회측에서 제공해준 train데이터에 대한 coco형식의 bbox를 yolo 형식으로 바꾸고,
                                      이미지 하나당 txt 파일 하나씩 저장시켜서 yolo8 학습을 준비해주는 코드

2_kwater_yolo8_train : 사전 학습된 yolo8모델을 불러와 epochs 설정 등 훈련간 사용할 수 있는 옵션들을 설정하여 나의 데이터에 맞게 학습을 진행
                       yaml파일에 학습이미지와 test 이미지 경로를 잘 입력하고, train/images 폴더에는 학습할 이미지들을 넣어두고,
                       train/labels 폴더에는 위 1번코드에서 실행결과로 생성된 coco json 파일을 yolo txt 파일로 변환시킨 텍스트 파일들을 넣어두면된다.
                       특히 이미지와 txt 파일이 1:1로 매칭되어 같은 파일명으로 저장되어 있어야 한다.(확장자만 다름)
                       그리고 yolo8 에서 좋은 기능이 배경을 학습 시킬수 있다는 것이다.
                       배경 학습방법은 train/images 에 배경 이미지들을 넣기만 하면 된다.(train/labels폴더에 txt 파일 없이)
                       그러면 배경으로 인식하여 자동으로 배경이 학습되어 물고기가 없는데 있다고 box를 치거나 하는 오탐률을 많이 낮출 수 있다.

3_kwater_yolo8_test : 위 2번 코드로 학습된 모델을 사용하여 predict 
                      test 이미지 경로를 파라미터로 넘겨주고, save_txt=True 를 해주면 이미지에서 detection 된 결과를 각각의 txt 파일로 저장

4_kwater_yolo8_test_submit_setting : 최종 3번 코드 실행으로 생성된 txt 파일들을 coco 형식에 맞춰 변경해주고, 대회에 제출할 json 파일을 만드는 코드
