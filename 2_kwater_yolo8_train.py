import yaml
import torch
from ultralytics import YOLO

#yaml 파일 학습을 위한 경로설정
data = { 'train' : 'C:/kwater/yolo8_dataset/train/images',    #train 이미지 경로
         'val' : 'C:/kwater/yolo8_dataset/valid/images',      #val 이미지 경로
         'test' : 'C:/kwater/yolo8_dataset/test/images',      #test 이미지 경로
         'names' : ['fish0', 'fish1', 'fish2', 'fish3', 'fish4', 'fish5', 'fish6','fish7'], #detection하고자 하는 class 이름
         'nc' : 8 }   #detection class 개수

#yaml 파일에 쓰기/저장
with open('C:/kwater/yolo8_dataset/yolodata.yaml', 'w') as f:  
  yaml.dump(data, f)

#저장한 yaml 파일 읽기
with open('C:/kwater/yolo8_dataset/yolodata.yaml', 'r') as f:  
  aquarium_yaml = yaml.safe_load(f)

#사전 학습된 YOLOv8n detection model 불러오기 
model = YOLO('yolov8n.pt')  

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # 커스텀 데이터로 yolo8n 추가 학습하기
    model.train(data='C:/kwater/yolo8_dataset/yolodata.yaml', epochs=100, patience=30, batch=32, imgsz=416)
    
    # 학습된 모델을 저장
    torch.save(model.state_dict(), 'fish_detection_model_yolov8.pth')


print(type(model.names), len(model.names))
print(model.names)
