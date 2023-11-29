import numpy as np
from ultralytics import YOLO

# 커스텀 데이터로 추가 학습한 yolo8n 모델 불러오기 
model = YOLO('./best.pt')  

print(type(model.names), len(model.names))
print(model.names)

#불러온 모델로 test 하기 (save 옵션으로 테스트 결과를 이미지 파일로 저장 여부 결정, true시 저장하나 test 이미지용령만큼 여유공간 고려)
#세부 옵션은 yolo8 참조
results = model.predict(source='./dataset/test/', save=True, save_txt=True,conf=0.5)
print(type(results), len(results))

#결과 확인을 위한 참고 확인한 코드로 삭제해도 무방
for result in results:
    uniq, cnt = np.unique(result.boxes.cls.cpu().numpy(), return_counts=True)  # Torch.Tensor를  numpy 로 변환
    uniq_cnt_dict = dict(zip(uniq, cnt))
    print('\n{class num:counts} =', uniq_cnt_dict,'\n')
    for c in result.boxes.cls:
        print('class num =', int(c), ', class_name =', model.names[int(c)])
