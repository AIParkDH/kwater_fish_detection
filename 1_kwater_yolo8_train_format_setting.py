import os
import json

# Convert Coco bb to Yolo
def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

# 이미지 파일 경로와 coco라벨링 포멧의 JSON 파일 경로 지정
image_directory = "./dataset/train/"
json_file_path = "./dataset/labels/train.json"

# JSON 파일 읽기 (utf-8 인코딩 사용)
with open(json_file_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)
    
# 어종이 등장하는 이미지들의 정보를 담을 리스트
selected_images = []

# 모든 train이미지에서 어종이 등장하는 이미지 정보만 선택하기
# 즉 annotation 되어 있는 이미지들의 박스 정보를 yolo 형식으로 바꾸기 위함
for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]
    bbox = annotation["bbox"]
    # 어종이 등장하는 경우만 선택
    if category_id >= 0:
        # 이미지 정보 가져오기
        for image_info in data["images"]:
            if image_info["id"] == image_id:
                image_filename = image_info["file_name"]
                image_width = image_info["width"]
                image_height = image_info["height"]
                break
        # 이미지 정보를 저장
        selected_images.append((image_filename, image_width,image_height, category_id, bbox))

#yolo8 학습을 위한 포멧으로 변환하기
for image_filename,image_width, image_height,category_id, bbox in selected_images:
    image_source_path = os.path.join(image_directory, image_filename)
    # 이미지 파일이 존재하면 
    if os.path.exists(image_source_path):
        # bbox 정보를 이용하여 이미지 label정보 저장
        x, y, width, height = bbox
        # coco 박스 형식을  yolo 박스 형식으로 변환하기
        bbox_trans = coco_to_yolo(x, y, width, height,image_width,image_height)
        #yolo8 학습을 위해 각 이미지별 class 와 bbox 정보 저장
        labelinfo = labelinfo = str(category_id) + ' ' + ' '.join(map(str, bbox_trans))
        #yolo8 학습을 위해 각 이미지별 class와 bbox정보를 txt파일로 저장
        output_label = f"./dataset/train_dataset/train/labels/"
        os.makedirs(output_label, exist_ok=True)
        label_path = os.path.join(output_label, image_filename[:-4])
        with open(f'{label_path}.txt', 'w') as file: 
            file.write(labelinfo)                     
        
print("yolo8 학습을 위한 준비 완료")
