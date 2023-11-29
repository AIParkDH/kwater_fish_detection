import os
import json
#from efficientnet_pytorch import EfficientNet

json_file_path = "./dataset/labels/answer_sample.json"

#Yolo bbox형식을 Coco 형식으로 변환
def yolo_to_coco(x, y, w, h, image_w, image_h):
    x, y, w, h, image_w, image_h = map(float, [x, y, w, h, image_w, image_h])
    x2 = w * image_w
    y2 = h * image_h
    x1 = ((2 * x * image_w - x2) / 2)
    y1 =((2 * y * image_h - y2) / 2)
    # 부동소수점 오차 보정
    epsilon = 1e-10  # 아주 작은 값을 추가하여 오차를 보정
    x1 = max(0, x1 + epsilon)
    y1 = max(0, y1 + epsilon)
    x2 = max(0, x2 + epsilon)
    y2 = max(0, y2 + epsilon)
    return round(x1,2), round(y1,2), round(x2,2), round(y2,2)

# 경진대회 결과 제출 양식인 answer_sample.json 파일 읽기 (test 이미지 id를 동일하게 불러오기 위함)
with open(json_file_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

# 테스트 데이터셋 이미지 폴더 경로
test_image_folder = "./dataset/test/"

# 테스트 데이터셋 이미지 파일 목록
test_image_files = os.listdir(test_image_folder)

# txt 파일 목록
folder_path = "./runs/detect/predict/labels"  # 텍스트 파일이 있는 폴더 경로
text_list = []  # 텍스트 파일의 내용을 저장할 빈 리스트를 생성
# 폴더 내의 모든 txt파일 목록을 가져오기
#       txt 파일목록을 순서에 맞게 정렬하기
#       ex) test_1, test_2 순으로 정렬하여야함. 안그러면 test_1, test_10, test_100 순으로 정렬됨
test_txt_files = os.listdir(folder_path)
test_txt_files = sorted(test_txt_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

# 'images'와 'annotations'에 대한 리스트 초기화
images_list = []
annotations_list = []

anno_id=0
txt_id=0
# 이미지를 하나씩 처리하며 어종 분류 및 annotation
for idx, image_filename in enumerate(test_image_files):
    if image_filename is not None:
        image_id = data["images"][idx]["id"]  # 'images'의 'id'로 사용될 이미지 ID
        image_name = data["images"][idx]["file_name"]
        #yolo8로 detection된 이미지 파일명과 같은 경우 annotation 정보를 저장하기
        if txt_id < len(test_txt_files) and test_txt_files[txt_id][0:-4] == image_name[0:-4]:
            txt_path = os.path.join(folder_path, test_txt_files[txt_id])
            txt_id=txt_id+1
            #detection된 이미지들의 txt파일을 읽어오기
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.readline()
                # class번호와 box 정보 추출
                class_id, x, y, w, h = text.split()
                # yolo형식의 bbox와 area 정보를 coco형식으로 변환
                x1, y1, x2, y2 = yolo_to_coco(x,y,w,h,640,480)
                area = x2  * y2 
                c_id = int(class_id)
                #변환된 annotation 정보 저장
                annotations_list.append({
                    "id": anno_id,  # 'annotations'의 'id'
                    "image_id": image_id,  # 'images'의 'id'를 사용
                    "category_id": c_id,  # 예측된 어종 ID
                    "segmentation": [],  # segmentation 정보가 없으므로 빈 리스트
                    "area": area,  # area 정보
                    "bbox": [x1, y1, x2, y2],  # bbox 정보
                    "iscrowd": 0,
                    "attributes": {
                        "occluded": False,
                        "rotation": 0.0
                    }
                })
                anno_id=anno_id+1
#경진대회 결과 제출 형식 JSON 객체 생성
output_json = {
    "images": data["images"],
    "annotations": annotations_list,
    "categories": data["categories"]
}
# JSON 파일로 저장
with open("./dataset/labels/kwater_results.json", "w", encoding="utf-8") as json_file:
    json.dump(output_json, json_file, ensure_ascii=False, indent=4)

print("테스트 결과가 kwater_results.json 파일로 저장되었습니다.")