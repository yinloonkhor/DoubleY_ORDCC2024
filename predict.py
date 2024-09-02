from ultralytics import YOLO
import argparse
import os
import torch
import csv

# Setup
HOME = os.getcwd()
torch.cuda.empty_cache()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser(description='YOLO Inference Script')
parser.add_argument('model_path', type=str, help='Path to the directory containing model pt files')
parser.add_argument('source_path', type=str, help='Path to the directory containing images for inference')
parser.add_argument('output_path', type=str, help='Path to the output directory')
args = parser.parse_args()

# Load the YOLO model
model_path = args.model_path

model_v5mu = YOLO(os.path.join(model_path,'yolov5mu_best.pt')).to(DEVICE)
model_v8m = YOLO(os.path.join(model_path,'yolov8m_best.pt')).to(DEVICE)
model_v8m_norway = YOLO(os.path.join(model_path,'yolov8m_norway_best.pt')).to(DEVICE)
model_v8x_11cls = YOLO(os.path.join(model_path,'yolov8x_11classes_best.pt')).to(DEVICE)

# Path to the directory containing images for inference
source_path = args.source_path
image_files = os.listdir(source_path)

# Run inference on the images
csv_list = []
for image in image_files:
    file_path = os.path.join(source_path, image)
    result_mu = model_v5mu.predict(source=file_path, conf=0.25, iou=0.999, imgsz=640, device=DEVICE, stream=True)
    result_m = model_v8m.predict(source=file_path, conf=0.25, iou=0.999, imgsz=640, device=DEVICE, stream=True)
    result_m_norway = model_v8m_norway.predict(source=file_path, conf=0.25, iou=0.999, imgsz=640, device=DEVICE, stream=True)
    result_x_11cls = model_v8x_11cls.predict(source=file_path, conf=0.25, iou=0.999, imgsz=640, device=DEVICE, stream=True)

    coor_list = []
    final_coor = ''
    # v5mu variant
    boxes = list(result_mu)[0].boxes
    if len(boxes.cls.tolist()) > 0:
        for cls, bbox in zip(boxes.cls, boxes.xyxy):
            bbox_str = ' '.join([str(int(x)) for x in bbox.tolist()])
            coor = str(int(cls)+1) + ' ' + bbox_str
            coor_list.append(coor)
            
    # m variant
    boxes = list(result_m)[0].boxes
    if len(boxes.cls.tolist()) > 0:
        for cls, bbox in zip(boxes.cls, boxes.xyxy):
            bbox_str = ' '.join([str(int(x)) for x in bbox.tolist()])
            coor = str(int(cls)+1) + ' ' + bbox_str
            coor_list.append(coor)

    # m_norway variant
    boxes = list(result_m_norway)[0].boxes
    if len(boxes.cls.tolist()) > 0:
        for cls, bbox in zip(boxes.cls, boxes.xyxy):
            bbox_str = ' '.join([str(int(x)) for x in bbox.tolist()])
            coor = str(int(cls)+1) + ' ' + bbox_str
            coor_list.append(coor)
            
    # x_11cls variant
    # mapping: {5: 'D00', 6: 'D10', 8: 'D20', 10: 'D40'}
    boxes = list(result_x_11cls)[0].boxes
    mapping = {5: 0, 6: 1, 8: 2, 10: 3}
    if len(boxes.cls.tolist()) > 0:
        for cls, bbox in zip(boxes.cls, boxes.xyxy):
            if int(cls) in mapping.keys():
                new_cls = mapping[int(cls)]
                bbox_str = ' '.join([str(int(x)) for x in bbox.tolist()])
                coor = str(int(new_cls)+1) + ' ' + bbox_str
                coor_list.append(coor)

    final_coor = ' '.join(list(set(coor_list)))
    csv_list.append([image, final_coor])

file_path = args.output_path
csv_file_path = os.path.join(file_path,'output.csv')
txt_file_path = os.path.join(file_path,'output.txt')

# Prepare the CSV file
with open(csv_file_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for result in csv_list:
        # Write the row to the CSV file
        csv_writer.writerow(result)

# Prepare txt file
with open(txt_file_path, 'w') as txtfile:
    for row in csv_list:
        txtfile.write(','.join(row) + '\n')

print(f"Predictions saved to {file_path}")
