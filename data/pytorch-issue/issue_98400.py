from ultralytics import YOLO
import cv2

model = YOLO("yolov8_proj/models/yolov8x_6batch.engine")
results = model("streams.streams", stream=True, show=True, classes=[0], task="detect")  # setting streams=True
result = next(results)

while True:
    result = next(results)
    people_count = list(map(lambda x:x.boxes.cls.tolist().count(0), results)) # 0 is label for person class
    print("people Count", people_count)    

    if cv2.waitKey(1) == "q":
        break