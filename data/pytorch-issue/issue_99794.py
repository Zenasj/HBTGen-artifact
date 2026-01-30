import cv2
import numpy as np
from ultralytics import YOLO

cv2.namedWindow("mps_test",cv2.WINDOW_NORMAL)
image = cv2.imread("pose_test.jpeg")
model = YOLO('yolov8s-pose.pt')  # load an official model

results = model(source=image,device="cpu",conf=0.6)
#results = model(source=image,device="mps",conf=0.6)

result_image = results[0].plot()

cv2.imshow("cpu_test", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()