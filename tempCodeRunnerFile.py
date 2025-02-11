from ultralytics import YOLO, checks
from collections import defaultdict
import csv

checks()

# Procesamiento
model = YOLO('yolov8_50epochs.pt')
print(model.names)