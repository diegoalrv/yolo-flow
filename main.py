import os
import cv2
import torch
from ultralytics import YOLO


def load_model():
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = YOLO("yolov10m.pt")
    return model

def detect_objects(model, image_path, output_path):
    img = cv2.imread(image_path)
    results = model.predict(img, classes=[2])
    print(results)

    # Obtiene la imagen con las detecciones
    img_with_boxes = results[0]
    print(type(img_with_boxes))
    print(img_with_boxes)
    # Guarda la imagen con las detecciones
    # cv2.imwrite(output_path, img_with_boxes)
    img_with_boxes.save(output_path)
    print(f"Image saved to {output_path}")

def detect_objects_in_video(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        img_with_boxes = results[0].plot()
        out.write(img_with_boxes)

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

def main():
    model = load_model()
    # image_path = os.getenv('IMAGE_PATH', '/app/image.png')
    # output_path = '/app/detected_objects.png'
    # detect_objects(model, image_path, output_path)

    video_path = os.getenv('VIDEO_PATH', '/app/data/DJI_20241111152049_0053_D2.MP4')
    output_path = '/app/data/output/detected_objects.mp4'
    detect_objects_in_video(model, video_path, output_path)

if __name__ == "__main__":
    main()
