import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Asegúrate de instalar SORT

def load_model():
    model = YOLO("yolov10m.pt")  # Cambia el modelo según tu configuración
    return model

def detect_objects_in_video_with_tracking(model, video_path, output_path):
    # Inicializa la captura de video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Inicializa SORT para seguimiento
    tracker = Sort()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detecta objetos en el cuadro actual
        results = model.predict(frame, classes=[0, 2, 3, 5], conf=0.5)  # Detecta personas, autos, buses y motos
        detections = results[0].boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas
        confidences = results[0].boxes.conf.cpu().numpy()  # Confianzas de las detecciones
        class_ids = results[0].boxes.cls.cpu().numpy()  # IDs de las clases detectadas

        # Construir entrada para SORT (xmin, ymin, xmax, ymax, score)
        dets = []
        for box, conf, cls_id in zip(detections, confidences, class_ids):
            dets.append([*box, conf])
        dets = np.array(dets)

        # Actualiza el seguimiento
        tracked_objects = tracker.update(dets)

        # Dibuja las cajas de los objetos rastreados
        for obj in tracked_objects:
            obj_id = int(obj[4])
            xmin, ymin, xmax, ymax = map(int, obj[:4])
            color = (0, 255, 0)  # Verde para todos los objetos
            label = f"ID {obj_id}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Escribe el cuadro procesado en el video de salida
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

def main():
    model = load_model()
    video_path = os.getenv('VIDEO_PATH', '/app/data/DJI_20241111152049_0053_D2.mp4')
    output_path = '/app/data/output/detected_objects_with_tracking.mp4'
    detect_objects_in_video_with_tracking(model, video_path, output_path)

if __name__ == "__main__":
    main()
