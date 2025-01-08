import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Asegúrate de instalar SORT: pip install filterpy

# Diccionario para mapear IDs de clases a nombres de objetos
CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "motorbike",
    5: "bus",
}

def load_model():
    # Cargar el modelo YOLO
    model = YOLO("yolov10m.pt")
    return model

def detect_and_track_objects_with_ids(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    # Instancia del rastreador SORT
    tracker = Sort()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detección de objetos con YOLO
        results = model.predict(frame, conf=0.5)  # Detectar objetos con confianza mínima del 50%
        detections = []

        # Extraer detecciones y convertirlas a un arreglo NumPy
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordenadas del cuadro
            conf = box.conf[0].item()  # Confianza de la detección
            cls = int(box.cls[0].item())  # Clase del objeto
            detections.append([x1, y1, x2, y2, conf, cls])
        detections = np.array(detections)  # Convertir a NumPy

        # Actualizar rastreador con detecciones actuales
        if detections.size > 0:
            tracked_objects = tracker.update(detections[:, :5])  # Pasar solo [x1, y1, x2, y2, conf]
        else:
            tracked_objects = []

        # Dibujar cajas, IDs, nombres de objetos y porcentaje en el cuadro
        for obj, det in zip(tracked_objects, detections):
            x1, y1, x2, y2, obj_id = map(int, obj)
            conf = det[4] * 100  # Convertir confianza a porcentaje
            cls = int(det[5])  # Recuperar la clase del objeto
            label = CLASS_NAMES.get(cls, "Unknown")  # Obtener el nombre de la clase

            # Dibujar cuadro delimitador
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Cuadro rojo, grosor 4

            # Dibujar texto con ID, nombre y porcentaje (solo 3 dígitos)
            cv2.putText(
                frame,
                f"ID: {obj_id}, {label}, {conf:.1f}%",
                (x1, y1 - 20),  # Posición sobre el cuadro
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,  # Tamaño de fuente más grande
                (0, 0, 255), 3  # Texto rojo, grosor 3
            )

        # Escribir el cuadro con las anotaciones en el archivo de salida
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video procesado y guardado en {output_path}")

def main():
    model = load_model()
    video_path = os.getenv('VIDEO_PATH', '/app/data/DJI_20241111152049_0053_D2.mp4')
    output_path = '/app/data/output/detected_objects_1.mp4'
    detect_and_track_objects_with_ids(model, video_path, output_path)

if __name__ == "__main__":
    main()


