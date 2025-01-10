import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist  # Para calcular distancias entre objetos

# Diccionario para mapear IDs de clases a nombres de objetos
CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}

# Parámetros de rastreo
MAX_DISTANCE = 80  # Incrementar distancia máxima permitida entre cuadros
MAX_AGE = 5  # Cuadros que un objeto puede desaparecer antes de perder su ID
MIN_HITS = 3  # Cuadros consecutivos necesarios para confirmar un objeto

# Diccionario para almacenar información del rastreador
object_tracker = {}

def load_model():
    # Cargar el modelo YOLO
    model = YOLO("yolov10m.pt")  # Cambia al modelo YOLO que estés usando
    return model

def calculate_centroid(x1, y1, x2, y2):
    """Calcula el centro de un cuadro delimitador."""
    return (x1 + x2) / 2, (y1 + y2) / 2

def assign_ids(detections, object_tracker):
    """Asigna IDs únicos a las detecciones basándose en su proximidad al cuadro anterior."""
    new_tracker = {}
    current_centroids = np.array([calculate_centroid(*det[:4]) for det in detections])
    previous_centroids = np.array([value['centroid'] for value in object_tracker.values()])

    if len(previous_centroids) > 0 and len(current_centroids) > 0:
        # Calcular distancias entre todos los pares de centroides
        distances = cdist(previous_centroids, current_centroids)

        # Asociar objetos actuales con IDs previos
        for prev_idx, current_idx in zip(*np.where(distances <= MAX_DISTANCE)):
            object_id = list(object_tracker.keys())[prev_idx]
            new_tracker[object_id] = {
                'centroid': current_centroids[current_idx],
                'bbox': detections[current_idx][:4],
                'conf': detections[current_idx][4],
                'cls': detections[current_idx][5],
                'age': 0,  # Reiniciar edad cuando el objeto es detectado
                'hits': object_tracker[object_id]['hits'] + 1,  # Incrementar confirmaciones
            }

    # Asignar nuevos IDs a los objetos no asociados
    next_id = max(object_tracker.keys(), default=0) + 1
    for current_idx in range(len(detections)):
        if not any(np.array_equal(detections[current_idx][:4], value['bbox']) for value in new_tracker.values()):
            new_tracker[next_id] = {
                'centroid': current_centroids[current_idx],
                'bbox': detections[current_idx][:4],
                'conf': detections[current_idx][4],
                'cls': detections[current_idx][5],
                'age': 0,
                'hits': 1,
            }
            next_id += 1

    # Incrementar la edad de los objetos que no fueron detectados en este cuadro
    for object_id, obj in object_tracker.items():
        if object_id not in new_tracker:
            obj['age'] += 1
            if obj['age'] <= MAX_AGE:
                new_tracker[object_id] = obj  # Mantener objeto si no ha superado MAX_AGE

    return new_tracker

def detect_and_track_objects(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    global object_tracker  # Permitir acceso al rastreador global

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar objetos con YOLO
        results = model.predict(frame, conf=0.6)
        detections = []

        # Extraer detecciones
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordenadas del cuadro
            conf = box.conf[0].item()  # Confianza de la detección
            cls = int(box.cls[0].item())  # Clase del objeto
            detections.append([x1, y1, x2, y2, conf, cls])

        # Convertir a NumPy
        detections = np.array(detections)

        # Asignar IDs únicos a las detecciones
        object_tracker = assign_ids(detections, object_tracker)

        # Dibujar cuadros y etiquetas
        for object_id, obj in object_tracker.items():
            if obj['hits'] < MIN_HITS:
                continue  # Ignorar objetos no confirmados

            x1, y1, x2, y2 = map(int, obj['bbox'])
            conf = obj['conf'] * 100  # Confianza como porcentaje
            conf_formatted = f"{conf:.1f}".replace('.', ',')  # Formatear porcentaje con coma
            cls_name = CLASS_NAMES.get(obj['cls'], "Unknown")  # Nombre de la clase

            # Dibujar cuadro delimitador
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Dibujar texto con ID, tipo de objeto y porcentaje
            cv2.putText(frame, f"ID: {object_id}", (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name}, {conf_formatted}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        out.write(frame)  # Escribir cuadro procesado en el video de salida

    cap.release()
    out.release()
    print(f"Video procesado y guardado en {output_path}")

def main():
    model = load_model()
    video_path = os.getenv('VIDEO_PATH', '/app/data/DJI_20241111152049_0053_D2.mp4')
    output_path = '/app/data/output/detected_objects_1.mp4'
    detect_and_track_objects(model, video_path, output_path)

if __name__ == "__main__":
    main()
