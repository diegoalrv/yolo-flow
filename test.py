from ultralytics import YOLO, checks
from collections import defaultdict
import csv

checks()

# Procesamiento
model = YOLO('citylab_50epochs.pt')

clases_a_detectar = [0, 3, 8, 9]  # 0 = pedestrian, 3 = car, 8 = bus, 9 = motor

objetos_unicos = {}
contador_clases = defaultdict(int)

# Archivo CSV para guardar coordenadas
with open("detecciones.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Clase", "ID", "Xmin", "Ymin", "Xmax", "Ymax"])  # Encabezado mejorado

    results = model.track(
        source="data\DJI_20241111172713_0057_D2.mp4",
        conf=0.4,
        iou=0.5,
        save=True,
        show=True,
        project="output",
        name="detectedObjects_#1",
        classes=clases_a_detectar,
        tracker="bytetrack.yaml",
        persist=True,
        show_labels=True,
        show_conf=True,
    )

    id_counter = 1  # Inicializar ID desde 1
    frame_number = 0  # Contador de frames

    for frame in results:
        frame_number += 1  # Incrementa el número de frame en cada iteración
        for obj in frame.boxes:
            original_id = int(obj.id[0]) if obj.id is not None else None
            class_id = int(obj.cls[0])
            coords = obj.xyxy[0].tolist()  # Extraer coordenadas (xmin, ymin, xmax, ymax)

            # Asignar un nuevo ID si es la primera vez que se detecta
            if original_id not in objetos_unicos:
                objetos_unicos[original_id] = id_counter
                id_counter += 1
                contador_clases[class_id] += 1  # Asegurar conteo correcto de clases
            new_id = objetos_unicos[original_id]

            print(f"🔹 Frame: {frame_number}, Clase: {model.names[class_id]}, ID: {new_id}, Coordenadas: {coords}")

            writer.writerow([frame_number, model.names[class_id], new_id, *coords])  # Escribir en CSV

print("\n🔹 Conteo final de objetos detectados:")
for class_id, count in contador_clases.items():
    print(f"{model.names[class_id]}: {count}")

print("\n✅ Procesamiento completado. Las coordenadas se han guardado en 'detecciones.csv'.")