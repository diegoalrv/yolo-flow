import os

def filter_invalid_classes(input_dir, allowed_classes):
    for annotation_file in os.listdir(input_dir):
        if annotation_file.endswith('.txt'):
            input_path = os.path.join(input_dir, annotation_file)
            with open(input_path, 'r') as f_in:
                lines = f_in.readlines()

            # Filtra las líneas con clases no permitidas
            valid_lines = []
            for line in lines:
                fields = line.strip().split()
                if len(fields) > 0:
                    class_id = int(fields[0])  # La clase está en la primera posición
                    if class_id in allowed_classes:
                        valid_lines.append(line)

            # Sobrescribe el archivo solo con las líneas válidas
            with open(input_path, 'w') as f_out:
                f_out.writelines(valid_lines)

if __name__ == "__main__":
    train_labels_dir = r"C:\Users\danie\Desktop\PRACTICA CITYLAB\Profesional\yolo-flow\data\labels\train"
    val_labels_dir = r"C:\Users\danie\Desktop\PRACTICA CITYLAB\Profesional\yolo-flow\data\labels\val"

    # Índices de clases permitidas
    allowed_classes = [0, 4, 9, 10]  # person, car, bus, motorcycle

    print("[INFO] Filtrando clases no válidas en las etiquetas de entrenamiento...")
    filter_invalid_classes(train_labels_dir, allowed_classes)

    print("[INFO] Filtrando clases no válidas en las etiquetas de validación...")
    filter_invalid_classes(val_labels_dir, allowed_classes)

----------------------------------------------------------------------


import os
import cv2

def check_images(image_dir):
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg'):
            img_path = os.path.join(image_dir, image_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Imagen corrupta o inexistente: {img_path}")

if __name__ == "__main__":
    train_images_dir = r"C:\Users\danie\Desktop\PRACTICA CITYLAB\Profesional\yolo-flow\data\images\train"
    val_images_dir = r"C:\Users\danie\Desktop\PRACTICA CITYLAB\Profesional\yolo-flow\data\images\val"

    print("[INFO] Verificando imágenes de entrenamiento...")
    check_images(train_images_dir)

    print("[INFO] Verificando imágenes de validación...")
    check_images(val_images_dir)
