import os
import cv2

def process_annotations(input_dir, output_dir, img_dir):
    os.makedirs(output_dir, exist_ok=True)

    for annotation_file in os.listdir(input_dir):
        if annotation_file.endswith('.txt'):
            input_path = os.path.join(input_dir, annotation_file)
            output_path = os.path.join(output_dir, annotation_file)

            print(f"\n[INFO] Procesando archivo: {input_path}")

            # Verificar que el archivo tiene contenido
            if os.path.getsize(input_path) == 0:
                print(f"[WARNING] Archivo vacío detectado: {input_path}")
                continue

            with open(input_path, 'r') as f_in:
                content = f_in.readlines()

                if not content:
                    print(f"[WARNING] Archivo sin líneas válidas: {input_path}")
                    continue

            # Leer la imagen correspondiente
            img_name = annotation_file.replace('.txt', '.jpg')
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                print(f"[ERROR] Imagen no encontrada: {img_path}")
                continue

            img = cv2.imread(img_path)

            if img is None:
                print(f"[ERROR] No se pudo leer la imagen: {img_path}")
                continue

            print(f"[INFO] Dimensiones de la imagen: {img.shape}")

            # Verificar datos y realizar conversión
            try:
                with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:  # Crear archivo en carpeta de salida
                    for line in f_in:
                        fields = line.strip().split(',')
                        if len(fields) < 8:
                            print(f"[ERROR] Línea corrupta en {input_path}: {line}")
                            continue

                        class_id = fields[0]
                        xmin = float(fields[1])
                        ymin = float(fields[2])
                        width = float(fields[3])
                        height = float(fields[4])

                        # Ignorar anotaciones marcadas como "ignored"
                        if int(fields[6]) == 1:
                            continue

                        # Convertir a formato YOLO
                        img_width, img_height = img.shape[1], img.shape[0]
                        x_center = (xmin + width / 2) / img_width
                        y_center = (ymin + height / 2) / img_height
                        norm_width = width / img_width
                        norm_height = height / img_height

                        if any(v < 0 or v > 1 for v in [x_center, y_center, norm_width, norm_height]):
                            print(f"[WARNING] Valores fuera de rango en {input_path}: {line}")
                            continue

                        f_out.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

                print(f"[INFO] Archivo convertido correctamente: {output_path}")
            except Exception as e:
                print(f"[ERROR] Error durante la conversión de {input_path}: {e}")

if __name__ == "__main__":
    # Rutas ajustadas
    train_labels_dir = r"C:\Users\danie\Desktop\PRACTICA CITYLAB\Profesional\yolo-flow\data\labels\train"
    train_images_dir = r"C:\Users\danie\Desktop\PRACTICA CITYLAB\Profesional\yolo-flow\data\images\train"
    train_output_dir = r"C:\Users\danie\Desktop\PRACTICA CITYLAB\Profesional\yolo-flow\data\labels\train_converted"

    val_labels_dir = r"C:\Users\danie\Desktop\PRACTICA CITYLAB\Profesional\yolo-flow\data\labels\val"
    val_images_dir = r"C:\Users\danie\Desktop\PRACTICA CITYLAB\Profesional\yolo-flow\data\images\val"
    val_output_dir = r"C:\Users\danie\Desktop\PRACTICA CITYLAB\Profesional\yolo-flow\data\labels\val_converted"

    print("[INFO] Procesando etiquetas de entrenamiento...")
    process_annotations(train_labels_dir, train_output_dir, train_images_dir)

    print("\n[INFO] Procesando etiquetas de validación...")
    process_annotations(val_labels_dir, val_output_dir, val_images_dir)
