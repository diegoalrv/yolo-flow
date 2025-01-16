import os
import cv2

def convert_visdrone_to_yolo(input_dir, output_dir, img_dir):
    os.makedirs(output_dir, exist_ok=True)
    for annotation_file in os.listdir(input_dir):
        if annotation_file.endswith('.txt'):
            input_path = os.path.join(input_dir, annotation_file)
            output_path = os.path.join(output_dir, annotation_file)

            with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
                for line in f_in:
                    fields = line.strip().split(',')
                    
                    # Verificar que la línea tenga al menos 7 campos
                    if len(fields) < 7:
                        print(f"Formato incorrecto en {annotation_file}: {line.strip()}")
                        continue

                    class_id = fields[0]
                    xmin = float(fields[1])
                    ymin = float(fields[2])
                    width = float(fields[3])
                    height = float(fields[4])

                    # Ignorar anotaciones marcadas como "ignored"
                    if int(fields[6]) == 1:
                        print(f"Línea ignorada en {annotation_file}: {line.strip()}")
                        continue

                    # Obtener dimensiones de la imagen
                    img_name = annotation_file.replace('.txt', '.jpg')
                    img_path = os.path.join(img_dir, img_name)

                    if not os.path.exists(img_path):
                        print(f"Imagen no encontrada: {img_path}")
                        continue

                    try:
                        img_width, img_height = get_image_dimensions(img_path)
                    except ValueError as e:
                        print(e)
                        continue

                    # Convertir a formato YOLO
                    x_center = (xmin + width / 2) / img_width
                    y_center = (ymin + height / 2) / img_height
                    width /= img_width
                    height /= img_height

                    f_out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

            print(f"Archivo convertido: {output_path}")

def get_image_dimensions(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")
    return img.shape[1], img.shape[0]

# Rutas locales
convert_visdrone_to_yolo(
    input_dir='data/labels/train', 
    output_dir='data/labels/train', 
    img_dir='data/images/train'
)

convert_visdrone_to_yolo(
    input_dir='data/labels/val', 
    output_dir='data/labels/val', 
    img_dir='data/images/val'
)