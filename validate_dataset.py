import os

def check_dataset(images_dir, labels_dir):
    images = {f.split('.')[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))}
    labels = {f.split('.')[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}

    missing_labels = images - labels
    if missing_labels:
        print(f"Imágenes sin etiquetas: {len(missing_labels)}")
        for img in missing_labels:
            print(f"  {img}.jpg")

    missing_images = labels - images
    if missing_images:
        print(f"Etiquetas sin imágenes: {len(missing_images)}")
        for lbl in missing_images:
            print(f"  {lbl}.txt")

    if not missing_labels and not missing_images:
        print("¡Dataset válido! Todas las imágenes tienen etiquetas correspondientes.")

if __name__ == "__main__":
    train_images_dir = "data/images/train"
    train_labels_dir = "data/labels/train"
    val_images_dir = "data/images/val"
    val_labels_dir = "data/labels/val"

    print("Verificando dataset de entrenamiento...")
    check_dataset(train_images_dir, train_labels_dir)

    print("\nVerificando dataset de validación...")
    check_dataset(val_images_dir, val_labels_dir)
