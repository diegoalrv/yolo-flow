import os
import cv2
from ultralytics import YOLO

def load_model():
    model = YOLO("yolov10m.pt")  # Ruta al modelo YOLO
    return model

def detect_objects_in_video(model, video_path, output_path):

    cap = cv2.VideoCapture(video_path)  # Abre el video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Realiza predicciones en el cuadro actual
        results = model.predict(frame, classes=[0, 2, 3, 5], conf=0.4)
        img_with_boxes = results[0].plot()  # Dibuja las cajas en el cuadro
        out.write(img_with_boxes)  # Guarda el cuadro procesado

    cap.release()
    out.release()
    print(f"Video procesado guardado en: {output_path}")

def main():
    model = load_model()

    # Configurar rutas del video de entrada y salida
    video_path = os.getenv('VIDEO_PATH', '/app/data/DJI_20241111152049_0053_D2.mp4')
    output_path = '/app/data/output/pruebaVisDrone.mp4'
    detect_objects_in_video(model, video_path, output_path)

if __name__ == "__main__":
    main()
