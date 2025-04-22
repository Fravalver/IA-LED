from ultralytics import YOLO

# Cargar modelo preentrenado base (YOLOv8 nano o small)
model = YOLO("yolov8n.pt")  # Puedes cambiar por 'yolov8s.pt' si tienes buena GPU

# Entrenamiento
model.train(
    data="led_config.yaml",   # Archivo de configuración
    epochs=50,                # Número de épocas
    imgsz=640,                # Tamaño de imagen (por defecto 640)
    batch=8,                  # Tamaño del batch
    name="modelo_led",        # Nombre del experimento
    project="runs_led"        # Carpeta donde se guardarán los resultados
)
