import os
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO


def find_latest_model():
    fixed_path = 'uploads/runs/detect/fault/weights/best.pt'
    if os.path.exists(fixed_path):
        return fixed_path
    return None


def run_prediction(image_path):
    weights_path = find_latest_model()
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError("No trained model found. Expected at uploads/runs/detect/fault/weights/best.pt")

    model = YOLO(weights_path)
    
    results = model.predict(
        source=image_path,
        conf=0.5,
        save=False,
        verbose=False
    )
    
    if not results or len(results) == 0:
        raise ValueError("No results from prediction.")
    
    r = results[0]
    detections = []
    
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = r.names.get(cls_id, f'class_{cls_id}')
            conf = float(box.conf[0]) * 100
            bbox = box.xyxy[0].tolist()
            
            detections.append({
                'class': class_name,
                'confidence': round(conf, 2),
                'bbox': [round(x, 2) for x in bbox]
            })
    
    # Annotated image
    annotated_img = r.plot()
    annotated_img = annotated_img[..., ::-1]   # BGR → RGB
    annotated_pil = Image.fromarray(annotated_img)
    
    buffered = BytesIO()
    annotated_pil.save(buffered, format="JPEG", quality=90)
    detected_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Original image
    with open(image_path, 'rb') as f:
        original_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    return {
        'original': original_base64,
        'detected': detected_base64,
        'predictions': detections,
        'detections': detections,
        'num_detections': len(detections)
    }