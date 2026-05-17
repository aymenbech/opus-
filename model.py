import os
import json
import torch
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path


def train_fault_detector(
    dataset_yaml_path="./uploads/dataset/data.yaml",
    output_dir="./uploads/runs/detect/fault_detector",
    epochs=20,
    batch_size=16,
    img_size=640,
    model_size="yolo11s.pt",
    patience=20,
    name="faultguard_yolo11"
):
    dataset_yaml_path = Path(dataset_yaml_path)
    output_dir = Path(output_dir)
   
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml_path}")

    print("🚀 Starting FaultGuard YOLO11 Model Training...")
    print(f"   Model      : {model_size}")
    print(f"   Epochs     : {epochs}")
    print(f"   Batch Size : {batch_size}")
    print(f"   Image Size : {img_size}")
    print(f"   Device     : {'GPU' if torch.cuda.is_available() else 'CPU'}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_size)

    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=name,
        project=str(output_dir),
        patience=patience,
        save_period=10,
        workers=0 if os.name == "nt" else 4,
        device=0 if torch.cuda.is_available() else "cpu",
        pretrained=True,
        optimizer="auto",
        amp=True,
        augment=True,
        val=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    metrics_summary = {
        "model_name": "FaultGuard-YOLO11",
        "version": "v1.0",
        "yolo_version": "YOLO11",
        "model_size_used": model_size,
        "best_mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)) * 100,
        "best_mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)) * 100,
        "best_precision": float(results.results_dict.get("metrics/precision(B)", 0)) * 100,
        "best_recall": float(results.results_dict.get("metrics/recall(B)", 0)) * 100,
        "best_epoch": int(getattr(results, "best_epoch", 0)),
        "training_time": str(datetime.now() - results.start_time).split('.')[0] if hasattr(results, 'start_time') else "N/A",
        "img_size": img_size,
        "epochs_completed": epochs,
        "device": "GPU" if torch.cuda.is_available() else "CPU"
    }

    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)

    print("\n✅ Training completed successfully!")
    print(f"   Best mAP50    : {metrics_summary['best_mAP50']:.2f}%")
    print(f"   Best mAP50-95 : {metrics_summary['best_mAP50_95']:.2f}%")
   
    return model, metrics_summary


def export_model(model, output_dir):
    export_path = Path(output_dir) / "export"
    export_path.mkdir(parents=True, exist_ok=True)
   
    print("📦 Exporting YOLO11 model...")
    
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    model.export(format="torchscript")
    
    print(f"✅ Model exported successfully to: {export_path}")


if __name__ == "__main__":
    model, metrics = train_fault_detector(
        epochs=50,
        batch_size=16,
        img_size=640,
        model_size="yolo11s.pt"
    )