import os
import streamlit as st
from ultralytics import YOLO
import torch

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    MODEL_PATH = os.path.join(_BASE_DIR, "weights", "bestyolo.pt")
else:
    MODEL_PATH = os.path.join(_BASE_DIR, "weights", "bestyolo.onnx")
@st.cache_resource
def load_model(model_path: str = MODEL_PATH) -> YOLO:
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy model tại: {model_path}")
        st.stop()
    with st.spinner("🚀 Đang tải mô hình YOLO..."):
        # Khai báo rõ task=detect để tránh WARNING
        return YOLO(model_path, task="detect")
