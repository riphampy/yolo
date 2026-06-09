import os
import streamlit as st
from ultralytics import YOLO

# Đường dẫn tuyệt đối: app/ -> lên 1 cấp -> weights/bestyolo.onnx
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE_DIR, "weights", "bestyolo.onnx")


@st.cache_resource
def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """
    Load YOLO model và cache lại — chỉ load 1 lần duy nhất.
    """
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy model tại: {model_path}")
        st.stop()
    with st.spinner("🚀 Đang tải mô hình YOLO..."):
        return YOLO(model_path)
