"""
Helmet Detection System — main entry point
Chạy: streamlit run main.py
"""

import os
import tempfile
from datetime import datetime

import numpy as np
import streamlit as st
import torch

# ── Internal modules (không định nghĩa lại ở đây) ──────────────────────────
from load_model import load_model
from PIL import Image
from processing import process_image, process_video, stream_webcam
from report import clear_report, generate_report

# ======================== CẤU HÌNH TRANG ========================
st.set_page_config(
    page_title="🚨 Helmet Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== LOAD MODEL ========================
model = load_model()

# ======================== SIDEBAR ========================
with st.sidebar:
    st.title("⚙️ Cài đặt")
    st.markdown("---")

    st.markdown("### 🔧 Thông số mô hình")
    conf_thresh = st.slider("Ngưỡng tin cậy (Confidence)", 0.1, 1.0, 0.50, 0.05)
    iou_thresh = st.slider("Ngưỡng IoU", 0.1, 1.0, 0.40, 0.05)

    st.markdown("---")
    st.markdown("### ⚠️ Ngưỡng cảnh báo (Webcam)")
    alert_thresh = st.slider(
        "Cảnh báo khi tỷ lệ không mũ ≥ (%)",
        10,
        100,
        50,
        5,
        help="Chỉ áp dụng cho chế độ webcam real-time",
    )
    device = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🔵 CPU"
st.sidebar.markdown(f"**Thiết bị:** {device}")
if torch.cuda.is_available():
    st.sidebar.caption(torch.cuda.get_device_name(0))
    st.markdown("---")
    st.markdown("""
    ### ℹ️ Chú thích
    - 🟢 **Xanh**: Có đội mũ bảo hiểm
    - 🔴 **Đỏ**: Không đội mũ bảo hiểm
    """)

# ======================== TIÊU ĐỀ ========================
st.markdown(
    """
    <h2 style="text-align:center;">
        🛡️ Ứng dụng nhận diện không đội mũ bảo hiểm
    </h2>
    <p style="text-align:center; color:gray;">
        Chọn nguồn dữ liệu để bắt đầu nhận diện
    </p>
    """,
    unsafe_allow_html=True,
)

# ======================== CHỌN NGUỒN ========================
source = st.radio(
    "Chọn nguồn dữ liệu:",
    ["📸 Hình ảnh", "🎥 Video", "📷 Webcam real-time"],
    horizontal=True,
)

# ──────────────────────────────────────────────────────────────
# NGUỒN: ẢNH
# ──────────────────────────────────────────────────────────────
if source == "📸 Hình ảnh":
    file = st.file_uploader(
        "Tải ảnh lên",
        type=["jpg", "jpeg", "png"],
        help="Ảnh chứa người đội/không đội mũ bảo hiểm",
    )
    if file:
        image = Image.open(file).convert("RGB")
        with st.expander("📤 Kết quả nhận diện", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Ảnh gốc", use_container_width=True)
            with col2:
                result, stats = process_image(
                    np.array(image), model, conf_thresh, iou_thresh
                )
                st.image(result, caption="Kết quả phát hiện", use_container_width=True)

        st.subheader("📊 Thống kê")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🧍 Tổng đối tượng", stats["total"])
        c2.metric("🟢 Có mũ", stats["helmet"])
        c3.metric("🔴 Không mũ", stats["no_helmet"])
        safety = stats.get("safety_rate", 0.0)
        c4.metric("🔒 Tỷ lệ an toàn", f"{safety:.1f}%")

        # Cảnh báo trực tiếp trên ảnh
        if stats["total"] > 0 and stats["no_helmet"] > 0:
            no_helmet_pct = stats["no_helmet"] / stats["total"] * 100
            if no_helmet_pct >= alert_thresh:
                st.error(
                    f"⚠️ **CẢNH BÁO:** {no_helmet_pct:.0f}% người không đội mũ bảo hiểm!"
                )

# ──────────────────────────────────────────────────────────────
# NGUỒN: VIDEO
# ──────────────────────────────────────────────────────────────
elif source == "🎥 Video":
    file = st.file_uploader(
        "Tải video lên",
        type=["mp4", "mov", "avi"],
        help="Video để phân tích theo thời gian thực",
    )
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            process_video(tmp_path, model, conf_thresh, iou_thresh)
        finally:
            try:
                os.remove(tmp_path)
            except OSError as e:
                st.warning(f"Không xóa được file tạm: {e}")

# ──────────────────────────────────────────────────────────────
# NGUỒN: WEBCAM
# ──────────────────────────────────────────────────────────────
elif source == "📷 Webcam real-time":
    st.info(
        "📷 Nhấn **Bắt đầu** để khởi động webcam. Cần cho phép trình duyệt/ứng dụng truy cập camera."
    )
    if st.button("▶️ Bắt đầu stream", type="primary", key="start_webcam"):
        stream_webcam(model, conf_thresh, iou_thresh, alert_threshold=alert_thresh)

# ======================== LỊCH SỬ BÁO CÁO ========================
report_df = generate_report()
if not report_df.empty:
    st.markdown("---")
    st.subheader("📋 Lịch sử thống kê")
    st.dataframe(report_df, use_container_width=True)

    col_dl, col_clear, _ = st.columns([1, 1, 4])
    with col_dl:
        csv = report_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "💾 Tải CSV",
            data=csv,
            file_name=f"helmet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with col_clear:
        if st.button("🗑️ Xóa lịch sử", type="secondary"):
            clear_report()
            st.rerun()

# ======================== FOOTER ========================
st.markdown(
    """
    <div style="text-align:center; color:#888; font-size:1.1em; margin-top:40px;">
        © Nhóm 11 – BÁO CÁO BÀI TẬP LỚN – 2026
    </div>
    """,
    unsafe_allow_html=True,
)
