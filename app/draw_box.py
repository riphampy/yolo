import cv2
import numpy as np
import streamlit as st

# ======================== CLASS COLOR MAP ========================
# Dễ mở rộng: thêm class mới chỉ cần thêm vào dict này
CLASS_COLORS = {
    'helmet':    (0, 255, 0),   # Xanh lá
    'no_helmet': (0, 0, 255),   # Đỏ
}
DEFAULT_COLOR = (255, 165, 0)   # Cam — cho class không xác định


def get_class_color(label: str) -> tuple:
    """Trả về màu BGR cho class label, fallback về DEFAULT_COLOR."""
    return CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)


def load_css(file_path: str = "style.css"):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def draw_boxes(image: np.ndarray, results, actual_fps: float = None,
               font_scale_base: float = 0.5) -> tuple[np.ndarray, dict]:
    """
    Vẽ bounding box lên ảnh và trả về stats.

    Args:
        image:           Frame BGR (numpy array)
        results:         Kết quả từ YOLO model
        actual_fps:      FPS thực tế (None = không hiển thị overlay)
        font_scale_base: Scale cơ bản cho font chữ

    Returns:
        (annotated_image, stats_dict)
    """
    class_names = results.names
    boxes = results.boxes
    stats = {
        'total': 0,
        'helmet': 0,
        'no_helmet': 0,
        'confidences': [],
    }

    frame_height, frame_width = image.shape[:2]
    font_scale = font_scale_base * (frame_width / 640)
    thickness = max(1, int(frame_width / 640 * 2))

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]
        color = get_class_color(label)

        # Vẽ bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Nền chữ để dễ đọc
        text = f"{label} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(image, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Cập nhật stats
        stats['total'] += 1
        stats['helmet'] += int(label == 'helmet')
        stats['no_helmet'] += int(label == 'no_helmet')
        stats['confidences'].append(conf)

    # Overlay thống kê góc trên trái (chỉ khi có actual_fps)
    if actual_fps is not None:
        overlay_w, overlay_h = 190, 95
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (overlay_w, overlay_h), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.65, image, 0.35, 0)

        ofs = font_scale * 1.0
        oth = max(1, int(thickness * 0.8))

        cv2.putText(image, f"Helmet:    {stats['helmet']}",
                    (10, 28), cv2.FONT_HERSHEY_DUPLEX, ofs, (0, 255, 0), oth)
        cv2.putText(image, f"No Helmet: {stats['no_helmet']}",
                    (10, 58), cv2.FONT_HERSHEY_DUPLEX, ofs, (0, 0, 255), oth)
        # ✅ FPS THỰC TẾ — không nhân hệ số giả
        cv2.putText(image, f"FPS: {actual_fps:.1f}",
                    (10, 88), cv2.FONT_HERSHEY_DUPLEX, ofs, (0, 255, 255), oth)

    return image, stats
