import cv2
import numpy as np
import streamlit as st
import time
from datetime import datetime

from draw_box import draw_boxes
from report import add_report_entry


# ======================== XỬ LÝ ẢNH ========================

def process_image(image: np.ndarray, model, conf_thresh: float, iou_thresh: float) -> tuple[np.ndarray, dict]:
    """
    Chạy inference trên một ảnh RGB, trả về ảnh đã annotate và stats.
    """
    with st.spinner("🔍 Đang nhận diện..."):
        results = model(image, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated, stats = draw_boxes(image_bgr, results)

    total = stats['total']
    safety_rate = (stats['helmet'] / total * 100) if total > 0 else 0.0
    stats['safety_rate'] = safety_rate

    add_report_entry({**stats, 'safety_rate': safety_rate}, 'Ảnh')
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), stats


# ======================== XỬ LÝ VIDEO ========================

def process_video(video_path: str, model, conf_thresh: float, iou_thresh: float,
                  skip_frames: int = 3) -> dict | None:
    """
    Xử lý video từ file, hiển thị real-time trên Streamlit.
    Trả về dict thống kê hoặc None nếu không mở được file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Không mở được video. Vui lòng kiểm tra file.")
        return None

    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    last_display_time = 0.0
    annotated_frame = None

    stats = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'helmet_counts': [],
        'no_helmet_counts': [],
        'fps_list': [],
        'start_time': datetime.now(),
    }

    status_text.info(f"Đang xử lý video ({total_frames} frames)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        resized = cv2.resize(frame, (640, 360))

        if frame_count % skip_frames == 0:
            t0 = time.time()
            results = model(resized, verbose=False, conf=conf_thresh, iou=iou_thresh)[0]
            actual_fps = 1.0 / max(time.time() - t0, 1e-6)

            annotated_frame, frame_stats = draw_boxes(resized.copy(), results, actual_fps=actual_fps)

            stats['helmet_counts'].append(frame_stats['helmet'])
            stats['no_helmet_counts'].append(frame_stats['no_helmet'])
            stats['fps_list'].append(actual_fps)
            stats['processed_frames'] += 1
        else:
            if annotated_frame is None:
                annotated_frame = resized  # fallback trước khi có frame đầu tiên

        # Hiển thị ~30 fps để mượt
        if time.time() - last_display_time > 0.033:
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.info(f"Đang xử lý... {progress * 100:.1f}%")
            last_display_time = time.time()

    cap.release()
    stats['processing_time'] = datetime.now() - stats['start_time']
    status_text.success(f"✅ Hoàn tất! Thời gian xử lý: {stats['processing_time'].seconds}s")

    # Tổng hợp
    total_helmet = sum(stats['helmet_counts'])
    total_no_helmet = sum(stats['no_helmet_counts'])
    total_objects = total_helmet + total_no_helmet
    avg_fps = float(np.mean(stats['fps_list'])) if stats['fps_list'] else 0.0
    safety_rate = (total_helmet / total_objects * 100) if total_objects > 0 else 0.0

    _render_video_stats(total_objects, total_helmet, total_no_helmet,
                        safety_rate, stats['processed_frames'], avg_fps)

    add_report_entry({
        'total': total_objects,
        'helmet': total_helmet,
        'no_helmet': total_no_helmet,
        'safety_rate': safety_rate,
        'fps': avg_fps,
        'frames': stats['processed_frames'],
    }, 'Video')

    return stats


def _render_video_stats(total, helmet, no_helmet, safety_rate, frames, fps):
    st.markdown("### 📊 Thống kê video")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🧍 Tổng",          str(total))
    c2.metric("🟢 Có mũ",         str(helmet))
    c3.metric("🔴 Không mũ",      str(no_helmet))
    c4.metric("🔒 An toàn",       f"{safety_rate:.1f}%")
    c5.metric("🎞️ Frames",        str(frames))
    c6.metric("⚡ FPS TB",         f"{fps:.1f}")


# ======================== WEBCAM REAL-TIME ========================

def stream_webcam(model, conf_thresh: float, iou_thresh: float,
                  alert_threshold: float = 50.0):
    """
    Stream webcam real-time với YOLO detection.
    Hiển thị cảnh báo khi tỷ lệ không đội mũ vượt alert_threshold (%).
    Dừng khi người dùng nhấn nút Stop.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Không tìm thấy webcam. Hãy kiểm tra kết nối thiết bị.")
        return

    # Cấu hình resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    stframe = st.empty()
    alert_box = st.empty()
    col1, col2, col3 = st.columns(3)
    metric_helmet    = col1.empty()
    metric_no_helmet = col2.empty()
    metric_fps       = col3.empty()

    stop_btn = st.button("⏹️ Dừng webcam", type="primary", key="stop_webcam")

    fps_buffer = []
    frame_idx = 0

    while not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Không đọc được frame từ webcam.")
            break

        frame_idx += 1

        t0 = time.time()
        results = model(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        actual_fps = 1.0 / max(time.time() - t0, 1e-6)

        annotated, stats = draw_boxes(frame.copy(), results, actual_fps=actual_fps)

        # FPS rolling average (30 frame)
        fps_buffer.append(actual_fps)
        if len(fps_buffer) > 30:
            fps_buffer.pop(0)
        smooth_fps = float(np.mean(fps_buffer))

        # Hiển thị frame
        stframe.image(annotated, channels="BGR", use_container_width=True)

        # Cập nhật metrics
        total = stats['total']
        no_helmet = stats['no_helmet']
        helmet = stats['helmet']
        no_helmet_rate = (no_helmet / total * 100) if total > 0 else 0.0

        metric_helmet.metric("🟢 Có mũ", helmet)
        metric_no_helmet.metric("🔴 Không mũ", no_helmet)
        metric_fps.metric("⚡ FPS", f"{smooth_fps:.1f}")

        # ⚠️ Cảnh báo tỷ lệ không đội mũ cao
        if total > 0 and no_helmet_rate >= alert_threshold:
            alert_box.error(
                f"⚠️ **CẢNH BÁO:** {no_helmet_rate:.0f}% người không đội mũ bảo hiểm! "
                f"({no_helmet}/{total} người)"
            )
        else:
            alert_box.empty()

    cap.release()
    stframe.empty()
    st.info("📷 Webcam đã dừng.")
