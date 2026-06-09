import pandas as pd
import streamlit as st
from datetime import datetime


def _ensure_state():
    if 'report_data' not in st.session_state:
        st.session_state.report_data = []


def add_report_entry(stats: dict, source_type: str):
    """Thêm một bản ghi vào lịch sử báo cáo."""
    _ensure_state()
    entry = {
        'Thời gian':        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Nguồn':            source_type,
        'Tổng đối tượng':   stats.get('total', 0),
        'Có mũ':            stats.get('helmet', 0),
        'Không mũ':         stats.get('no_helmet', 0),
        'Tỷ lệ an toàn (%)': round(stats.get('safety_rate', 0), 2),
        'FPS TB':           round(stats.get('fps', 0), 2) if 'fps' in stats else '-',
        'Số frame':         stats.get('frames', '-'),
    }
    st.session_state.report_data.append(entry)


def generate_report() -> pd.DataFrame:
    """Trả về DataFrame từ toàn bộ lịch sử."""
    _ensure_state()
    if not st.session_state.report_data:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.report_data)


def clear_report():
    """Xoá toàn bộ lịch sử."""
    st.session_state.report_data = []
