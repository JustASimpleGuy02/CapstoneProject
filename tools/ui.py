from typing import Optional
import streamlit as st
from streamlit.delta_generator import DeltaGenerator


def display_centered_text(text: str, container: Optional[DeltaGenerator] = None):
    if container is not None:
        container.markdown(
            f"<div style='text-align: center;'>{text}</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='text-align: center;'>{text}</div>",
            unsafe_allow_html=True)
