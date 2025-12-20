"""Shared UI styling helpers for Streamlit pages."""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


def inject_global_css() -> None:
    """Inject global CSS for a consistent dashboard design system."""

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.75rem;
            padding-bottom: 3.5rem;
            padding-left: 3.5rem;
            padding-right: 3.5rem;
        }
        .hrp-title {
            font-size: 2.25rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .hrp-subtitle {
            font-size: 1rem;
            color: #6B7280;
            margin-bottom: 2rem;
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 2rem 0 0.5rem 0;
        }
        .section-subtitle {
            font-size: 0.95rem;
            color: #6B7280;
            margin-bottom: 1.25rem;
        }
        .section-spacer {
            height: 1.5rem;
        }
        .hrp-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 1.5rem 1.5rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.5rem;
        }
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .card-subtitle {
            font-size: 0.9rem;
            color: #6B7280;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def card(title: str | None = None, subtitle: str | None = None):
    """Render a styled card container."""

    st.markdown('<div class="hrp-card">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="card-subtitle">{subtitle}</div>', unsafe_allow_html=True
        )
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)
