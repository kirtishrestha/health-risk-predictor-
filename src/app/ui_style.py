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
            padding-top: 3rem;
            padding-bottom: 4rem;
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
            margin-bottom: 2.5rem;
        }
        .section-title {
            font-size: 1.6rem;
            font-weight: 600;
            margin: 2.5rem 0 0.5rem 0;
        }
        .section-subtitle {
            font-size: 1rem;
            color: #6B7280;
            margin-bottom: 1.5rem;
        }
        .section-spacer {
            height: 1.5rem;
        }
        .hrp-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 1.75rem 1.75rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.75rem;
        }
        .hrp-card.kpi-card {
            padding: 1.75rem 1.5rem;
            min-height: 140px;
        }
        .hrp-card.filter-card {
            padding: 1.25rem 1.5rem 0.5rem;
        }
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .card-subtitle {
            font-size: 0.95rem;
            color: #6B7280;
            margin-bottom: 1rem;
        }
        .kpi-value {
            font-size: 2.1rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.2;
        }
        .kpi-label {
            font-size: 0.95rem;
            color: #6B7280;
            margin-top: 0.35rem;
        }
        .kpi-context {
            font-size: 0.85rem;
            color: #9CA3AF;
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def card(
    title: str | None = None,
    subtitle: str | None = None,
    *,
    class_name: str | None = None,
):
    """Render a styled card container."""

    extra_class = f" {class_name}" if class_name else ""
    st.markdown(f'<div class="hrp-card{extra_class}">', unsafe_allow_html=True)
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
