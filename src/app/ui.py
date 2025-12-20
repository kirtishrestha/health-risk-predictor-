"""UI helpers for consistent Streamlit layout."""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


_THEME_KEY = "_ui_theme_set"


def set_page_theme() -> None:
    """Inject shared CSS once per session."""

    if st.session_state.get(_THEME_KEY):
        return

    st.markdown(
        """
<style>
:root {
  --app-card-bg: rgba(250, 250, 250, 0.04);
}

div[data-testid="stContainer"][data-border="true"] {
  border-radius: 14px;
  background-color: var(--app-card-bg);
  padding: 1rem 1.25rem;
}

div[data-testid="stContainer"][data-border="true"] > div {
  gap: 0.75rem;
}

.section-title {
  margin-bottom: 0.25rem;
}
</style>
""",
        unsafe_allow_html=True,
    )
    st.session_state[_THEME_KEY] = True


def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a consistent section header."""

    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


@contextmanager
def card(title: str | None = None, subtitle: str | None = None):
    """Render a bordered card container for grouped content."""

    container = st.container(border=True)
    with container:
        if title:
            st.subheader(title)
        if subtitle:
            st.caption(subtitle)
        yield container


def empty_state(message: str, icon: str | None = None) -> None:
    """Render a consistent empty-state message."""

    label = f"{icon} {message}" if icon else message
    st.info(label)
