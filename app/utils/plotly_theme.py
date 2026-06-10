from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


PLOTLY_TEXT_COLOR = "#000000"
PLOTLY_GRID_COLOR = "#d9d9d9"


def apply_presentation_text(fig: go.Figure) -> go.Figure:
    """Force high-contrast text on a Plotly figure before rendering."""
    fig.update_layout(
        font=dict(color=PLOTLY_TEXT_COLOR, size=15),
        title=dict(font=dict(color=PLOTLY_TEXT_COLOR, size=20)),
        legend=dict(
            title=dict(font=dict(color=PLOTLY_TEXT_COLOR, size=14)),
            font=dict(color=PLOTLY_TEXT_COLOR, size=13),
        ),
    )
    for axis in fig.layout.to_plotly_json():
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            ax = fig.layout[axis]
            # Preserve tick font size if already set larger than default
            cur_tick_size = getattr(ax.tickfont, "size", None) if ax.tickfont else None
            tick_size = cur_tick_size if cur_tick_size and cur_tick_size > 13 else 13
            # Preserve title font size if already set larger than default
            title_font = getattr(ax.title, "font", None) if ax.title else None
            cur_title_size = getattr(title_font, "size", None) if title_font else None
            title_size = cur_title_size if cur_title_size and cur_title_size > 16 else 16
            fig.layout[axis].update(
                title_font=dict(color=PLOTLY_TEXT_COLOR, size=title_size),
                tickfont=dict(color=PLOTLY_TEXT_COLOR, size=tick_size),
                gridcolor=PLOTLY_GRID_COLOR,
                zerolinecolor=PLOTLY_GRID_COLOR,
            )
    fig.update_traces(textfont=dict(color=PLOTLY_TEXT_COLOR), selector=dict(type="bar"))
    fig.update_traces(textfont=dict(color=PLOTLY_TEXT_COLOR), selector=dict(type="histogram"))
    fig.update_traces(textfont=dict(color=PLOTLY_TEXT_COLOR), selector=dict(type="pie"))
    fig.update_traces(textfont=dict(color=PLOTLY_TEXT_COLOR), selector=dict(type="scatter"))
    return fig


def _patch_streamlit_plotly_chart() -> None:
    if getattr(st, "_uv_plotly_theme_patched", False):
        return

    original_plotly_chart = st.plotly_chart

    def plotly_chart_with_presentation_text(fig_or_data, *args, **kwargs):
        if isinstance(fig_or_data, go.Figure):
            apply_presentation_text(fig_or_data)
        return original_plotly_chart(fig_or_data, *args, **kwargs)

    st.plotly_chart = plotly_chart_with_presentation_text
    st._uv_plotly_theme_patched = True


def configure_plotly_theme() -> None:
    """Set the default Plotly theme for presentation-friendly contrast."""
    pio.templates["uv_presentation"] = go.layout.Template(
        layout=go.Layout(
            font=dict(color=PLOTLY_TEXT_COLOR, size=15),
            title=dict(font=dict(color=PLOTLY_TEXT_COLOR, size=20)),
            xaxis=dict(
                title=dict(font=dict(color=PLOTLY_TEXT_COLOR, size=16)),
                tickfont=dict(color=PLOTLY_TEXT_COLOR, size=13),
                gridcolor=PLOTLY_GRID_COLOR,
                zerolinecolor=PLOTLY_GRID_COLOR,
            ),
            yaxis=dict(
                title=dict(font=dict(color=PLOTLY_TEXT_COLOR, size=16)),
                tickfont=dict(color=PLOTLY_TEXT_COLOR, size=13),
                gridcolor=PLOTLY_GRID_COLOR,
                zerolinecolor=PLOTLY_GRID_COLOR,
            ),
            legend=dict(
                title=dict(font=dict(color=PLOTLY_TEXT_COLOR, size=14)),
                font=dict(color=PLOTLY_TEXT_COLOR, size=13),
            ),
            coloraxis=dict(
                colorbar=dict(
                    title=dict(font=dict(color=PLOTLY_TEXT_COLOR, size=14)),
                    tickfont=dict(color=PLOTLY_TEXT_COLOR, size=13),
                )
            ),
        )
    )
    pio.templates.default = "plotly_white+uv_presentation"
    _patch_streamlit_plotly_chart()
