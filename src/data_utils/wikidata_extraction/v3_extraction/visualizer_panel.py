"""
SCOR-DS & EO Triplet Visualizer — Panel version
------------------------------------------------
Run locally:
    pip install panel pyvis pandas
    panel serve visualizer_panel.py --show

Export to standalone HTML (runs in browser via WebAssembly):
    panel convert visualizer_panel.py --to html --out visualizer_panel.html
"""

import io
import base64
import panel as pn
import pandas as pd
from pyvis.network import Network

pn.extension(sizing_mode="stretch_width")

# ── Color maps (matching original Streamlit app) ──────────────────────────────
NODE_COLOR_MAP = {
    "business":      "#97c2fc",
    "enterprise":    "#97c2fc",
    "corporation":   "#97c2fc",
    "supplier":      "#ffcc00",
    "vendor":        "#ffcc00",
    "product":       "#fb7e81",
    "raw material":  "#fb7e81",
}
DEFAULT_NODE_COLOR = "#eeeeee"
TARGET_NODE_COLOR = "#dddddd"

# ── Widgets ───────────────────────────────────────────────────────────────────
file_input = pn.widgets.FileInput(
    accept=".csv",
    name="Drop or click to load CSV",
)

relation_selector = pn.widgets.CheckBoxGroup(
    name="SCOR-DS Relations",
    options=[],
    value=[],
)

limit_slider = pn.widgets.IntSlider(
    name="Number of Triplets",
    start=10,
    end=500,
    step=10,
    value=100,
)

status_text = pn.pane.Markdown(
    "_No file loaded yet. Drop a CSV above to begin._",
    styles={"color": "#888", "font-style": "italic"},
)

graph_pane = pn.pane.HTML(
    "<div style='display:flex;align-items:center;justify-content:center;"
    "height:600px;color:#888;font-family:monospace;font-size:13px;'>"
    "Graph will appear here after loading a CSV.</div>",
    height=650,
    sizing_mode="stretch_width",
)

# ── Internal state ────────────────────────────────────────────────────────────
_df = None   # holds the loaded dataframe


# ── Core rendering ────────────────────────────────────────────────────────────
def build_graph(df: pd.DataFrame, selected_relations: list, limit: int) -> str:
    """Return HTML string of vis.js graph for the given filters."""
    filtered = df[df["relation_label"].isin(selected_relations)].head(limit)

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        directed=True,
    )
    net.set_options("""
    {
      "edges": {
        "color": { "inherit": false, "color": "#4a6fa5" },
        "smooth": { "enabled": true, "type": "dynamic" },
        "font": { "size": 10, "color": "#aaaaaa" }
      },
      "physics": {
        "enabled": true,
        "stabilization": { "enabled": true, "iterations": 500, "fit": true },
        "barnesHut": { "gravitationalConstant": -3000, "springLength": 120 }
      },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }
    """)

    for _, row in filtered.iterrows():
        org_type = str(row.get("org_type", "")).lower()
        node_color = NODE_COLOR_MAP.get(org_type, DEFAULT_NODE_COLOR)

        net.add_node(
            row["org_id"],
            label=row["org_label"],
            title=str(row.get("org_type", "")),
            color=node_color,
        )
        net.add_node(
            row["target_id"],
            label=row["target_label"],
            color=TARGET_NODE_COLOR,
        )
        net.add_edge(
            row["org_id"],
            row["target_id"],
            label=row["relation_label"],
        )

    # Return raw HTML string (no file write needed)
    return net.generate_html()


def refresh_graph():
    """Re-render graph with current widget values."""
    global _df
    if _df is None:
        return

    selected = relation_selector.value
    if not selected:
        graph_pane.object = (
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:600px;color:#888;font-family:monospace;font-size:13px;'>"
            "Select at least one relation to display the graph.</div>"
        )
        return

    html = build_graph(_df, selected, limit_slider.value)
    graph_pane.object = html
    status_text.object = (
        f"✅ Showing **{min(limit_slider.value, len(_df[_df['relation_label'].isin(selected)]))}** "
        f"triplets across **{len(selected)}** relation(s) — "
        f"**{len(_df)}** total rows in file."
    )


# ── File upload callback ──────────────────────────────────────────────────────
def on_file_upload(event):
    global _df
    if file_input.value is None:
        return

    try:
        raw = file_input.value
        # Panel gives bytes; decode to StringIO
        content = io.StringIO(raw.decode("utf-8"))
        df = pd.read_csv(content)

        required = {"org_id", "org_label", "org_type",
                    "target_id", "target_label", "relation_label"}
        missing = required - set(df.columns)
        if missing:
            status_text.object = f"❌ CSV missing columns: `{', '.join(missing)}`"
            return

        _df = df
        relations = sorted(df["relation_label"].unique().tolist())
        relation_selector.options = relations
        relation_selector.value = relations[:2] if len(
            relations) >= 2 else relations

        refresh_graph()

    except Exception as e:
        status_text.object = f"❌ Error loading file: `{e}`"


file_input.param.watch(on_file_upload, "value")
relation_selector.param.watch(lambda e: refresh_graph(), "value")
limit_slider.param.watch(lambda e: refresh_graph(), "value")


# ── Layout ────────────────────────────────────────────────────────────────────
sidebar = pn.Column(
    pn.pane.Markdown("## ⬡ SCOR-DS Visualizer",
                     styles={"margin-bottom": "4px"}),
    pn.layout.Divider(),

    pn.pane.Markdown("**Load Data**"),
    file_input,
    status_text,

    pn.layout.Divider(),

    pn.pane.Markdown("**Filter Relations**"),
    relation_selector,

    pn.layout.Divider(),

    pn.pane.Markdown("**Triplet Limit**"),
    limit_slider,

    pn.layout.Divider(),

    pn.pane.Markdown(
        "💡 _Scroll to zoom · Drag nodes · Click to inspect_",
        styles={"color": "#888", "font-size": "12px"},
    ),

    width=280,
    styles={"padding": "16px", "background": "#f5f5f5"},
)

main = pn.Column(
    pn.pane.Markdown("# SCOR-DS & EO Triplet Visualizer"),
    graph_pane,
    sizing_mode="stretch_width",
    styles={"padding": "16px"},
)

layout = pn.Row(sidebar, main, sizing_mode="stretch_width")

# ── Serve / export entry point ────────────────────────────────────────────────
layout.servable()
