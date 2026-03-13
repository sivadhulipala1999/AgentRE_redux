import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import os

# 1. FORCE WIDE LAYOUT (Must be the first Streamlit command)
st.set_page_config(layout="wide", page_title="SCOR-DS Visualizer")


@st.cache_data
def load_data():
    # Ensure path is correct for your environment
    path = "D:/Projects/AgentRE_redux/src/data/Wikidata/wikidata_v3/stage1_organisations.csv"
    return pd.read_csv(path)


df = load_data()

st.title("SCOR-DS & EO Triplet Visualizer")

# 2. Sidebar Implementation
# Note: Moved the header inside the sidebar for better UI
with st.sidebar:
    st.header("Graph Filters")
    selected_relation = st.multiselect(
        "Select SCOR-DS Relations",
        options=df['relation_label'].unique().tolist(),
        default=df['relation_label'].unique()[:2].tolist()
    )
    limit = st.slider("Number of Triplets to Show", 10, 500, 100)

    st.divider()
    st.info("💡 Use the mouse wheel to zoom. Click and drag nodes to rearrange.")

# Filter Data
filtered_df = df[df['relation_label'].isin(selected_relation)].head(limit)

# 3. Create Pyvis Network
# We set width to something slightly less than 100% or use CSS to ensure it stays in bounds
net = Network(height="600px", width="100%", bgcolor="#ffffff",
              font_color="black", directed=True)

color_map = {
    "business": "#97c2fc",
    "enterprise": "#97c2fc",
    "corporation": "#97c2fc",
    "supplier": "#ffcc00",
    "vendor": "#ffcc00",
    "product": "#fb7e81",
    "raw material": "#fb7e81"
}

for _, row in filtered_df.iterrows():
    net.add_node(row['org_id'], label=row['org_label'], title=row['org_type'],
                 color=color_map.get(row['org_type'].lower(), "#eeeeee"))
    net.add_node(row['target_id'], label=row['target_label'], color="#dddddd")
    net.add_edge(row['org_id'], row['target_id'],
                 label=row['relation_label'], weight=1)

# 4. Improved Save and Display logic
# Using a temporary file path that is less likely to conflict
path = "tmp_visuals"
if not os.path.exists(path):
    os.makedirs(path)

net.save_graph(f"{path}/temp_graph.html")

with open(f"{path}/temp_graph.html", 'r', encoding='utf-8') as f:
    source_code = f.read()

# Using scrolling=False but a higher height often looks cleaner
components.html(source_code, height=750, scrolling=True)
