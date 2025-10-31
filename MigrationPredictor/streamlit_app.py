import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Portable path logic (env overrides -> repo data folders -> Downloads)
BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
# Candidate repo locations to check for data files
REPO_CLEANED = BASE_DIR / 'cleaned'
REPO_WB = BASE_DIR / 'world_bank_data'

def find_data_file(env_name: str, candidates: list[str]):
    # 1) environment variable
    val = os.environ.get(env_name)
    if val:
        p = Path(val)
        if p.exists():
            return p
    # 2) repo cleaned/ and world_bank_data
    for d in candidates:
        p = Path(d)
        if p.exists():
            return p
    # 3) Downloads fallback
    return Path(os.environ.get(env_name, DOWNLOADS / candidates[0].name))

# default candidate filenames (Path objects for name access)
_bilat_candidates = [REPO_CLEANED / 'bilat_mig.csv', REPO_WB / 'bilat_mig.csv']
_countries_candidates = [REPO_CLEANED / 'countries.csv', REPO_WB / 'countries.csv']

BILAT_FILE = find_data_file('BILAT_FILE', _bilat_candidates)
COUNTRIES_FILE = find_data_file('COUNTRIES_FILE', _countries_candidates)

st.set_page_config(page_title="Migration Predictor (Streamlit)", layout="wide")

st.title("Global Migration Predictor — Streamlit")

st.markdown("This lightweight Streamlit wrapper reads bilateral migration and country centroids and shows migration outflows by origin.\n\nSet env vars `BILAT_FILE` and `COUNTRIES_FILE` to point to your data if they are not in Downloads.")

if not BILAT_FILE.exists():
    st.error(f"Bilateral migration file not found: {BILAT_FILE}\n\nPlease set the BILAT_FILE env var or place bilat_mig.csv in your Downloads folder.")
    st.stop()

if not COUNTRIES_FILE.exists():
    st.error(f"Countries file not found: {COUNTRIES_FILE}\n\nPlease set the COUNTRIES_FILE env var or place countries.csv in your Downloads folder.")
    st.stop()

# Load data
bilat = pd.read_csv(BILAT_FILE)
centroids = pd.read_csv(COUNTRIES_FILE)

# normalize column names for compatibility
if 'orig' not in bilat.columns and 'origin' in bilat.columns:
    bilat = bilat.rename(columns={'origin':'orig', 'destination':'dest'})

year_col = 'year0' if 'year0' in bilat.columns else ('year' if 'year' in bilat.columns else None)
if year_col is None:
    st.warning('No year column found in bilateral data. Defaulting to available rows as a single snapshot.')
    bilat['year0'] = 0
    year_col = 'year0'

# Prepare country coords mapping — try common column names
if 'COUNTRY' in centroids.columns:
    name_col = 'COUNTRY'
elif 'country' in centroids.columns:
    name_col = 'country'
elif 'ISO' in centroids.columns:
    name_col = 'ISO'
else:
    name_col = centroids.columns[0]

if 'latitude' not in centroids.columns or 'longitude' not in centroids.columns:
    st.error('Countries file must contain latitude and longitude columns.')
    st.stop()

coords = {str(r[name_col]).strip(): (float(r['latitude']), float(r['longitude'])) for _, r in centroids.iterrows()}

# UI controls
years = sorted(bilat[year_col].dropna().unique())
selected_year = st.sidebar.selectbox('Year / snapshot', years)
top_k = st.sidebar.slider('Top K flows per origin to display', 1, 20, 5)

df = bilat[bilat[year_col] == selected_year].copy()
if 'orig' not in df.columns or 'dest' not in df.columns:
    st.error('Bilateral CSV must include "orig" and "dest" columns (or origin/destination).')
    st.stop()

# compute migration out per origin
if 'mig_rate' in df.columns:
    migration_out = df.groupby('orig')['mig_rate'].sum().to_dict()
else:
    # fallback to counts if present
    if 'mig_count' in df.columns:
        migration_out = df.groupby('orig')['mig_count'].sum().to_dict()
    else:
        migration_out = df.groupby('orig').size().to_dict()

locations = list(migration_out.keys())
z = [migration_out.get(loc, 0) for loc in locations]

fig = go.Figure()
fig.add_trace(go.Choropleth(
    locations=locations,
    locationmode='country names',
    z=z,
    colorscale='Blues',
    colorbar_title='Migration Out'
))

# Add flow lines for top flows
for orig, group in df.groupby('orig'):
    try:
        lat0, lon0 = coords.get(orig)
    except Exception:
        continue
    # pick top K destinations by mig_rate or count
    if 'mig_rate' in group.columns:
        group_sorted = group.sort_values('mig_rate', ascending=False).head(top_k)
    elif 'mig_count' in group.columns:
        group_sorted = group.sort_values('mig_count', ascending=False).head(top_k)
    else:
        group_sorted = group.head(top_k)
    for _, row in group_sorted.iterrows():
        dest = row['dest']
        if dest not in coords:
            continue
        lat1, lon1 = coords[dest]
        fig.add_trace(go.Scattergeo(
            lon=[lon0, lon1], lat=[lat0, lat1], mode='lines',
            line=dict(width=1, color='rgba(0,0,255,0.6)'),
            hoverinfo='text', text=f"{orig} → {dest}: {row.get('mig_rate', row.get('mig_count', ''))}"
        ))

fig.update_geos(showcoastlines=True, showcountries=True, showframe=False)
fig.update_layout(title=f"Migration flows — {selected_year}", margin=dict(l=0, r=0, t=30, b=0))

st.plotly_chart(fig, use_container_width=True)

st.markdown('---')
st.write('Data snapshot (first 200 rows):')
st.dataframe(df.head(200))
