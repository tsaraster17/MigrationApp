import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# --------------------
# Paths (portable)
# --------------------
# Base directory (package folder)
BASE_DIR = Path(__file__).resolve().parent
# Allow overrides via environment variables; otherwise prefer user's Downloads then repository files
DOWNLOADS = Path.home() / "Downloads"
BILAT_CSV = Path(os.environ.get("BILAT_FILE", DOWNLOADS / "bilat_mig.csv"))
COUNTRIES_CSV = Path(os.environ.get("COUNTRIES_FILE", DOWNLOADS / "countries.csv"))
# Local artifact files (kept in working directory)
MODEL_FILE = Path(os.environ.get("MODEL_FILE", BASE_DIR / "migration_models.pkl"))
PREDICTIONS_FILE = Path(os.environ.get("PREDICTIONS_FILE", BASE_DIR / "migration_predictions.pkl"))

# --------------------
# Load country coordinates
# --------------------
centroids = pd.read_csv(COUNTRIES_CSV)
country_coords = {row['COUNTRY']: (row['latitude'], row['longitude']) for _, row in centroids.iterrows()}

# --------------------
# Load migration data
# --------------------
data = pd.read_csv(BILAT_CSV)
data.rename(columns={'orig': 'origin', 'dest': 'destination'}, inplace=True)

target_col = 'mig_rate'
feature_cols = ['sd_drop_neg', 'sd_rev_neg', 'da_min_open', 'da_min_closed', 'da_pb_closed']

# --------------------
# Train or load models
# --------------------
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        models = pickle.load(f)
else:
    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    xgb = XGBRegressor(n_estimators=200, random_state=42, eval_metric='rmse')  # fixed warning
    lgb = LGBMRegressor(n_estimators=200, random_state=42)
    hgb = HistGradientBoostingRegressor(max_iter=200, random_state=42)

    for model in [rf, xgb, lgb, hgb]:
        model.fit(X_train, y_train)

    models = {'rf': rf, 'xgb': xgb, 'lgb': lgb, 'hgb': hgb}

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(models, f)

# --------------------
# Generate or load predictions 2026–2050
# --------------------
if os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, 'rb') as f:
        all_predictions = pickle.load(f)
else:
    all_predictions = {}
    start_year, end_year = 2026, 2050
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    X_full = data[feature_cols]

    total_pop = data.groupby('origin')['pop_origin'].mean().to_dict() if 'pop_origin' in data.columns else {}
    total_mig = data.groupby('origin')['mig_count'].mean().to_dict() if 'mig_count' in data.columns else {}

    for year in range(start_year, end_year + 1):
        for q in quarters:
            quarter_key = f"{q} {year}"
            preds = np.mean([m.predict(X_full) for m in models.values()], axis=0)

            quarter_df = data[['origin', 'destination']].copy()
            quarter_df['mig_rate'] = preds
            quarter_df['mig_count'] = preds * data.get('pop_origin', 1e6)  # fake absolute count if not in CSV
            all_predictions[quarter_key] = quarter_df

    with open(PREDICTIONS_FILE, 'wb') as f:
        pickle.dump(all_predictions, f)

# --------------------
# Dash App
# --------------------
app = dash.Dash(__name__)
app.title = "Global Migration Predictor"

quarter_options = list(all_predictions.keys())

app.layout = html.Div([
    html.H1("🌍 Global Migration Prediction Dashboard", style={'textAlign': 'center'}),
    dcc.Slider(
        id='quarter-slider',
        min=0,
        max=len(quarter_options) - 1,
        value=0,
        marks={i: quarter_options[i] for i in range(0, len(quarter_options), 4)},
        step=1
    ),
    dcc.Store(id='clicked-countries', data=[]),
    dcc.Graph(id='migration-map', style={'height': '90vh'})
])

# --------------------
# Callbacks
# --------------------
@app.callback(
    Output('clicked-countries', 'data'),
    Input('migration-map', 'clickData'),
    State('clicked-countries', 'data')
)
def toggle_country(clickData, clicked):
    if clickData is None:
        return clicked
    point = clickData['points'][0]
    country = point.get('location') or point.get('text').split(' → ')[0]
    if country in clicked:
        clicked.remove(country)
    else:
        clicked.append(country)
    return clicked


@app.callback(
    Output('migration-map', 'figure'),
    Input('quarter-slider', 'value'),
    Input('migration-map', 'hoverData'),
    State('clicked-countries', 'data')
)
def update_map(slider_value, hoverData, clicked_countries):
    quarter = quarter_options[slider_value]
    df = all_predictions[quarter]

    migration_out = df.groupby('origin')['mig_rate'].sum().to_dict()

    fig = go.Figure()

    # Choropleth – blue gradient for migration intensity
    fig.add_trace(go.Choropleth(
        locations=list(migration_out.keys()),
        locationmode='country names',
        z=list(migration_out.values()),
        colorscale='Blues',
        colorbar_title="Migration Out (%)",
        marker_line_color='black'
    ))

    # Determine hover/click behavior
    countries_to_show = clicked_countries.copy()
    if hoverData is not None:
        hover_country = hoverData['points'][0].get('location')
        if hover_country and hover_country not in countries_to_show:
            countries_to_show.append(hover_country)

    # Arrows for migration flows
    for _, row in df.iterrows():
        origin = row['origin']
        dest = row['destination']
        if origin not in country_coords or dest not in country_coords:
            continue
        if origin not in countries_to_show:
            continue

        lat0, lon0 = country_coords[origin]
        lat1, lon1 = country_coords[dest]
        mig_rate = row['mig_rate']
        mig_count = row['mig_count']

        fig.add_trace(go.Scattergeo(
            lat=[lat0, lat1],
            lon=[lon0, lon1],
            mode='lines',
            line=dict(width=max(1, mig_rate * 8), color='rgba(0, 0, 255, 0.6)'),  # thicker line for stronger flow
            hoverinfo='text',
            text=f"<b>{origin} → {dest}</b><br>"
                 f"Migration Rate: {mig_rate:.2f}%<br>"
                 f"Migrants: {int(mig_count):,}"
        ))

    fig.update_geos(showcoastlines=True, showcountries=True, showframe=False)
    fig.update_layout(
        title=f"Global Migration - {quarter}",
        geo=dict(projection_type='natural earth'),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


# --------------------
# Run app
# --------------------
if __name__ == "__main__":
    app.run(debug=True)
