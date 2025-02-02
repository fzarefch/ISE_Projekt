import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine
import dash_bootstrap_components as dbc

# Dash App initialisieren
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
app.config.suppress_callback_exceptions = True
server = app.server

# Verbindung zur PostgreSQL-Datenbank
engine = create_engine('postgresql://postgres:Rayan1388@localhost/restaurant_data')

# Laden der Daten und Extrahieren der Geokoordinaten
def load_violation_data():
    query = """
    SELECT
        substring(geocoded_location FROM '\\(([0-9\\.-]+),') AS latitude,
        substring(geocoded_location FROM ', ([0-9\\.-]+)\\)') AS longitude
    FROM restaurant_inspection_violations
    """
    df = pd.read_sql(query, con=engine)
    # Umwandlung der Spalten in Float
    df[['latitude', 'longitude']] = df[['latitude', 'longitude']].astype(float)
    return df.dropna()

# App-Layout
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Restaurant Violations Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    dbc.Row([
        dbc.Col(
            dbc.Tabs(
                [
                    dbc.Tab(label="Violation Heatmap", tab_id="tab-heatmap"),
                    dbc.Tab(label="Other Analysis (Placeholder)", tab_id="tab-placeholder"),
                ],
                id="tabs",
                active_tab="tab-heatmap",
                className='mb-3'
            ), width=12
        )
    ]),
    html.Div(id="content")
], fluid=True)

# Callback für den aktiven Tab-Inhalt
@app.callback(
    Output("content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "tab-heatmap":
        return html.Div([
            dbc.Row([
                dbc.Col(html.H1("Violation Heatmap", className="text-center text-light my-4"), width=12)
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id="violation-heatmap", style={'height': '600px'}),
                    width=12
                )
            ])
        ])
    elif active_tab == "tab-placeholder":
        return html.Div([
            dbc.Row([
                dbc.Col(html.H1("Other Analysis Placeholder", className="text-center text-light my-4"), width=12)
            ])
        ])
    else:
        return "No content available"

# Callback für die Heatmap
@app.callback(
    Output("violation-heatmap", "figure"),
    Input("tabs", "active_tab")
)
def update_heatmap(active_tab):
    # Heatmap nur aktualisieren, wenn der richtige Tab aktiv ist
    if active_tab != "tab-heatmap":
        return dash.no_update

    df = load_violation_data()

    # Heatmap erstellen
    fig = px.density_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        radius=10,
        center=dict(lat=df["latitude"].mean(), lon=df["longitude"].mean()),
        zoom=10,
        mapbox_style="open-street-map",
        title="Violation Density Heatmap"
    )

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Violation Density"),
    )

    return fig

# App starten
if __name__ == '__main__':
    app.run_server(debug=True, port=8052)