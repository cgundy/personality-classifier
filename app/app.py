import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from personality_classifier.ml.model_pipeline import get_config
import requests

config = get_config()

app = dash.Dash(__name__)

radio_items = []
valid_models = config["valid_models"]

for model_type in valid_models:
    radio_items.append({"label": model_type, "value": model_type})

app.layout = html.Div(
    [
        dcc.RadioItems(options=radio_items, id="model-selection"),
        html.Button("Submit", id="submit-val", n_clicks=0),
        html.Br(),
        html.Div(id="accuracy"),
    ]
)


@app.callback(
    Output("accuracy", "children"),
    [Input("submit-val", "n_clicks")],
    [State("model-selection", "value")],
)
def update_output(*args):
    states = dash.callback_context.states
    # avoid calling the API if the state is "None"
    if not states["model-selection.value"]:
        return
    r = requests.get(f"http://0.0.0.0:8000/accuracy/{states['model-selection.value']}")
    return r.json()["data"]


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
