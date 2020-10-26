import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import requests
import yaml
import json
from typing import Dict

# should eventually have a shared utitlies file, but okay for now
def get_config() -> Dict:
    """Load the config file"""
    with open("config.yml") as c:
        config = yaml.load(c, Loader=yaml.FullLoader)
    return config


# move somewhere else eventually
container_name = "personality"

config = get_config()

app = dash.Dash(__name__)

radio_items = []
valid_models = config["valid_models"]

for model_type in valid_models:
    radio_items.append({"label": model_type, "value": model_type})

app.layout = html.Div(
    [
        dcc.RadioItems(options=radio_items, id="model-selection"),
        html.Br(),
        dcc.Input(id="text-input", type="text", placeholder="text"),
        html.Br(),
        html.Button("Submit", id="submit-val", n_clicks=0),
        html.P("Accuracy: "),
        html.Div(id="accuracy"),
        html.P("Prediction: "),
        html.Div(id="prediction"),
    ]
)


@app.callback(
    Output("accuracy", "children"),
    [Input("submit-val", "n_clicks")],
    [State("model-selection", "value")],
)
def get_accuracy(*args):
    states = dash.callback_context.states
    # avoid calling the API if the state is "None"
    if not states["model-selection.value"]:
        return
    r = requests.get(f"http://0.0.0.0:8000/accuracy/{states['model-selection.value']}")
    return r.json()["data"]


@app.callback(
    Output("prediction", "children"),
    [Input("submit-val", "n_clicks")],
    [State("model-selection", "value"), State("text-input", "value")],
)
def get_prediction(*args):
    states = dash.callback_context.states
    # avoid calling the API if the state is "None"
    if not states["model-selection.value"]:
        return
    url = "http://0.0.0.0:8000/predict"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    myobj = {
        "text_input": {"data": [states["text-input.value"]]},
        "model_type": {"data": states["model-selection.value"]},
    }

    r = requests.post(url, headers=headers, data=json.dumps(myobj))
    return r.json()["data"]


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8051)
