import dash_core_components as dcc
import dash_html_components as html


# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


# Custom Display Components
def Card(children, **kwargs):
    return html.Section(
        style={"margin": "0.5x 0px"},
        className="card", children=children, **_omit(["style"], kwargs))


# Added name attribute and break before slider
def FormattedSlider(name, **kwargs):
    return html.Div(
        style=kwargs.get("style", {}), children=[
            html.Br(),
            html.P(f"{name}:"),
            dcc.Slider(**_omit(["style"], kwargs))
        ]
    )


# Added name attribute to slider component
def NamedSlider(name, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            html.P(f"{name}:"),
            html.Div(style={"margin-left": "6px"}, children=dcc.Slider(**kwargs)),
        ],
    )


# Added name attribute to input component
def NamedInput(name, **kwargs):
    return html.Div(
        style={"margin": "10px 0px"},
        children={
            html.P(children=f'{name}', style={"margin-left": "3px"}),
            html.Div(dcc.Input(**kwargs)),
        },
    )


# Added name attribute to upload component
def NamedUpload(name, **kwargs):
    return html.Div(
        style={'margin': "10px 0px"},
        children=[
            html.P(children=f"{name}:", style={"margin-left": "3px"}),
            dcc.Upload(html.Button("Upload pickle file")),
        ]
    )


# Added name attribute for dropdown component
def NamedDropdown(name, **kwargs):
    return html.Div(
        style={"margin": "10px 0px"},
        children=[
            html.P(children=f"{name}:", style={"margin-left": "3px"}),
            dcc.Dropdown(**kwargs),
        ]
    )


# Added name to radio item component
def NamedRadioItems(name, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[html.P(children=f"{name}:"), dcc.RadioItems(**kwargs)],
    )
