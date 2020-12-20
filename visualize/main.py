import os
import io
import cv2
import dash
import base64
import numpy as np 
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
import utils.dash_reusable_components as drc

from mxnet import nd
from dash.dependencies import Input, Output, State
from analytics.cnn import MRI
from analytics.data_ingestion import Data


app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
# Initialize server
server = app.server

# Define model 
net = MRI(
        train_data_loader=None, 
        val_data_loader=None, 
    )
file_name = '../data/params/net.params'
net.load_params(file_name)

# Instantiate data object
d = Data()


def get_images(contents, fname):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    jpg_as_np = np.frombuffer(decoded, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np,flags=1)

    return [img]

   
def layout():
    """
    Return the layout Dash will use to serve content to the user

    Returns
    -------
    HTML Div component : Returns the root node for the app layout, all of the
    app's layout is served under the root node
    """
    return html.Div(
        children=[
            html.Div(
                className="banner",
                children=[
                    html.Div(
                        className="container scalable",
                        children=[
                            html.H2(
                                id="banner-title",
                                children=[
                                    html.A(
                                        # Change App Name here
                                        "Magnetic Resonance Imaging Tumor Classification App",
                                        href="https://github.com/plotly/dash-svm",
                                        style={
                                            "text-decoration": "none",
                                            "color": "inherit",
                                            "text-align": 'center'
                                        },
                                    ),
                                ],
                            ),
                            html.A(
                                html.Img(
                                    id="power-logo",
                                    src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FMagnetic_resonance_imaging&psig=AOvVaw3Ee7HMqxyWEkNugkNuS9oo&ust=1607975819095000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCNiM-O_ey-0CFQAAAAAdAAAAABAD",
                                    alt="Not loaded",
                                    style={'width': 75, 'height': 75}
                                )
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="body",
                className="container scalable",
                children=[
                    html.Div(
                        id="app-container",
                        children=[
                            html.Div(
                                # Define components to the left of the web page
                                # Components are separated by 'cards', providing
                                # a visual separation for each group of components
                                id="left-column",
                                children=[
                                    drc.Card(
                                        # """""""""""""""""""""""""""""""""""""""""""""
                                        # First card, plots the average envelope for all
                                        # time series provided in a data set and allows
                                        # the user to upload a ML model
                                        # """""""""""""""""""""""""""""""""""""""""""""
                                        id="first-card",
                                        children=[
                                            html.Div(["Upload image"]),
                                            # Dash core component to upload a file
                                            dcc.Upload(
                                                # id is used to retrieve file object from component
                                                id="upload-model-button",
                                                children=[
                                                    html.Button(
                                                    # Button used to upload the file and to indicate the
                                                    # user what of algorithm is uploaded by manipulating the
                                                    # button's title through a callback
                                                    html.P(id="model-upload"),
                                                    style={"color": "white"}),                                        
                                                ],
                                                style={
                                                    "margin-left": "0.5px",
                                                },
                                                multiple=False
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            # Dash core component, used to present a spinner while the graph is loaded
                            dcc.Loading(
                                html.Div(
                                    # id attribute is used to provide an actual graph though callbacks
                                    id="div-graphs",
                                    children=dcc.Loading(
                                        id="loading-graph",
                                        fullscreen=True,
                                        type='cube',
                                        children=dcc.Graph(
                                            id="graph-sklearn",
                                            figure=dict(
                                                layout=dict(
                                                    plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                                )
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


app.layout = layout



@app.callback(Output("model-upload", "children"),
              [Input('upload-model-button', 'contents')],
              [State("upload-model-button", "filename"),
               State("upload-model-button", "last_modified")])
# """""""""""""""""""""""""""""""""""""""""""""
# update_button: returns a string to be displayed by the upload file button
# """""""""""""""""""""""""""""""""""""""""""""
def update_button(contents, file_name, last_modified):
    if contents is not None:
        return "Upload another"
    else:
        return "Upload image"



# """""""""""""""""""""""""""""""""""""""""""""
# Model training callbacks
# """""""""""""""""""""""""""""""""""""""""""""
@app.callback(
    Output('div-graphs', 'children'),
    [Input('upload-model-button', 'contents')],
    [State("upload-model-button", "filename"),
     State("upload-model-button", "last_modified")],
)
def update_svm_graph(
        file_contents,
        file_name,
        file_last_modified
):
    """
        Parameters
        ----------
        file_contents: contents of the file uploaded, denoted by 'upload-model-button'

        number_signals: number of signals to analyze, denote by 'number-signals'

        file_name: name of the file uploaded, provided by the state of the button, denoted by 'upload-model-button'

        file_last_modified: last modified state of the file, denoted by 'upload-model-button'

        Returns
        -------
        list : composed of HTML and Dash core components
    """
    if file_contents is not None:
        # print(file_contents)
        images = get_images(file_contents, file_name)

        # Create data loader
        data_shape = (3, 300, 300)
        fake_y = [1 for i in range(len(images))]
        data_loader = d.data_loader(images, fake_y, data_shape=data_shape, transform=False, shuffle=True)
        
        (images, confidence_intervals, predictions) = net.get_predictions(data_loader)
        print("[*] Done Predicting")
        # Perform a pseudo voting scheme to detect an event
        output = list()
        imgs = list()
        for i in range(len(predictions)):
            interval    = confidence_intervals[i].asscalar()
            prediction  = predictions[i]
            imgs.append(nd.transpose(data=images[i], axes=(1,2,0)).asnumpy())
        
            if 'Not' in prediction:
                interval = 1 - interval
            output.append('Tumor {}, {:.2f}% confidence'.format(prediction, interval*100))

        images = None

        # Lambda function to print HTML component
        generate_event_outputs = lambda x: html.P(x)

        # Return the graph for the number of signals specified
        return [
            html.Div(
                id="svm-graph-container",
                children=[
                    html.Div(
                        # Print event indication through list comprehension
                        children=html.P(children=[generate_event_outputs(i) for i in output],
                                        style={"backgroundColor": "#2f3445",
                                               "fontSize": 20}),
                    ),
                    html.Div(
                        html.Img(src=file_contents,
                        style={
                            'width': 500,
                            'height': 500
                        })
                    ),
                ]
            ),
        ]
    else:
        return True


# Running the server
if __name__ == "__main__":
    # debug value allows the web page to be update whenever the code is updated
    app.run_server(debug=True, host='0.0.0.0')
