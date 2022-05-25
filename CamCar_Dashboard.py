import os, json, time
import socket
import pandas as pd
import datetime
import plotly.express as px
from dash import dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State
import dash_daq as daq
import dash_bootstrap_components as dbc
from flask import Flask, Response, request
import CamCar as CC


df = None
car = CC.CamCar()


server = Flask(__name__)
app = dash.Dash(
    __name__,
    server = server,
    external_stylesheets=[dbc.themes.SUPERHERO],
    meta_tags=[
        {"name": "viewport"},
        {"content": "width = device,width, initial-scale=1.0"},
    ],
)


@server.route('/video_feed')
def video_feed():
    """Will return the video feed from the camera

    Returns:
        Response: Response object with the video feed
    """
    return Response(car.get_image_bytes(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def get_ip_address():
    """Ermittlung der IP-Adresse im Netzwerk

    Returns:
        str: lokale IP-Adresse
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    socket_ip = s.getsockname()[0]
    s.close()
    return socket_ip


def getLoggerFiles():
    """Files im Ordner "Logger" auflisten

    Returns:
        list: Log-Dateien
    """
    fileList = []
    outputList = []
    if os.path.isdir("Logger"):
        fileList = os.listdir("Logger")
        for file in fileList:
            if file.partition(".")[2] == "log":
                outputList.append(file)
            outputList.sort()
            outputList.reverse()
    return outputList


def getLogItemsList():
    """Definition der Spaltennamen für die Daten im Logfile

    Returns:
        list: Spaltennamen
    """
    return [
        "time",
        "v",
        "dir",
        "st_angle",
    ]


def load_data_to_df(pfad):
    """aktualisiert die Daten im Dataframe

    Args:
        pfad (str): Dateipfad zum File das geladen werden soll
    """
    global df
    df = pd.read_json(os.path.join("Logger", pfad))
    df.columns = getLogItemsList()


FP_LISTE = [  # Liste der auswählbaren Fahrprogramme
    {"label": "FP 1 --- Parameter-Tuning", "value": 1},
    {"label": "FP 2 --- OpenCV", "value": 2},
    {"label": "FP 3 --- DeepNN", "value": 3},
]

kpi_1 = dbc.Card([dbc.CardBody([html.H6("vMax"), html.P(id="kpi1")])])
kpi_2 = dbc.Card([dbc.CardBody([html.H6("vMin"), html.P(id="kpi2")])])
kpi_3 = dbc.Card([dbc.CardBody([html.H6("vMean"), html.P(id="kpi3")])])
kpi_4 = dbc.Card([dbc.CardBody([html.H6("time"), html.P(id="kpi4")])])
kpi_5 = dbc.Card([dbc.CardBody([html.H6("vm"), html.P(id="kpi5")])])

row_joystick = dbc.Row(
    [
        dbc.Col([html.P("Manuell on/off"), dbc.Switch(id="sw_manual")], width=4),
        dbc.Col(
            daq.Joystick(id="joystick", size=100, className="mb-3"),
            width=4,
        ),
        dbc.Col(
            [
                html.P(id="value_joystick"),
            ],
            width=4,
        ),
    ]
)
CARD_ManuelleSteuerung = dbc.Card(
    [
        dbc.Row(
            [  # Titel Manuelle Steuerung
                html.H3(
                    id="label_test",
                    children="Manuelle Steuerung",
                    style={
                        "textAlign": "left",
                        "paddingTop": 20,
                        "paddingBottom": 20,
                    },
                )
            ]
        ),
        dbc.Row(
            [  # Slider Speed
                dbc.Col([html.H6("max. speed:")], width=4),
                dbc.Col(
                    [
                        dcc.Slider(
                            min=0,
                            max=100,
                            step=10,
                            id="slider_speed",
                            value=40,
                            updatemode="drag",
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
        row_joystick,
    ],
)
COL_Headline = [  # Col Headline
    dbc.Col(
        [  # Col 1
            html.H1(
                id="title_main",
                children="Camp2Code - Gruppe 4",
                style={
                    "textAlign": "center",
                    "marginTop": 40,
                    "marginBottom": 40,
                    "text-decoration": "underline",
                },
            )
        ],
        width=12,
    ),
]
dbc.Col([html.P("Manuell on/off"), dbc.Switch(id="sw_manual")], width=4),
COL_Tuning = [  # Col Tuning
    dbc.Col(
                    [   
                        dbc.Switch(id="switch_canny"),
                        html.Div(id='switch_canny-output'),
                        dbc.Switch(id="switch_houghes"),
                        html.Div(id='switch_houghes-output'),
                        dcc.Slider(
                            min=1,
                            max=10,
                            step=1,
                            id="slider_scale",
                            value=1,
                            updatemode="drag",
                        ),
                        html.Div(id='output-container-scale-slider'),
                        dcc.Slider(
                            min=0,
                            max=255,
                            step=5,
                            id="slider_canny_lower",
                            value=50,
                            updatemode="drag",
                        ),
                        html.Div(id='output-container-canny-lower-slider'),
                        dcc.Slider(
                            min=0,
                            max=255,
                            step=5,
                            id="slider_canny_upper",
                            value=125,
                            updatemode="drag",
                        ),
                        html.Div(id='output-container-canny-upper-slider'),
                        dcc.Slider(
                            min=0,
                            max=100,
                            step=2,
                            id="slider_houghes_threshold",
                            value=40,
                            updatemode="drag",
                        ),
                        html.Div(id='output-container-houghes_threshold'),
                        dcc.Slider(
                            min=0,
                            max=140,
                            step=2,
                            id="slider_houghes_minLineLength",
                            value=70,
                            updatemode="drag",
                        ),
                        html.Div(id='output-container-houghes_minLineLength'),
                        dcc.Slider(
                            min=0,
                            max=100,
                            step=2,
                            id="slider_houghes_maxLineGap",
                            value=30,
                            updatemode="drag",
                        ),
                        html.Div(id='output-container-houghes_maxLineGap'),
                    ],
            )
]
COL_Logfiles = [  # Col Logfiles
    dbc.Row(
        [
            html.H2(  # Überschrift
                id="title_Logfiles",
                children="Logfiles",
            ),
            dcc.Dropdown(  # Dropdown Logfiles
                id="dd_Logfiles",
                placeholder="Bitte Logging-File wählen!",
                options=getLoggerFiles(),
                value="0",
                multi=False,
                style={"color": "black"},
            ),
        ]
    ),
    dbc.Row(
        [
            dbc.Col([kpi_1]),
            dbc.Col([kpi_2]),
            dbc.Col([kpi_3]),
            dbc.Col([kpi_4]),
            dbc.Col([kpi_5]),
        ],
        style={"paddingTop": 10, "paddingBottom": 10},
    ),
]
COL_Plot = [  # Col Plot
    dbc.Row(
            [
                html.H3(id="titel_LogDetails", children="Plot-Line Auswahl"),
                dcc.Dropdown(  # Dropdown Log-Details
                    id="dd_LogDetails",
                    options=getLogItemsList()[1:],
                    value=getLogItemsList()[1:4],
                    multi=True,
                    style={"color": "black"},
                ),
            ],
            style={"paddingBottom": 10},
        ),
    dbc.Row(
            [
                dcc.Graph(id="plot_Logfile"),
                        html.P(
                            id="Fussnote",
                            children="hier das Copyright ;)",
                            style={"textAlign": "right"},
                        ),
                        html.Div(id="dummy"),
                        html.Div(id="dummy2"),
                        dcc.Interval(id="intervall_10s", interval=10000),
                        dcc.Interval(
                            id="interval_startup",
                            max_intervals=1,
                        ),
            ]
    )
]
COL_Fahrzeugsteuerung = [  # Col Fahrzeugsteuerung
    dbc.Row(
        [  # Titel
            html.H2(
                id="titel_Fahrzeugsteuerung",
                children="Fahrzeugsteuerung",
                style={
                    "textAlign": "center",
                    "paddingBottom": 10,
                },
            )
        ]
    ),
    dbc.Row(
        [  # Dropdown Fahrprogramm
            dcc.Dropdown(
                id="dd_Fahrprogramm",
                placeholder="Bitte Fahrprogramm wählen:",
                options=FP_LISTE,
                value=0,
                style={"color": "black"},
            )
        ]
    ),
    dbc.Row(
        [  # Buttons
            dbc.Col(
                [
                    dbc.Button(
                        children="Start",
                        id="btn_start",
                        color="dark",
                        className="me-1",
                        n_clicks=0,
                    )
                ],
                width=4,
            ),
            dbc.Col(
                [   
                    html.Div([
                        dbc.Button(
                        [dbc.Spinner(size="sm"), " Driving..."],
                        color="primary",
                        disabled=True,
                        )],
                        id='spinner_drive',
                        style={'display':'none'},
                    )
                ],
                width=4,
            ),
            dbc.Col(
                [
                    dbc.Button(
                        children="STOP",
                        id="btn_stop",
                        color="dark",
                        className="me-1",
                        n_clicks=0,
                    )
                ],
                width=4,
            ),
        ],
        style={"paddingTop": 10, "paddingBottom": 10},
        justify="between",
    ),
    dbc.Row(
        [  # Manuelle Steuerung
            CARD_ManuelleSteuerung,
        ]
    ),
]
COL_Kamera = [  # Col Kamera
    dbc.Row(
            [  # Titel
                html.H2(
                    id="titel_Kamera",
                    children="Kamera",
                    style={
                        "textAlign": "center",
                        "paddingBottom": 10,
                    },
                )
            ]
        ),
    dbc.Row(
            [  # Kamerabild
                html.Img(src="/video_feed")
            ]
        ),
]

app.layout = dbc.Container(
    [
        dbc.Row(
            [  # Row 1
                dbc.Col(
                    COL_Headline,
                ),
            ],
            justify="center",
        ),
        dbc.Row(
            [  # Row 2
                dbc.Col(
                    COL_Fahrzeugsteuerung,
                ),
                dbc.Col([], width=1),  # Col Space
                dbc.Col(
                    COL_Kamera,
                ),
            ],
            justify="center",
            style={"paddingTop": 20, "paddingBottom": 20},
        ),
        dbc.Row(
            [   # Row 3
                dbc.Col(  # Tuning
                    COL_Tuning,
                ),
            ],
            style={"paddingTop": 20, "paddingBottom": 20},
        ),
        dbc.Row(
            [   # Row 4
                dbc.Col(  # Logfile-Handling
                    COL_Logfiles,
                ),
            ],
            style={"paddingTop": 20, "paddingBottom": 20},
        ),
        dbc.Row(
            [   # Row 5
                dbc.Col(  # Plot
                    COL_Plot,
                ),
            ],
            style={"paddingTop": 20, "paddingBottom": 20},
        ),
    ]
)


@app.callback(
    Output("value_joystick", "children"),
    Input("joystick", "angle"),
    Input("joystick", "force"),
    State("sw_manual", "value"),
    State("slider_speed", "value"),
)
def joystick_values(angle, force, switch, max_Speed):
    """Steuerung über Joystick
        berechnet anhand der Joystick-Werte den Lenkeinschlag
        und zusätzlich mit der eingestellten Maximalgeschwindigkeit die Fahrgeschwindigkeit

    Args:
        angle (float): Winkelwert des Joysticks
        force (float): Kraftweg des Joysticks
        switch (bool): Schalter "manuelle Fahrt"
        max_Speed (int): Wert Slider "slider_speed"

    Returns:
        str: gibt die ermittelten Sollwerte zurück
    """
    debug = ""
    if angle != None and force != None:
        if switch:
            power = round(force, 1)
            winkel = 0
            dir = 0
            if force == 0:
                winkel = 0
                dir = 0
                car.drive(0, 0)
                car.steering_angle = 90
            else:
                power = power * max_Speed
                if power > max_Speed:
                    power = max_Speed
                if angle <= 180:
                    dir = 1
                    winkel = round(45 - (angle / 2), 0)
                else:
                    dir = -1
                    winkel = round(((angle - 180) / 2) - 45, 0)
                car.drive(int(power), dir)
                car.steering_angle = winkel+90
                debug = f"Angle: {winkel} speed: {power} dir: {dir}"
        else:
            debug = "Man. mode off"
            car.drive(0, 0)
            car.steering_angle = 90
    else:
        debug = "None-Value"
    return debug


def computeKPI(data):
    """Berechnung der Kenndaten des Log-Files

    Args:
        data (pandas.DataFrame): Log-Daten als DataFrame

    Returns:
        tuple of str: berechnete Werte
    """
    vmax = data["v"].max()
    vmin = data["v"][1:].min()
    vmean = round(data["v"].mean(), 2)
    duration = round(data["time"].max(), 2)
    route = round(vmean * duration, 2)
    return vmax, vmin, vmean, duration, route


@app.callback(
    Output(component_id="kpi1", component_property="children"),
    Output(component_id="kpi2", component_property="children"),
    Output(component_id="kpi3", component_property="children"),
    Output(component_id="kpi4", component_property="children"),
    Output(component_id="kpi5", component_property="children"),
    Input("dd_Logfiles", "value"),
)
def update_KPI_DD(logFile):
    """Aktualisieren der Kennwerte nach auswahl eines neuen Files

    Args:
        logFile (str): nur wegen input nötig

    Returns:
        str: setzt die "children" der kpi's
    """
    global df
    time.sleep(0.2)  # damit das File erst geladen werden kann
    try:
        vmax, vmin, vmean, duration, route = computeKPI(df)
        return vmax, vmin, vmean, duration, route
    except:
        return 0, 0, 0, 0, 0


@app.callback(
    Output("plot_Logfile", "figure"),
    Input("dd_Logfiles", "value"),
    Input("dd_LogDetails", "value"),
)
def selectedLog(logFile, logDetails):
    """Auswahl des Logfiles

    Args:
        logFile (str): Log-File
        logDetails (list): Liste mit Elementen die angezeigt werden sollen

    Returns:
        plotly figure: Graph mit ausgewählten Daten
    """
    global df
    if logFile != "0":
        load_data_to_df(logFile)
        # df = pd.read_json(os.path.join("Logger", logFile))
        # time.sleep(0.1)
        # df.columns = getLogItemsList()
        if logDetails != []:
            fig = px.line(df, x="time", y=logDetails, title=logFile)
        else:
            fig = px.line(df, x="time", y="st_angle", title=logFile)
        return fig
    else:
        return px.line()


@app.callback(
    Output("dd_Logfiles", "options"),
    Input("intervall_10s", "n_intervals"),
)
def updateFileList(value):
    """alle 10 Sekunden den Logger-Ordner auf neue Files prüfen"""
    return getLoggerFiles()


@app.callback(
    Output("sw_manual", "value"),
    Output('spinner_drive', 'style'),
    Input("btn_start", "n_clicks"),
    Input("btn_stop", "n_clicks"),
    State("dd_Fahrprogramm", "value"),
    State("slider_speed", "value"),
)
def button_action(btn_start, btn_stop, fp, speed):
    """Buttons "Start" und "Stop" verarbeiten

    Returns:
        int: Schalter für manuellen Betrieb auf 0 setzen
    """
    changed_id = [p["prop_id"] for p in callback_context.triggered][0]
    if "btn_start" in changed_id:
        #spinner_drive = {'display': 'block'}
        if fp == 1:
            car.parameter_tuning()
        elif fp == 2:
            car.fp_opencv(speed)
        elif fp == 3:
            car.fp_deepnn(speed)

    if "btn_stop" in changed_id:
        #spinner_drive = {'display': 'none'}
        car._active = False
        time.sleep(1)

    return 0, spinner_drive


@app.callback(
    Output('switch_canny-output', 'children'),
    Output('switch_houghes-output', 'children'),
    Output('output-container-scale-slider', 'children'),
    Output('output-container-canny-lower-slider', 'children'),
    Output('output-container-canny-upper-slider', 'children'),
    Output('output-container-houghes_threshold', 'children'),
    Output('output-container-houghes_minLineLength', 'children'),
    Output('output-container-houghes_maxLineGap', 'children'),
    Input('switch_canny', 'value'),
    Input('switch_houghes', 'value'),
    Input("slider_scale", "value"),
    Input("slider_canny_lower", "value"),
    Input("slider_canny_upper", "value"),
    Input("slider_houghes_threshold", "value"),
    Input("slider_houghes_minLineLength", "value"),
    Input("slider_houghes_maxLineGap", "value"),
)
def slider_action(sw_canny, sw_houghes, scale, canny_lower, canny_upper, houghes_threshold, houghes_minLineLength, houghes_maxLineGap):
    """XXX"""
    if sw_canny:
        car._canny_frame = True
    else:
        car._canny_frame = False

    if sw_houghes:
        car._houghLP_frame = True
    else:
        car._houghLP_frame = False

    car._frame_scale = 1/scale
    car._canny_lower = canny_lower
    car._canny_upper = canny_upper
    car._houghes_threshold = houghes_threshold
    car._houghes_minLineLength = houghes_minLineLength
    car._houghes_maxLineGap = houghes_maxLineGap

    return 'Canny Edge Detection: "{}"'.format(sw_canny), 'Houghes Lines: "{}"'.format(sw_houghes), 'Frame-Scale: "{}"'.format(scale), 'Canny-Lower: "{}"'.format(canny_lower), 'Canny-Upper: "{}"'.format(canny_upper), 'Houghes-Threshold: "{}"'.format(houghes_threshold), 'Houghes-minLineLength: "{}"'.format(houghes_minLineLength), 'Houghes-maxLineGap: "{}"'.format(houghes_maxLineGap)


if __name__ == "__main__":
    app.run_server(debug=False, host=get_ip_address())