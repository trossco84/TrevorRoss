import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input,Output
import dash_auth

VALID_USERNAME_PASSWORD_PAIRS = [
    ['xpyragt-a', 'Cooper.777'],
    ['xpyragt-a', 'cooper.777'],
    ['tross', 'admin']
]

#data folder is weekly outputs, eventually be able to filter on this
#for now:
wiq = pd.read_csv('/Users/trevorross/Desktop/My Projects/bettingatwork/weekly_outputs/3_22_2021')
wiq2 = wiq.copy()
wiq2.set_index('Player',inplace=True)
wiq2['Totals'] = [wiq2.loc[x].Amount if wiq2.loc[x].Action == "Request" else (wiq2.loc[x].Amount*(-1)) for x in wiq2.index]
wiq2.reset_index(inplace=True)
agent_totals = wiq2.drop(['Amount'],axis=1).groupby('Agent').sum().reset_index()


#create data
totals = pd.read_csv('/Users/trevorross/Desktop/My Projects/bettingatwork/agent_totals.csv')
raws = pd.read_csv('/Users/trevorross/Desktop/My Projects/bettingatwork/raw_archives.csv')
last_week = raws.Week.max()
lw_data = raws[raws.Week == last_week]
raw2 = raws.copy()

#overall revenue
overall = totals.reset_index()
overall.head()

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.title= 'RapiDash'
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.layout = html.Div(
    children=[
        html.Div(children=[
                html.Img(src="https://raw.githubusercontent.com/trossco84/TrevorRoss/master/Sports%20Science/logo2.jpg",className="header-img",alt="logo"),
                html.H1(
                    children="", className="header-title"
                ),
                html.P(
                    children="Agent Dashboard for Betting at Work",
                    className="header-description",
                ),
                html.P(
                    children="",
                    className="header-description",
                ),
                dcc.Dropdown(
                    id='DataFilter',
                    options=[
                        {'label': 'Overall Revenue', 'value': 'OR'},
                        {'label': 'Average Players per Week','value':'AP'},
                        {'label': 'Average Revenue per Week', 'value': 'AR'}
                        ],
                        value='OR',
                        clearable=False,
                        className='dropdown',
                        )
            ],
            className="header",
        ),
    dcc.Graph(
        id='OverallGraph',
        className='card'
    ),
]
)


@app.callback(
    [Output("OverallGraph","figure")],
    [
        Input("DataFilter","value")
    ],
)

def update_overall(selectedfilter):
    if selectedfilter == 'OR':
        OverallGraph_figure = [{
            'data':[
                {
                    'x':overall.Agent,
                    'y':overall['Total Revenue'],
                    'type':'bar'
                }
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                },
                'title': 'Total Revenue'
                },
            }]
    elif selectedfilter =='AP':
        OverallGraph_figure = [{
            'data':[
                {
                    'x':overall.Agent,
                    'y':overall['Avg Players per Week'],
                    'type':'bar'
                }
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                },
                'title': 'Average Players Per Week'
                },
            }]
    elif selectedfilter == 'AR':
        OverallGraph_figure = [{
            'data':[
                {
                    'x':overall.Agent,
                    'y':overall['Avg Revenue per Week'],
                    'type':'bar'
                }
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                },
                'title': 'Average Revenue Per Week'
        },
        }]
    
    return OverallGraph_figure


if __name__ == '__main__':
    app.run_server(debug=True)


#   style={'backgroundColor': colors['background']}, 
    # dcc.Graph(
#         id='Graph1',
#         figure={
#             'data': [
#                 {'x': ['Trev'], 'y': [700], 'type': 'bar', 'name': 'Trev'},
#                 {'x': ['Gabe'], 'y': [300], 'type': 'bar', 'name': 'Gabe'},
#             ],
            # 'layout': {
            #     'plot_bgcolor': colors['background'],
            #     'paper_bgcolor': colors['background'],
            #     'font': {
            #         'color': colors['text']
            #     }
#             }
#         }
#     )
# ]

# dcc.Dropdown(
#     options=[
#         {'label': 'New York City', 'value': 'NYC'},
#         {'label': 'Montréal', 'value': 'MTL'},
#         {'label': 'San Francisco', 'value': 'SF'}
#     ],
#     value='MTL'
# )

                # html.Img(src="/Users/trevorross/Desktop/My Projects/bettingatwork/dashboard/assets/logo2.jpg",className="header-img",alt="logo")
                # html.H1(
                #     children="RapiDash", className="header-title"
                # ),
                # html.P(
                #     children="Agent Dashboard for Betting at Work",
                #     className="header-description",
                # )
