# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

sns.set()

app = Dash(__name__)

DATASET_ANALYSIS_2020_PATH = "../../data/tweets_analysis_2020.csv"
DATASET_ANALYSIS_2021_PATH = "../../data/tweets_analysis_2021.csv"
tweet_analysis_2020 = pd.read_csv(filepath_or_buffer=DATASET_ANALYSIS_2020_PATH, sep=",")
# tweet_analysis_2021 = pd.read_csv(filepath_or_buffer=DATASET_ANALYSIS_2021_PATH, sep=",")

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
          'December']
week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_phases = ['Dawn', 'Morning', 'Afternoon', 'Evening', 'Night']


def retweet_count_by_day_phase():
    tweet_analysis_2020_ret_day_phase = tweet_analysis_2020.groupby(['day_phase']).agg(
        day_phase_count=pd.NamedAgg(column="day_phase", aggfunc="count"),
        retweet_count=pd.NamedAgg(column="retweet_count", aggfunc="mean"),
        likes_count=pd.NamedAgg(column="like_count", aggfunc="mean")).reindex(day_phases).reset_index()
    return tweet_analysis_2020_ret_day_phase


def analysis_chart(df, x_bar, y_bar, y2_line, x_title, y_title, y2_title, title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar chart
    fig.add_trace(go.Bar(x=df[x_bar], y=df[y_bar], name=y_title), secondary_y=False)

    # Add line
    fig.add_trace(go.Scatter(x=df[x_bar], y=df[y2_line], name=y2_title), secondary_y=True)

    # Add figure title
    fig.update_layout(title_text=title)

    # Set x-axis title
    fig.update_xaxes(title_text=x_title)

    # Set y-axes titles
    fig.update_yaxes(title_text=y_title, secondary_y=False)
    fig.update_yaxes(title_text=y2_title, secondary_y=True)

    return fig


app.layout = html.Div(children=[
    html.H1(children='Twitter Analysis'),

    html.Div(
        html.Div(
            dcc.Graph(
                id='example-graph',
                figure=analysis_chart(retweet_count_by_day_phase(),
                                      x_bar='day_phase', y_bar='retweet_count', y2_line='likes_count',
                                      x_title="Day phase", y_title="Retweets count", y2_title="Likes count",
                                      title='Average Retweet count per phase of the day in 2020')
            )
        )
    , style={'border': '2px'}),
])

if __name__ == '__main__':
    app.run_server(debug=True)
