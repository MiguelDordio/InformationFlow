import pandas as pd
from itertools import cycle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import re

DATASET_PATH = "../../data/processed_data/tweets_2020_2021_v2.csv"
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']
week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_phases = ['Morning', 'Afternoon', 'Dusk', 'Night', 'Middle of the night']
day_phases_old = ['Dawn', 'Morning', 'Afternoon', 'Evening', 'Night']
sentiments = ['Negative', 'Neutral', 'Positive']
topics = ['Book', 'Brand', 'Entities', 'Holiday', 'Interest and Hobbies', 'Music', 'News', 'Other', 'Person',
          'Political', 'Sport', 'TV and Movies', 'Video Game']


def analysis():
    df = pd.read_csv(filepath_or_buffer=DATASET_PATH, sep=",", engine=None)
    df = df.sort_values(by='timestamp', ascending=True)
    df = df.drop(['index', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'], axis=1)

    topics_categories = df['topics_cleaned'].unique()[1:]
    tweet_analysis = df[
        ['text', 'year', 'day_phase', 'day_of_week', 'month', 'retweet_count', 'quote_count', 'like_count',
         'reply_count', 'sentiment', 'topics_cleaned']]

    # Average retweet and like count per phase of the day
    df_analysis = retweets_likes_analysis_by_year(tweet_analysis, ['day_phase'], day_phases_old)
    analysis_chart(df_analysis, 'day_phase', 'retweet_count', 'likes_count', 'Day phase', 'Retweet count',
                   'Likes count', 'Average retweet and like count per phase of the day')

    # Average retweet and like count during the week
    df_analysis = retweets_likes_analysis_by_year(tweet_analysis, ['day_of_week'], week_days)
    analysis_chart(df_analysis, 'day_of_week', 'retweet_count', 'likes_count', 'Day phase', 'Retweet count',
                   'Likes count', 'Average retweet and like count during the week')

    # Average retweet count per month
    df_analysis = retweets_likes_analysis_by_year(tweet_analysis, ['month'], months)
    analysis_chart(df_analysis, 'month', 'retweet_count', 'likes_count', 'Month', 'Retweet count', 'Likes count',
                   'Average retweet count per month in 2020')

    # Tweets performance by sentiment
    df_analysis = retweets_likes_analysis_by_year(tweet_analysis, ['sentiment'], sentiments)
    analysis_chart(df_analysis, 'sentiment', 'retweet_count', 'likes_count', 'Sentiment', 'Retweet count',
                   'Likes count', 'Tweets performance by sentiment')

    # Tweets performance by topics
    topic_analysis = tweet_analysis[tweet_analysis['topics_cleaned'].notnull()].copy()
    topic_analysis['topics_cleaned'].value_counts(normalize=True)

    # Performance of each topic in retweets and likes
    df_analysis = retweets_likes_analysis_by_year(topic_analysis, ['topics_cleaned'], topics)
    analysis_chart(df_analysis, 'topics_cleaned', 'retweet_count', 'likes_count', 'Topics', 'Retweet count',
                   'Likes count', 'Performance of each topic by average of retweets and likes in')

    # Average retweet count per topic during the day
    df_analysis = retweets_likes_analysis_by_year(topic_analysis, ['day_phase', 'topics_cleaned'], day_phases_old)
    # print dataframe to table

    # Average retweet count per topic during the week
    df_analysis = retweets_likes_analysis_by_year(topic_analysis, ['day_of_week', 'topics_cleaned'], week_days)

    # Average retweet count per topic during the year
    df_analysis = retweets_likes_analysis_by_year(topic_analysis, ['month', 'topics_cleaned'], months)

    # Impact of hashtags in topic popularity
    topic_analysis['has_hashtags'] = [has_hashtags(text) for text in topic_analysis['text']]
    create_hashtags_df(topics_categories, topic_analysis)

    # Tweet sentiment per topic
    df_analysis = retweets_likes_analysis_by_year(topic_analysis, ['sentiment', 'topics_cleaned'], sentiments)


def retweets_likes_analysis_by_year(source_df, cols, cats_sort):
    df = pd.DataFrame()
    for y in source_df['year'].unique():
        dfy = source_df[source_df['year'] == y]
        dfy = dfy.groupby(cols).agg(
                                        **{
                                            "count_" + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")
                                        },
                                        retweet_count=pd.NamedAgg(column="retweet_count", aggfunc="mean"),
                                        likes_count=pd.NamedAgg(column="like_count", aggfunc="mean"))
        if len(cols) == 1:
            dfy = dfy.reindex(cats_sort).reset_index()
            dfy['year'] = [y for i in range(len(cats_sort))]
        else:
            dfy = dfy.reset_index()
            dfy['year'] = [y for i in range(len(cats_sort) * len(dfy[cols[1]].unique()))]
        df = pd.concat([df, pd.DataFrame.from_records(dfy)])
    return df


def analysis_chart(df, x_col, y_bar, y_line, x_name, y_bar_name, y_line_name, plot_title):
    palette = cycle(px.colors.qualitative.Safe)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for y in df['year'].unique():
        dfy = df[df['year'] == y]
        fig.add_trace(go.Bar(x=dfy[x_col], y=dfy[y_bar], name=str(y), marker_color=next(palette), width=0.4), secondary_y=False)
        fig.add_trace(go.Scatter(x=dfy[x_col], y=dfy[y_line], name=str(y), marker_color=next(palette)), secondary_y=True)
    fig.update_yaxes(title_text=y_line_name, secondary_y=True)
    fig.update_yaxes(title_text=y_bar_name, secondary_y=False)
    fig.update_layout(title_text=plot_title, width=900, height=500)
    fig.update_xaxes(title_text=x_name)
    fig.show()


def has_hashtags(text):
    return len(re.findall(r'\B#\w*[a-zA-Z]+\w*', text)) > 0


def create_hashtags_df(topics_categories, topic_analysis):
    cats_vals = []
    for category in topics_categories:
        tweets_by_topic = topic_analysis[(topic_analysis['topics_cleaned'] == category)]

        avg_retweeters = tweets_by_topic[tweets_by_topic['has_hashtags'] == True]['retweet_count'].mean()
        avg_no_retweets = tweets_by_topic[tweets_by_topic['has_hashtags'] == False]['retweet_count'].mean()

        cat_vals = dict()
        cat_vals['Type'] = 'Has hashtags'
        cat_vals['Topic'] = category
        cat_vals['Average'] = avg_retweeters
        cats_vals.append(cat_vals)

        cat_vals = dict()
        cat_vals['Type'] = 'No hashtags'
        cat_vals['Topic'] = category
        cat_vals['Average'] = avg_no_retweets
        cats_vals.append(cat_vals)

    print(cats_vals)
    df_plot = pd.DataFrame(cats_vals, columns=['Type', 'Topic', 'Average'])
    df_plot
