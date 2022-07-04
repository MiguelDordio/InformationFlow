import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from os import walk
import seaborn as sns
sns.set()


DATASETS_PATH = "../../data/processed_tweets/"

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_phases = ['Morning', 'Afternoon', 'Dusk', 'Night', 'Middle of the night']
day_phases_old = ['Dawn', 'Morning', 'Afternoon', 'Evening', 'Night']
sentiments = ['Negative', 'Neutral', 'Positive']
hashtags = [True, False]
offline_graphs = True
palette = ['rgb(136, 204, 238)', 'rgb(204, 102, 119)', 'rgb(221, 204, 119)', 'rgb(51, 34, 136)', '#D62728',
               '#FF9900', 'rgb(170, 68, 153)', 'rgb(68, 170, 153)', 'rgb(153, 153, 51)', 'rgb(136, 34, 85)',
               'rgb(102, 17, 0)', 'rgb(136, 136, 136)']


def analysis(filenames: list):
    print("Starting general analysis")
    df = ensemble_dataset(filenames)

    topics_categories = df['topics_cleaned'].unique()[1:]
    tweet_analysis = df[['text', 'year', 'day_phase', 'day_of_week', 'month', 'retweet_count', 'quote_count',
                         'like_count', 'reply_count', 'sentiment', 'hashtags', 'topics_cleaned']]

    # Average retweet and like count per phase of the day
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['day_phase'], day_phases_old)
    analysis_chart(df_analysis, 'day_phase', '% with retweets', '% with likes', 'Day phase', '% with retweets',
                   '% with likes', 'Percentage of retweets and likes during the day')
    analysis_chart(df_analysis, 'day_phase', 'retweets mean', 'likes mean', 'Day phase', 'Retweets mean', 'Likes mean',
                   'Average retweets and likes during the day')

    # Average retweet and like count during the week
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['day_of_week'], week_days)
    analysis_chart(df_analysis, 'day_of_week', '% with retweets', '% with likes', 'Weekday', '% with retweets',
                   '% with likes', 'Percentage of retweets and likes during the week')
    analysis_chart(df_analysis, 'day_of_week', 'retweets mean', 'likes mean', 'Weekday', 'Retweets mean', 'Likes mean',
                   'Average retweets and likes during the week')

    # Average retweet count per month
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['month'], months)
    analysis_chart(df_analysis, 'month', '% with retweets', '% with likes', 'Month', '% with retweets',
                   '% with likes', 'Percentage of retweets and likes during the year')
    analysis_chart(df_analysis, 'month', 'retweets mean', 'likes mean', 'Month', 'Retweets mean', 'Likes mean',
                   'Average retweets and likes during the year')

    # Tweets performance by sentiment
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['sentiment'], sentiments)
    analysis_chart(df_analysis, 'sentiment', '% with retweets', '% with likes', 'Sentiment', 'Retweet count',
                   'Likes count', 'Percentage of retweets and likes by sentiment')
    analysis_chart(df_analysis, 'sentiment', 'retweets mean', 'likes mean', 'Sentiment', 'Retweets mean', 'Likes mean',
                   'Average retweets and likes number by sentiment')

    # Tweets performance by topics
    topic_analysis = tweet_analysis[tweet_analysis['topics_cleaned'].notnull()].copy()
    topic_analysis['topics_cleaned'].value_counts(normalize=True)

    # Performance of each topic in retweets and likes
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['topics_cleaned'], topics_categories)
    analysis_chart(df_analysis, 'topics_cleaned', '% with retweets', '% with likes', 'Topics', 'Retweet count',
                   'Likes count', 'Percentage of retweets and likes by topic')
    analysis_chart(df_analysis, 'topics_cleaned', 'retweets mean', 'likes mean', 'Topics', 'Retweets mean',
                   'Likes mean',
                   'Average tweets performance by topic')

    # Average retweet count per topic during the day
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['day_phase', 'topics_cleaned'], day_phases)
    multiple_analysis_chart(df_analysis, "day_phase", "% with retweets", "year", "topics_cleaned",
                            "Percentage of retweets by topic during the day",
                            "Day phase", "% with retweets", offline_graphs)

    # Average retweet count per topic during the week
    df_analysis = retweets_likes_info_by_year(topic_analysis, ['day_of_week', 'topics_cleaned'], week_days)
    multiple_analysis_chart(df_analysis, "day_of_week", "% with retweets", "year", "topics_cleaned",
                            "Percentage of retweets by topic during the week",
                            "Weekday", "% with retweets", offline_graphs)

    # Average retweet count per topic during the year
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['month', 'topics_cleaned'], months)
    multiple_analysis_chart(df_analysis, "month", "% with retweets", "year", "topics_cleaned",
                            "Percentage of retweets by topic during the year",
                            "Month", "% with retweets", offline_graphs)

    # Impact of hashtags in topic popularity
    df_analysis = retweets_likes_info_by_year(topic_analysis, ['hashtags', 'topics_cleaned'], hashtags)
    multiple_analysis_chart(df_analysis, "topics_cleaned", "% with retweets", "year", "hashtags",
                            "Hashtags presence by topic and corresponding % retweet count",
                            "Topics", "% with retweets", offline_graphs)

    # Tweet sentiment per topic
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['sentiment', 'topics_cleaned'], sentiments)
    multiple_analysis_chart(df_analysis, "topics_cleaned", "% with retweets", "year", "sentiment",
                            "Tweet sentiment by topic and corresponding % retweet count",
                            "Topics", "% with retweets", offline_graphs)


def ensemble_dataset(filenames):
    df = pd.DataFrame()
    for filename in filenames:
        df_temp = pd.read_csv(filepath_or_buffer=filename, sep=",", engine=None)
        df_temp = df_temp.sort_values(by='timestamp', ascending=True)
        df_temp = df_temp.drop(['index', 'Unnamed: 0'], axis=1)
        df = pd.concat([df, pd.DataFrame.from_records(df_temp)])
    return df.reset_index()


def retweets_likes_info_by_year(source_df, cols, cats_sort):
    df = pd.DataFrame()
    for y in np.sort(source_df['year'].unique()):
        dfy = source_df[source_df['year'] == y]
        df_all = dfy.groupby(cols).agg(
                                        **{"count " + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")},
                                        **{"retweets mean": pd.NamedAgg(column="retweet_count", aggfunc="mean")},
                                        **{"likes mean": pd.NamedAgg(column="like_count", aggfunc="mean")}).round(2)

        df_rets = dfy[dfy['retweet_count'] > 0].groupby(cols).agg(**{"count " + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")})
        df_likes = dfy[dfy['like_count'] > 0].groupby(cols).agg(**{"count " + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")})

        if len(cols) == 1:
            df_all = df_all.reindex(cats_sort).reset_index()
            df_rets = df_rets.reindex(cats_sort).reset_index()
            df_likes = df_likes.reindex(cats_sort).reset_index()
            df_all['year'] = [y for i in range(len(cats_sort))]
            df_all['% with retweets'] = np.round((df_rets["count " + cols[0]] / df_all["count " + cols[0]]) * 100, 2)
            df_all['% with likes'] = np.round((df_likes["count " + cols[0]] / df_all["count " + cols[0]]) * 100, 2)
        else:
            df_all = df_all.reset_index()
            df_rets = df_rets.reset_index()
            df_likes = df_likes.reset_index()

            year_items = []
            for cat in cats_sort:
                count = len(df_all[df_all[cols[0]] == cat][cols[1]].unique())
                year_items += [y for i in range(count)]

            df_all['year'] = year_items

            filter_rets = df_all.merge(df_rets,on=[cols[0], cols[1]])
            df_all['% with retweets'] = np.round((filter_rets["count " + cols[0] + "_y"] / filter_rets["count " + cols[0] + "_x"]) * 100, 2)

            filter_likes = df_all.merge(df_likes,on=[cols[0], cols[1]])
            df_all['% with likes'] = np.round((filter_likes["count " + cols[0] + "_y"] / filter_likes["count " + cols[0] + "_x"]) * 100, 2)

        df_all['sum'] = [df_all["count " + cols[0]].sum() for i in range(df_all.shape[0])]
        df_all['% ' + cols[0]] = (df_all["count " + cols[0]] / df_all['sum']) * 100

        df = pd.concat([df, pd.DataFrame.from_records(df_all)])

    if len(cols) == 1:
        return df[['year', cols[0], "count " + cols[0], '% ' + cols[0], '% with retweets', '% with likes', 'retweets mean', 'likes mean']]
    else:
        return df[['year', cols[0], cols[1], "count " + cols[0], '% ' + cols[0], '% with retweets', '% with likes', 'retweets mean', 'likes mean']]


def analysis_chart(df, x_col, y_bar, y_line, x_name, y_bar_name, y_line_name, plot_title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    years = np.sort(df['year'].unique())
    years_count = len(df['year'].unique())

    for i in range(years_count):
        year = years[i]
        dfy = df[df['year'] == year]
        fig.add_trace(go.Bar(x=dfy[x_col], y=dfy[y_bar], text=list(map(str, dfy[y_bar].tolist())), name=y_bar_name + ' ' + str(year), marker_color=palette[i], width=0.28, textposition='inside'), secondary_y=False)
        fig.add_trace(go.Scatter(x=dfy[x_col], y=dfy[y_line], name=y_line_name + ' ' + str(year), marker_color=palette[i+years_count]), secondary_y=True)

    fig.update_yaxes(title_text=y_line_name, secondary_y=True)
    fig.update_yaxes(title_text=y_bar_name, secondary_y=False)
    fig.update_layout(title_text=plot_title, width=900, height=500)
    fig.update_xaxes(title_text=x_name)
    fig.show()


def multiple_analysis_chart(df, x, y, color, text, title, x_title, y_title, offline):
    df.year = df.year.astype(str)
    fig = px.bar(df, x=x, y=y, color=color, text=text, title=title, width=900, height=500, barmode="group", color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    fig.show()
    if offline:
        plotly.offline.plot(fig, filename='../../data/charts/' + title + '.html')
