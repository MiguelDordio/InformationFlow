import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from os import walk
import datetime
from statistics import mean
from dateutil.relativedelta import relativedelta
import seaborn as sns

sns.set()

DATASETS_PATH = "../../data/processed_tweets/"
DATASETS_RETWEETS_PATH = "../../data/processed_retweets/"

topics_categories = ['Brand', 'Holiday', 'Person', 'Interest and Hobbies', 'Sport',
                     'TV and Movies', 'Other', 'Video Game', 'Entities', 'Political',
                     'Music', 'Book', 'News']
palette = ['#006D77', '#FBD1A2', '#7DCFB6', '#00B2CA', '#1D4E89', '#F79256', '#aed9e0', '#b8f2e6', '#faf3dd', '#ffa69e',
           '#FE7F2D', '#FCCA46', '#ED7B84', '#F92A82']
offline_charts = True


def retweets_analysis(filenames: list, retweets_filenames: list):
    print("Starting retweeters analysis")
    df = essemble_dataset(filenames)
    df_retweets_info = essemble_dataset(retweets_filenames)

    df_analysis = shared_tweets(df)
    shared_chart(df_analysis)

    print("Generating analysis tables")
    df_retweeters_chars = retweeters_characteristics(df, df_retweets_info, 0)
    df_popular_retweeters_chars = retweeters_characteristics(df, df_retweets_info, 10)

    print("Initializing retweeters analysis")
    # Analysing average followers count of retweeters per topic
    retweeters_info_chart(df_retweeters_chars, topics_categories, "Topic", 'Average Retweeters Followers',
                          "Analysing average followers count of retweeters per topic", offline_charts)

    # Analysing average retweet time ratio of retweeters per topic
    retweeters_info_chart(df_retweeters_chars, topics_categories, "Topic", "Average Retweets Time",
                          "Analysing average retweet time ratio of retweeters per topic", offline_charts)

    # Analysing average retweeter’s account age per topic
    retweeters_info_chart(df_retweeters_chars, topics_categories, "Topic", "Average Retweeters Account Age",
                          "Analysing average retweeters account age per topic", offline_charts)

    print("Initializing popular tweets, retweeters analysis")
    print("Popular retweets (>10) account for only:", (df[df['retweet_count'] > 10].shape[0] /
                                                       df[df['retweet_count'] > 0].shape[0]) * 100, "% of the dataset")

    time_analysis_charts(df_popular_retweeters_chars, "")

    analysis_chart(df_popular_retweeters_chars, "Topic", 'Average Retweeters Followers First Half',
                   'Average Retweeters Followers Second Half', "Topics", 'Average Retweeters Followers First Half',
                   'Average Retweeters Followers Second Half', "Average retweeters followers count split by median",
                   offline_charts)


def essemble_dataset(filenames):
    df = pd.DataFrame()
    for filename in filenames:
        df_temp = pd.read_csv(filepath_or_buffer=filename, sep=",", engine=None)
        df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
        df_temp['timestamp'] = [i.replace(tzinfo=datetime.timezone.utc) for i in df_temp['timestamp']]
        df_temp = df_temp.sort_values(by='timestamp', ascending=True)
        df_temp = df_temp.drop(['index', 'Unnamed: 0'], axis=1)
        df = pd.concat([df, pd.DataFrame.from_records(df_temp)])
    return df.reset_index()


def get_avg_followers_retweets(df_retweets_info, og_tweet_id):
    matches = df_retweets_info[df_retweets_info['ref_tweed_id'] == og_tweet_id]
    if matches.shape[0] == 0:
        return -1
    return int(matches['followers'].mean())


def time_diff_from_original_old(og_tweet_time, retweet_time):
    return abs(og_tweet_time.replace(tzinfo=datetime.timezone.utc) - retweet_time.replace(
        tzinfo=datetime.timezone.utc)).total_seconds() / 3600.0


def time_diff_from_original(og_tweet_time, retweet_time):
    return abs(og_tweet_time - retweet_time).total_seconds() / 3600.0


def get_avg_retweets_time(df_retweets_info, og_tweet_time, og_tweet_id):
    matches = df_retweets_info[df_retweets_info['ref_tweed_id'] == og_tweet_id]
    if matches.shape[0] == 0:
        return -1
    t = [time_diff_from_original(og_tweet_time, retweet_time) for retweet_time in matches['timestamp']]
    return mean(t)


def get_avg_retweets_account_age_old(df_retweets_info, og_tweet_id):
    matches = df_retweets_info[df_retweets_info['ref_tweed_id'] == og_tweet_id]
    if matches.shape[0] == 0:
        return -1
    t = [(datetime.datetime.now().replace(tzinfo=datetime.timezone.utc) - created_at.replace(
        tzinfo=datetime.timezone.utc)) / np.timedelta64(1, 'Y') for created_at in matches['created_at']]
    return mean(t)


def get_avg_retweets_account_age(df_retweets_info, og_tweet_id):
    matches = df_retweets_info[df_retweets_info['ref_tweed_id'] == og_tweet_id]
    if matches.shape[0] == 0:
        return -1

    ages = pd.to_datetime(matches['created_at'], utc=True).dt.strftime("%Y-%m-%d")
    ages = pd.to_datetime(ages)
    ages = ages.apply(lambda x: relativedelta(datetime.datetime.now(), x).years)

    return ages.mean()


def get_retweets_half_time(df_retweets_info, og_tweet_time, og_tweet_id):
    matches = df_retweets_info[df_retweets_info['ref_tweed_id'] == og_tweet_id]
    if matches.shape[0] == 0:
        return -1
    fist_half_retweet_time = matches.iloc[int(len(matches) / 2)]['timestamp']
    fist_half_retweet_time_diff = time_diff_from_original(og_tweet_time, fist_half_retweet_time)
    return fist_half_retweet_time_diff


def get_retweets_total_time(df_retweets_info, og_tweet_time, og_tweet_id):
    matches = df_retweets_info[df_retweets_info['ref_tweed_id'] == og_tweet_id]
    if matches.shape[0] == 0:
        return -1
    total_retweet_time = matches.iloc[int(len(matches) - 1)]['timestamp']
    total_retweet_time_diff = time_diff_from_original(og_tweet_time, total_retweet_time)
    return total_retweet_time_diff


def get_retweets_followers_split(df_retweets_info, og_tweet_id):
    matches = df_retweets_info[df_retweets_info['ref_tweed_id'] == og_tweet_id]
    if matches.shape[0] == 0:
        return -1
    return matches[:int(len(matches) / 2)]['followers'].mean()


def get_retweets_followers_second_split(df_retweets_info, og_tweet_id):
    matches = df_retweets_info[df_retweets_info['ref_tweed_id'] == og_tweet_id]
    if matches.shape[0] == 0:
        return -1
    return matches[:int(len(matches) - 1)]['followers'].mean()


def retweeters_characteristics(df, df_retweets_info, min_retweets):
    df_final = pd.DataFrame()
    for y in np.sort(df['year'].unique()):

        dfy = df[df['year'] == y]
        cats_vals = []

        topics = dfy[~dfy['topics_cleaned'].isna()]['topics_cleaned'].unique()
        topics = sorted(topics, key=topics_categories.index)

        for topic in topics:
            tweets_by_topic = dfy[(dfy['topics_cleaned'] == topic) & (dfy['retweet_count'] > min_retweets)].copy()

            cat_vals = dict()
            cat_vals['Year'] = str(y)
            cat_vals['Topic'] = topic

            get_basic_analysis(tweets_by_topic, df_retweets_info, cat_vals)
            get_timing_analysis(tweets_by_topic, df_retweets_info, cat_vals)

            cats_vals.append(cat_vals)

        df_year = pd.DataFrame(cats_vals, columns=['Topic', 'Average Retweeters Followers', 'Average Retweets Time',
                                                   'Average Retweeters Account Age', 'Average Retweeters Half Time',
                                                   'Average Retweeters Total Time', '% Time to get 50% retweets',
                                                   'Average Retweeters Followers First Half',
                                                   'Average Retweeters Followers Second Half'])
        df_year['Year'] = [str(y) for _ in range(len(topics))]
        df_final = pd.concat([df_final, pd.DataFrame.from_records(df_year)])

    return df_final


def get_basic_analysis(df, df_retweets_info, topic_values):
    df['avg_retweeters_followers'] = [get_avg_followers_retweets(df_retweets_info, x) for x in zip(df['tweet_id'])]
    df['avg_retweeters_time'] = [get_avg_retweets_time(df_retweets_info, x, y) for x, y in
                                 zip(df['timestamp'], df['tweet_id'])]
    df['avg_retweeters_account_age'] = [get_avg_retweets_account_age(df_retweets_info, x) for x in df['tweet_id']]

    avg_retweeters_followers = df[df['avg_retweeters_followers'] != -1]['avg_retweeters_followers'].mean()
    avg_retweeters_time = df[df['avg_retweeters_time'] != -1]['avg_retweeters_time'].mean()
    avg_retweeters_account_age = df[df['avg_retweeters_account_age'] != -1]['avg_retweeters_account_age'].mean()

    if pd.isna(avg_retweeters_followers):
        avg_retweeters_followers = 0

    if pd.isna(avg_retweeters_time):
        avg_retweeters_time = 0

    if pd.isna(avg_retweeters_account_age):
        avg_retweeters_account_age = 0

    topic_values['Average Retweeters Followers'] = int(avg_retweeters_followers)
    topic_values['Average Retweets Time'] = int(avg_retweeters_time)
    topic_values['Average Retweeters Account Age'] = avg_retweeters_account_age


def get_timing_analysis(df, df_retweets_info, topic_values):
    df['first_half_time'] = [get_retweets_half_time(df_retweets_info, x, y) for x, y in zip(df['timestamp'],
                                                                                            df['tweet_id'])]
    df['total_time'] = [get_retweets_total_time(df_retweets_info, x, y) for x, y in zip(df['timestamp'],
                                                                                        df['tweet_id'])]
    df['first_half_avg_foll'] = [get_retweets_followers_split(df_retweets_info, x) for x in zip(df['tweet_id'])]
    df['second_half_avg_foll'] = [get_retweets_followers_second_split(df_retweets_info, x) for x in zip(df['tweet_id'])]

    avg_first_half_time = df[df['first_half_time'] != -1]['first_half_time'].mean()
    avg_total_time = df[df['total_time'] != -1]['total_time'].mean()
    avg_first_half_avg_foll = df[df['first_half_avg_foll'] != -1]['first_half_avg_foll'].mean()
    avg_second_half_avg_foll = df[df['second_half_avg_foll'] != -1]['second_half_avg_foll'].mean()

    if pd.isna(avg_first_half_time):
        avg_first_half_time = 0

    if pd.isna(avg_total_time):
        avg_total_time = 0

    if pd.isna(avg_first_half_avg_foll):
        avg_first_half_avg_foll = 0

    if pd.isna(avg_second_half_avg_foll):
        avg_second_half_avg_foll = 0

    topic_values['Average Retweeters Half Time'] = avg_first_half_time
    topic_values['Average Retweeters Total Time'] = avg_total_time
    if avg_first_half_time != 0 and avg_total_time != 0:
        topic_values['% Time to get 50% retweets'] = (np.round(avg_first_half_time / avg_total_time, 2)) * 100
    else:
        topic_values['% Time to get 50% retweets'] = 0
    topic_values['Average Retweeters Followers First Half'] = avg_first_half_avg_foll
    topic_values['Average Retweeters Followers Second Half'] = avg_second_half_avg_foll


def shared_tweets(df):
    df_test = pd.DataFrame()
    cols = ['topics_cleaned']
    cats_sort = topics_categories

    for y in np.sort(df['year'].unique()):
        dfy = df[df['year'] == y]
        df_all = dfy.groupby(cols).agg(
            **{"count " + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")},
            **{"retweets count": pd.NamedAgg(column="retweet_count", aggfunc="count")},
            **{"retweets mean": pd.NamedAgg(column="retweet_count", aggfunc="mean")},
            **{"followers": pd.NamedAgg(column="followers", aggfunc="mean")},
        )
        df_rets = dfy[dfy['retweet_count'] > 0].groupby(cols).agg(
            **{"count " + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")},
            **{"followers": pd.NamedAgg(column="followers", aggfunc="mean")})
        df_likes = dfy[dfy['like_count'] > 0].groupby(cols).agg(
            **{"count " + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")},
            **{"followers": pd.NamedAgg(column="followers", aggfunc="mean")})

        df_all = df_all.reindex(cats_sort).reset_index()
        df_rets = df_rets.reindex(cats_sort).reset_index()
        df_likes = df_likes.reindex(cats_sort).reset_index()

        df_all['year'] = [str(y) for _ in range(len(cats_sort))]
        df_all['sum'] = [df_all["count " + cols[0]].sum() for _ in range(df_all.shape[0])]
        df_all['% ' + cols[0]] = (df_all["count " + cols[0]] / df_all['sum']) * 100
        df_all['% with retweets'] = np.round((df_rets["count " + cols[0]] / df_all["count " + cols[0]]) * 100, 2)
        df_all['% with likes'] = np.round((df_likes["count " + cols[0]] / df_all["count " + cols[0]]) * 100, 2)

        df_all['followers mean'] = df_all['followers']
        df_all['shared followers mean'] = df_rets['followers']

        df_test = pd.concat([df_test, pd.DataFrame.from_records(df_all)])

    return df_test[
        ['year', cols[0], "count " + cols[0], '% ' + cols[0], '% with retweets', '% with likes', 'followers mean',
         'shared followers mean']]


def shared_chart(df):
    fig = px.bar(df, x="topics_cleaned", y="shared followers mean", color="year", color_discrete_sequence=palette,
                 barmode="group",
                 title="Average followers between shared tweets and not shared by topic", width=900, height=500)
    fig.update_yaxes(title_text="Average followers count")
    fig.show()


def retweeters_info_chart(df, x_col_categories, x_col, y_col, title, offline):
    fig = px.bar(df, x=x_col, y=y_col, color="Year", color_discrete_sequence=palette, barmode="group",
                 category_orders={x_col: x_col_categories,
                                  'year': df['Year'].unique()})
    fig.update_layout(title=title)
    fig.show()
    if offline:
        plotly.offline.plot(fig, filename='../../data/charts/' + title + '.html')


def analysis_chart(df, x_col, y_bar, y_line, x_name, y_bar_name, y_line_name, title, offline):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    years = np.sort(df['Year'].unique())
    years_count = len(df['Year'].unique())

    for i in range(years_count):
        year = years[i]
        dfy = df[df['Year'] == year]
        fig.add_trace(go.Bar(x=dfy[x_col], y=dfy[y_bar], text=list(map(str, dfy[y_bar].tolist())),
                             name=y_bar_name + ' ' + str(year), marker_color=palette[i], width=0.28,
                             textposition='inside'), secondary_y=False)
        fig.add_trace(go.Scatter(x=dfy[x_col], y=dfy[y_line], name=y_line_name + ' ' + str(year),
                                 marker_color=palette[i + years_count]), secondary_y=True)

    fig.update_yaxes(title_text=y_line_name, secondary_y=True)
    fig.update_yaxes(title_text=y_bar_name, secondary_y=False)
    fig.update_layout(title_text=title, width=1100, height=500)
    fig.update_xaxes(title_text=x_name)
    fig.show()
    if offline:
        plotly.offline.plot(fig, filename='../../data/charts/' + title + '.html')


def time_analysis_charts(df, title):
    fig = px.bar(df, x="Topic", y='% Time to get 50% retweets', color="Year", barmode="group",
                 color_discrete_sequence=palette,
                 category_orders={"Topic": topics_categories,
                                  'year': df['year'].unique()})
    fig.update_layout(title_text=title, width=950, height=500)
    fig.show()
