import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from statistics import mean
import seaborn as sns

sns.set()

CHARTS_PATH = '../data/charts/retweets_analysis/'

topics_categories = ['Brand', 'Holiday', 'Person', 'Interest and Hobbies', 'Sport', 'TV and Movies', 'Other',
                     'Video Game', 'Entities', 'Political', 'Music', 'Book', 'News']
palette = ['#006D77', '#FBD1A2', '#7DCFB6', '#E29578', '#254E70', '#C33C54', '#FFDDD2', '#004346', '#faf3dd', '#00B2CA',
           '#37718E', '#aed9e0', '#09BC8A']
offline_charts = True


def retweets_analysis(filenames: list, retweets_filenames: list):
    print("Starting retweeters analysis")
    df = ensemble_dataset(filenames)
    df_retweets_info = ensemble_dataset(retweets_filenames)

    df_analysis = shared_tweets(df, 'topics_cleaned', topics_categories)
    analysis_chart(df_analysis, "topics_cleaned", 'shared followers mean', "not shared followers mean", "Tópicos",
                   'Média de seguidores em tweets partilhados', 'Média de seguidores em tweets não partilhados',
                   "Média de seguidores entre tweets partilhados e não partilhados por tópico", offline_charts)

    print("Generating analysis tables")
    df_retweeters_chars = retweeters_characteristics(df, df_retweets_info, 0)
    df_popular_retweeters_chars = retweeters_characteristics(df, df_retweets_info, 10)

    print("Initializing retweeters analysis")
    # Analysing average followers count of retweeters per topic
    retweeters_info_chart(df_retweeters_chars, topics_categories, "Topic", 'Average Retweeters Followers', 'Tópicos',
                          'Média de seguidores', "Média de seguidores dos retweeters por tópico", offline_charts)

    # Analysing average retweet time ratio of retweeters per topic
    retweeters_info_chart(df_retweeters_chars, topics_categories, "Topic", 'Average total time retweets', 'Tópicos',
                          "Média de dias", "Média de dias para obter todos os retweets por tópico", offline_charts)

    # Analysing average retweeter’s account age per topic
    retweeters_info_chart(df_retweeters_chars, topics_categories, "Topic", 'Average Retweeters Account Age', 'Tópicos',
                          "Média de antiguidade", "Média de antiguidade dos retweeters por tópico", offline_charts)

    print("Initializing popular tweets, retweeters analysis")
    print("Popular retweets (>10) account for only:", (df[df['retweet_count'] > 10].shape[0] /
                                                       df[df['retweet_count'] > 0].shape[0]) * 100, "% of the shared "
                                                                                                    "tweets")

    analysis_chart(df_popular_retweeters_chars, "Topic", 'Average half time retweets',
                   'Average second half time retweets', "Topics",
                   'Média de dias para obter os primeiros 50% dos retweets',
                   'Média de dias para obter os restantes retweets', "Média de dias para obter os retweets dividindo "
                                                                     "pela mediana e tópicos", offline_charts)

    analysis_chart(df_popular_retweeters_chars, "Topic", 'Average Retweeters Followers First Half',
                   'Average Retweeters Followers Second Half', "Topics", 'Média de seguidores para os primeiros 50% '
                                                                         'retweets',
                   'Média de seguidores para os restantes 50% retweets', "Média de seguidores dos retweeters dividindo "
                                                                         "pela mediana de retweets e tópicos",
                   offline_charts)


def ensemble_dataset(filenames):
    df = pd.DataFrame()
    for filename in filenames:
        df_temp = pd.read_csv(filepath_or_buffer=filename, sep=",", engine=None)
        df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
        df_temp['timestamp'] = [i.replace(tzinfo=datetime.timezone.utc) for i in df_temp['timestamp']]
        df_temp = df_temp.sort_values(by='timestamp', ascending=True)
        df_temp = df_temp.drop(['index', 'Unnamed: 0'], axis=1)
        df = pd.concat([df, pd.DataFrame.from_records(df_temp)])
    return df.reset_index()


def get_avg_followers_retweets(df, df_retweets_info):
    df_tmp = df.rename({'tweet_id': 'match_id'}, axis=1)
    df_retweets_info_tmp = df_retweets_info.rename({'ref_tweed_id': 'match_id'}, axis=1)
    res = pd.merge(df_tmp[['match_id']], df_retweets_info_tmp[['match_id', 'followers']], how='inner', on=['match_id'])
    return res['followers'].mean()


def time_diff_from_original_in_days(og_tweet_time, retweet_time):
    return (abs(og_tweet_time - retweet_time).total_seconds() / 3600.0) / 24


def get_time_to_get_retweets(df_retweets_info, og_tweet_time, og_tweet_id):
    matches = df_retweets_info[df_retweets_info['ref_tweed_id'] == og_tweet_id]
    size = matches.shape[0]
    if size == 0:
        return -1, -1, -1

    time_for_all = time_diff_from_original_in_days(og_tweet_time, matches['timestamp'].iloc[size - 1])
    if size < 2:
        time_for_first_half = time_for_all
        time_for_second_half = time_for_all
    else:
        time_for_first_half = time_diff_from_original_in_days(og_tweet_time, matches['timestamp'].iloc[int(size / 2)])
        time_for_second_half = time_diff_from_original_in_days(matches['timestamp'].iloc[int(size / 2)],
                                                               matches['timestamp'].iloc[int(size - 1)])
    return time_for_first_half, time_for_second_half, time_for_all


def get_avg_retweets_account_age(df, df_retweets_info):
    df_tmp = df.rename({'tweet_id': 'match_id'}, axis=1)
    df_retweets_info_tmp = df_retweets_info.rename({'ref_tweed_id': 'match_id'}, axis=1)
    res = pd.merge(df_tmp[['match_id']], df_retweets_info_tmp[['match_id', 'seniority']], how='inner', on=['match_id'])
    return res['seniority'].mean()


def get_retweets_followers_split(df_retweets_info, og_tweet_id):
    matches = df_retweets_info[df_retweets_info['ref_tweed_id'] == og_tweet_id]
    size = matches.shape[0]
    if size == 0:
        return -1, -1
    if size == 1:
        return matches['followers'].mean(), matches['followers'].mean()
    return matches[:int(size / 2)]['followers'].mean(), matches[int(size / 2):size]['followers'].mean()


def get_retweets_followers_mean_by_median(df, df_retweets_info):
    first_half_avg_foll, second_half_avg_foll = [], []
    for x in df['tweet_id']:
        first, second = get_retweets_followers_split(df_retweets_info, x)
        first_half_avg_foll.append(first)
        second_half_avg_foll.append(second)
    df['first_half_avg_foll'] = first_half_avg_foll
    df['second_half_avg_foll'] = second_half_avg_foll


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
            cats_vals.append(cat_vals)

        df_year = pd.DataFrame(cats_vals, columns=['Topic',
                                                   'Average Retweeters Followers',
                                                   'Average Retweeters Followers First Half',
                                                   'Average Retweeters Followers Second Half',
                                                   'Average half time retweets',
                                                   'Average second half time retweets',
                                                   'Average total time retweets',
                                                   '% Time(D) to get 50% retweets',
                                                   'Average Retweeters Account Age'])
        df_year['Year'] = [str(y) for _ in range(len(topics))]
        df_final = pd.concat([df_final, pd.DataFrame.from_records(df_year)])

    return df_final[['Year', 'Topic', 'Average Retweeters Followers', 'Average Retweeters Followers First Half',
                     'Average Retweeters Followers Second Half', 'Average half time retweets',
                     'Average second half time retweets', 'Average total time retweets',
                     '% Time(D) to get 50% retweets', 'Average Retweeters Account Age']]


def get_basic_analysis(df, df_retweets_info, topic_values):
    first_half_mean, second_half_mean, total_time_mean, perc_to_half = get_timing_analysis(df, df_retweets_info)

    avg_retweeters_followers = get_avg_followers_retweets(df, df_retweets_info)
    avg_retweeters_account_age = get_avg_retweets_account_age(df, df_retweets_info)
    get_retweets_followers_mean_by_median(df, df_retweets_info)

    avg_first_half_avg_foll = df[df['first_half_avg_foll'] != -1]['first_half_avg_foll'].mean()
    avg_second_half_avg_foll = df[df['second_half_avg_foll'] != -1]['second_half_avg_foll'].mean()

    if pd.isna(avg_retweeters_followers):
        avg_retweeters_followers = 0

    if pd.isna(avg_retweeters_account_age):
        avg_retweeters_account_age = 0

    if pd.isna(avg_first_half_avg_foll):
        avg_first_half_avg_foll = 0

    if pd.isna(avg_second_half_avg_foll):
        avg_second_half_avg_foll = 0

    topic_values['Average Retweeters Followers'] = np.round(avg_retweeters_followers, 2)
    topic_values['Average Retweeters Followers First Half'] = np.round(avg_first_half_avg_foll, 2)
    topic_values['Average Retweeters Followers Second Half'] = np.round(avg_second_half_avg_foll, 2)

    topic_values['Average half time retweets'] = np.round(first_half_mean, 2)
    topic_values['Average second half time retweets'] = np.round(second_half_mean, 2)
    topic_values['Average total time retweets'] = np.round(total_time_mean, 2)
    topic_values['% Time(D) to get 50% retweets'] = np.round(perc_to_half, 2)

    topic_values['Average Retweeters Account Age'] = np.round(avg_retweeters_account_age, 2)


def get_timing_analysis(df, df_retweets_info):
    first_half_times, second_half_times, total_times = [], [], []
    for og_tweet_time, og_tweet_id in zip(df['timestamp'], df['tweet_id']):
        first_half, second_half, total = get_time_to_get_retweets(df_retweets_info, og_tweet_time, og_tweet_id)
        if first_half != -1 and total != -1:
            first_half_times.append(first_half)
            second_half_times.append(second_half)
            total_times.append(total)

    if len(first_half_times) == 0 and len(second_half_times) == 0 and len(total_times) == 0:
        return 0, 0, 0, 0

    return mean(first_half_times), mean(second_half_times), mean(total_times), \
           (np.round(mean(first_half_times) / mean(total_times), 2)) * 100


def shared_tweets(source_df, cols, cats_sort):
    df = pd.DataFrame()
    for y in np.sort(source_df['year'].unique()):
        dfy = source_df[source_df['year'] == y]

        df_all = dfy.groupby(cols).agg(**{"followers": pd.NamedAgg(column="followers", aggfunc="mean")})
        df_shared = dfy[dfy['retweet_count'] > 0].groupby(cols).agg(**{"followers": pd.NamedAgg(column="followers",
                                                                                                aggfunc="mean")})
        df_not_shared = dfy[dfy['retweet_count'] == 0].groupby(cols).agg(**{"followers": pd.NamedAgg(column="followers",
                                                                                                     aggfunc="mean")})

        df_all = df_all.reindex(cats_sort).reset_index()
        df_rets = df_shared.reindex(cats_sort).reset_index()
        df_not_shared = df_not_shared.reindex(cats_sort).reset_index()

        df_all['Year'] = [str(y) for _ in range(len(cats_sort))]
        df_all['not shared followers mean'] = df_not_shared['followers']
        df_all['followers mean'] = df_all['followers']
        df_all['shared followers mean'] = df_rets['followers']

        df = pd.concat([df, pd.DataFrame.from_records(df_all)])

    return df[['Year', cols, 'not shared followers mean', 'followers mean', 'shared followers mean']]


def retweeters_info_chart(df, x_col_categories, x_col, y_col, x_name, y_name, title, offline):
    fig = px.bar(df, x=x_col, y=y_col, color="Year", color_discrete_sequence=palette, barmode="group",
                 category_orders={x_col: x_col_categories,
                                  'year': df['Year'].unique()})
    fig.update_xaxes(title_text=x_name)
    fig.update_yaxes(title_text=y_name)
    fig.update_layout(title=title, width=1200, height=500)
    fig.show()
    if offline:
        plotly.offline.plot(fig, filename=CHARTS_PATH + title + '.html')


def analysis_chart(df, x_col, y_bar, y_line, x_name, y_bar_name, y_line_name, title, offline):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    years = np.sort(df['Year'].unique())
    years_count = len(df['Year'].unique())

    for i in range(years_count):
        year = years[i]
        dfy = df[df['Year'] == year]
        fig.add_trace(go.Bar(x=dfy[x_col], y=dfy[y_bar], name=y_bar_name + ' ' + str(year), marker_color=palette[i],
                             width=0.28, textposition='inside'), secondary_y=False)
        fig.add_trace(go.Scatter(x=dfy[x_col], y=dfy[y_line], name=y_line_name + ' ' + str(year),
                                 marker_color=palette[i + years_count]), secondary_y=True)

    fig.update_yaxes(title_text=y_line_name, secondary_y=True)
    fig.update_yaxes(title_text=y_bar_name, secondary_y=False)
    fig.update_layout(title_text=title, width=1200, height=500)
    fig.update_xaxes(title_text=x_name)
    fig.show()
    if offline:
        plotly.offline.plot(fig, filename=CHARTS_PATH + title + '.html')
