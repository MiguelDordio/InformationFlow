import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns

sns.set()

CHARTS_PATH = '../data/charts/tweets_analysis/'

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
          'December']
week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_phases = ['Morning', 'Afternoon', 'Dusk', 'Night', 'Middle of the night']
sentiments = ['Negative', 'Neutral', 'Positive']
hashtags = [True, False]
offline_graphs = True
palette = ['#006D77', '#FBD1A2', '#7DCFB6', '#E29578', '#254E70', '#C33C54', '#FFDDD2', '#004346', '#faf3dd', '#00B2CA',
           '#37718E', '#aed9e0', '#09BC8A']


def analysis(filenames: list):
    print("Starting general analysis")
    df = ensemble_dataset(filenames)

    get_data_characteristics(df)

    topics_categories = df['topics_cleaned'].unique()[1:]
    tweet_analysis = df[['text', 'year', 'day_phase', 'day_of_week', 'month', 'retweet_count', 'quote_count',
                         'like_count', 'reply_count', 'sentiment', 'hashtags', 'topics_cleaned', 'reach']]

    # Average retweet and like count per phase of the day
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['day_phase'], day_phases)
    analysis_chart(df_analysis, 'day_phase', '% with retweets', '% with likes', 'Fases do dia', '% tweets com retweets',
                   '% tweets com gostos', 'Percentagem de tweets com retweets e gostos durante o dia', offline_graphs)
    analysis_chart(df_analysis, 'day_phase', 'retweets mean', 'likes mean', 'Fases do dia', 'Média de retweets',
                   'Média de gostos', 'Média de retweets e gostos durante o dia', offline_graphs)

    # Average retweet and like count during the week
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['day_of_week'], week_days)
    analysis_chart(df_analysis, 'day_of_week', '% with retweets', '% with likes', 'Dias da semana',
                   '% tweets com retweets', '% tweets com gostos', 'Percentagem de tweets com retweets e gostos durante'
                                                                   ' a semana', offline_graphs)
    analysis_chart(df_analysis, 'day_of_week', 'retweets mean', 'likes mean', 'Dias da semana', 'Média de retweets',
                   'Média de gostos', 'Média de retweets e gostos durante a semana', offline_graphs)

    # Average retweet count per month
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['month'], months)
    analysis_chart(df_analysis, 'month', '% with retweets', '% with likes', 'Meses', '% tweets com retweets',
                   '% tweets com gostos', 'Percentagem de tweets com retweets e gostos durante o ano', offline_graphs)
    analysis_chart(df_analysis, 'month', 'retweets mean', 'likes mean', 'Meses', 'Média de retweets',
                   'Média de gostos', 'Média de retweets e gostos durante o ano', offline_graphs)

    # Tweets performance by sentiment
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['sentiment'], sentiments)
    analysis_chart(df_analysis, 'sentiment', '% with retweets', '% with likes', 'Sentimentos', '% tweets com retweets',
                   '% tweets com gostos', 'Percentagem de tweets com retweets e gostos por sentimento', offline_graphs)
    analysis_chart(df_analysis, 'sentiment', 'retweets mean', 'likes mean', 'Sentimentos', 'Média de retweets',
                   'Média de gostos', 'Média de retweets e likes por sentimento', offline_graphs)

    # Tweets performance by topic
    print("Analyzing tweets performance by topic")

    # Performance of each topic in retweets and likes
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['topics_cleaned'], topics_categories)
    analysis_chart(df_analysis, 'topics_cleaned', '% with retweets', '% with likes', 'Tópicos', '% tweets com retweets',
                   '% tweets com gostos', 'Percentagem de tweets com retweets e gostos por tópicos', offline_graphs)
    analysis_chart(df_analysis, 'topics_cleaned', 'retweets mean', 'likes mean', 'Tópicos', 'Média de retweets',
                   'Média de gostos', 'Média de retweets e gostos por tópicos', offline_graphs)

    # Average retweet count per topic during the day
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['day_phase', 'topics_cleaned'], day_phases)
    multi_label_chart(df_analysis, "topics_cleaned", day_phases, "day_phase", "% with retweets", "Fase do dia",
                      '% tweets com retweets', "Percentagem de tweets com retweets por tópico durante o dia",
                      offline_graphs)

    # Average retweet count per topic during the week
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['day_of_week', 'topics_cleaned'], week_days)
    multi_label_chart(df_analysis, "topics_cleaned", week_days, "day_of_week", "% with retweets", "Dia da semana",
                      '% tweets com retweets', "Percentagem de tweets com retweets por tópico durante a semana",
                      offline_graphs)

    # Average retweet count per topic during the year
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['month', 'topics_cleaned'], months)
    multi_label_chart(df_analysis, "topics_cleaned", months, "month", "% with retweets", "Meses",
                      '% tweets com retweets', "Percentagem de tweets com retweets por tópico durante o ano",
                      offline_graphs)

    # Average tweet reach per topic
    df_analysis = reach_by_topic(tweet_analysis, ['topics_cleaned'], topics_categories)
    multi_label_chart(df_analysis, 'topics_cleaned', topics_categories, 'topics_cleaned', 'reach mean', 'Tópicos',
                      'Média de alcance (nº pessoas)', 'Alcance médio dos tweets por tópico', offline_graphs)

    # Impact of hashtags in topic popularity
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['hashtags', 'topics_cleaned'], hashtags)
    multi_label_chart(df_analysis, "topics_cleaned", topics_categories, "hashtags", "% with retweets", "Hashtags",
                      '% tweets com retweets', 'Presença de hashtags por tópico e correspondente percentagem de tweets '
                                               'com retweets', offline_graphs)

    # Tweet sentiment per topic
    df_analysis = retweets_likes_info_by_year(tweet_analysis, ['sentiment', 'topics_cleaned'], sentiments)
    multi_label_chart(df_analysis, "topics_cleaned", topics_categories, "sentiment", "% with retweets", "Sentimento",
                      '% tweets com retweets', "Sentimeto dos tweets por tópicos e correspondente percentagem de tweets"
                                               " com retweets", offline_graphs)


def ensemble_dataset(filenames):
    df = pd.DataFrame()
    for filename in filenames:
        df_temp = pd.read_csv(filepath_or_buffer=filename, sep=",", engine=None)
        df_temp = df_temp.sort_values(by='timestamp', ascending=True)
        df_temp = df_temp.drop(['index', 'Unnamed: 0'], axis=1)
        df = pd.concat([df, pd.DataFrame.from_records(df_temp)])
    return df.reset_index()


def get_data_characteristics(df):
    print("\nData characteristics")
    print("Average nº of tweets per day of the week:", int(df['day_of_week'].value_counts().mean()))
    print("Nº of tweets with known topics {:d} and are {:0.2f}% of the data".format(
        df[~df['topics_cleaned'].isna()].shape[0],
        (df[~df['topics_cleaned'].isna()].shape[0] / df.shape[0]) * 100))
    print("Nº of tweets do collect retweeters information:",
          df[(~df['topics_cleaned'].isna()) & df['retweet_count'] > 0].shape[0])
    print("{:0.2f}% have retweets\n".format((df[df['retweet_count'] > 0].shape[0] / df.shape[0]) * 100))


def retweets_likes_info_by_year(source_df, cols, cats_sort):
    df = pd.DataFrame()
    for y in np.sort(source_df['year'].unique()):
        dfy = source_df[source_df['year'] == y]
        df_all = dfy.groupby(cols).agg(
            **{"count " + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")},
            **{"retweets mean": pd.NamedAgg(column="retweet_count", aggfunc="mean")},
            **{"likes mean": pd.NamedAgg(column="like_count", aggfunc="mean")}).round(2)

        df_rets = dfy[dfy['retweet_count'] > 0].groupby(cols).agg(
            **{"count " + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")})
        df_likes = dfy[dfy['like_count'] > 0].groupby(cols).agg(
            **{"count " + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")})

        if len(cols) == 1:
            df_all = df_all.reindex(cats_sort).reset_index()
            df_rets = df_rets.reindex(cats_sort).reset_index()
            df_likes = df_likes.reindex(cats_sort).reset_index()
            df_all['year'] = [y for _ in range(len(cats_sort))]
            df_all['% with retweets'] = np.round((df_rets["count " + cols[0]] / df_all["count " + cols[0]]) * 100, 2)
            df_all['% with likes'] = np.round((df_likes["count " + cols[0]] / df_all["count " + cols[0]]) * 100, 2)
        else:
            df_all = df_all.reset_index()
            df_rets = df_rets.reset_index()
            df_likes = df_likes.reset_index()

            year_items = []
            for cat in cats_sort:
                count = len(df_all[df_all[cols[0]] == cat][cols[1]].unique())
                year_items += [y for _ in range(count)]

            df_all['year'] = year_items

            filter_rets = df_all.merge(df_rets, on=[cols[0], cols[1]])
            df_all['% with retweets'] = np.round(
                (filter_rets["count " + cols[0] + "_y"] / filter_rets["count " + cols[0] + "_x"]) * 100, 2)

            filter_likes = df_all.merge(df_likes, on=[cols[0], cols[1]])
            df_all['% with likes'] = np.round(
                (filter_likes["count " + cols[0] + "_y"] / filter_likes["count " + cols[0] + "_x"]) * 100, 2)

        df_all['sum'] = [df_all["count " + cols[0]].sum() for _ in range(df_all.shape[0])]
        df_all['% ' + cols[0]] = (df_all["count " + cols[0]] / df_all['sum']) * 100

        df = pd.concat([df, pd.DataFrame.from_records(df_all)])

    if len(cols) == 1:
        return df[
            ['year', cols[0], "count " + cols[0], '% ' + cols[0], '% with retweets', '% with likes', 'retweets mean',
             'likes mean']]
    else:
        return df[['year', cols[0], cols[1], "count " + cols[0], '% ' + cols[0], '% with retweets', '% with likes',
                   'retweets mean', 'likes mean']]


def reach_by_topic(source_df, cols, cats_sort):
    df = pd.DataFrame()
    for y in np.sort(source_df['year'].unique()):
        dfy = source_df[source_df['year'] == y]
        df_all = dfy.groupby(cols).agg(
            **{"count " + cols[0]: pd.NamedAgg(column=cols[0], aggfunc="count")},
            **{"reach mean": pd.NamedAgg(column="reach", aggfunc="mean")}).round(2)

        df_all = df_all.reindex(cats_sort).reset_index()
        df_all['year'] = [y for _ in range(len(cats_sort))]

        df_all['sum'] = [df_all["count " + cols[0]].sum() for _ in range(df_all.shape[0])]
        df_all['% ' + cols[0]] = (df_all["count " + cols[0]] / df_all['sum']) * 100

        df = pd.concat([df, pd.DataFrame.from_records(df_all)])

    return df[['year', cols[0], "count " + cols[0], '% ' + cols[0], 'reach mean']]


def analysis_chart(df, x_col, y_bar, y_line, x_name, y_bar_name, y_line_name, title, offline):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    years = np.sort(df['year'].unique())
    years_count = len(df['year'].unique())

    for i in range(years_count):
        year = years[i]
        dfy = df[df['year'] == year]
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


def multi_label_chart(df, category_label, categories, x_col, y_col, x_name, y_name, title, offline):
    df['year'] = df['year'].astype(str)
    fig = px.bar(df, x=x_col, y=y_col, color=category_label, barmode="stack", text=y_col, facet_col="year",
                 color_discrete_sequence=palette, category_orders={x_col: categories, 'year': df['year'].unique()})

    fig.update_xaxes(title_text=x_name)
    fig.update_yaxes(title_text=y_name)
    fig.update_layout(title_text=title, width=1200, height=500)

    fig.show()
    if offline:
        plotly.offline.plot(fig, filename=CHARTS_PATH + title + '.html')
