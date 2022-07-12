import pandas as pd
import numpy as np
import json
import datetime
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from dateutil.relativedelta import relativedelta


FINAL_DATASET = "../data/processed_tweets/tweets_"
FINAL_DATASET_RETWEETS_INFO = "../data/processed_retweets/retweets_info_"


def load_datasets(tweets_path, user_path):
    print("(1/10) - Loading datasets")
    df_tweets = pd.read_csv(filepath_or_buffer=tweets_path, sep=",")
    df_users = pd.read_csv(filepath_or_buffer=user_path, sep=",")

    df_tweets = df_tweets.sort_values(by='timestamp', ascending=True).reset_index()
    df_users = df_users.drop_duplicates(subset=['user_id'], keep="first")

    return df_tweets, df_users


def get_tweet_reach(df_tweets, df_users, retweets_path, retweeters_path):
    print("(2/10) - Preparing to calculate tweet reach")
    df_retweeters = pd.read_csv(filepath_or_buffer=retweeters_path, sep=",")
    df_retweets = pd.read_csv(filepath_or_buffer=retweets_path, sep=",")

    # remove duplicated retweeters
    df_retweeters = df_retweeters.drop_duplicates(subset=['user_id'], keep="first")

    # sort by timestamp
    df_retweets = df_retweets.sort_values(by='timestamp', ascending=True).reset_index()

    # merge retweeters datasets
    df_retweets_info = pd.merge(df_retweets[['tweet_id', 'text', 'timestamp', 'user_id', 'like_count', 'retweet_count',
                                             'quote_count', 'reply_count', 'referenced_tweets']],
                                df_retweeters[['user_id', 'followers', 'following', 'tweet_count', 'verified',
                                               'created_at']], on="user_id").reset_index()

    # remove NANs
    df_retweets_info = df_retweets_info[df_retweets_info['referenced_tweets'].notna()]

    df_retweets_info['ref_tweed_id'] = [get_ref_tweet_id(ref_tweet) for ref_tweet in
                                        df_retweets_info['referenced_tweets']]

    df_tweets['ref_tweed_id'] = [get_ref_tweet_id(ref_tweet) for ref_tweet in df_tweets['referenced_tweets']]
    # converting the reference ids that exist to int
    df_tweets['ref_tweed_id'] = pd.to_numeric(df_tweets['ref_tweed_id'], errors='coerce').fillna(0).astype(np.int64)

    print("         Calculating tweet reach\n")
    df_tweets['reach'] = [calculate_tweet_reach(row[0], row[1], row[2], row[3], row[4], df_users, df_retweets_info)
                          for row in zip(df_tweets['tweet_id'], df_tweets['user_id'], df_tweets['topics'],
                                         df_tweets['retweet_count'], df_tweets['ref_tweed_id'])]

    return df_retweets_info


def get_ref_tweet_id(ref_tweets):
    g = str(ref_tweets)
    if isinstance(g, str) and g != 'nan' and not pd.isna(g):
        return int(g.split()[1].replace(',', ''))
    else:
        return None


def calculate_tweet_reach(tweet_id, user_id, topics, retweets_count, ref_tweed_id, df_users, df_retweets_info):
    total_reach = 0
    total_reach += df_users[df_users['user_id'] == user_id]['followers'].sum()
    if topics != '[]' and retweets_count > 0:
        if ref_tweed_id is not None and ref_tweed_id != 0:
            total_reach += df_retweets_info[(df_retweets_info['ref_tweed_id'] == tweet_id) |
                                            (df_retweets_info['ref_tweed_id'] == ref_tweed_id)]['followers'].sum()
        else:
            total_reach += df_retweets_info[df_retweets_info['ref_tweed_id'] == tweet_id]['followers'].sum()
    return total_reach


def process_topics(df_tweets):
    print("(3/10) - Doing topics cleaning")
    df_tweets['topics'] = [process_topic(topics) for topics in df_tweets['topics']]
    df_tweets['topics_ids'] = [process_topic(topics_ids) for topics_ids in df_tweets['topics_ids']]

    print("         Topics: group topics by category")
    df_tweets['topics_cleaned'] = [group_topics(topic) for topic in df_tweets['topics']]

    print("         Topics: removing NaNs and int conversion\n")
    df_tweets['topics_ids'].fillna(value=-1, inplace=True)
    df_tweets['topics_ids'] = df_tweets['topics_ids'].apply(lambda x: int(x))
    df_tweets['topics'].fillna(value="", inplace=True)


def process_topic(topics: str):
    s = topics
    for i in range(topics.count('\'')):
        s = s.replace('\'', '"')
    t = json.loads(s)
    if len(t) > 0:
        return t[0]
    return None


def group_topics(topic):
    if pd.isna(topic):
        return None

    if not isinstance(topic, str):
        print(topic)

    if 'Brand' in topic or 'Product' in topic:
        return 'Brand'
    elif 'Person' in topic:
        return 'Person'
    elif 'Sport' in topic or 'Athlete' in topic or 'Coach' in topic or 'Hockey' in topic or 'Football' in topic or \
            'NFL' in topic:
        return 'Sport'
    elif 'TV' in topic or 'Movie' in topic or 'Award' in topic or 'Actor' in topic or 'Fictional Character' in topic \
            or 'Entertainment' in topic:
        return 'TV and Movies'
    elif 'Music' in topic or 'Musician' in topic or 'Concert' in topic or 'Song' in topic or 'Radio' in topic:
        return 'Music'
    elif 'Book' in topic:
        return 'Book'
    elif 'Hobbies' in topic:
        return 'Interest and Hobbies'
    elif 'Video Game' in topic or 'Esports' in topic or 'eSport' in topic:
        return 'Video Game'
    elif 'Political' in topic or 'Politicians' in topic:
        return 'Political'
    elif 'Holiday' in topic:
        return 'Holiday'
    elif 'News' in topic:
        return 'News'
    elif 'Entities' in topic:
        return 'Entities'
    else:
        return 'Other'


def sentiment_classification(df_tweets):
    print("(4/10) - Doing sentiment classification\n")
    sid_obj = SentimentIntensityAnalyzer()
    df_tweets['sentiment'] = [sentiment_scores(tweet_text, False, sid_obj) for tweet_text in df_tweets['text']]


def sentiment_scores(sentence, prints, sid_obj):
    if prints:
        print("\nSentence:", sentence)

    sentiment_dict = sid_obj.polarity_scores(sentence)

    if sentiment_dict['compound'] >= 0.05:
        if prints:
            print("Positive")
        return "Positive"
    elif sentiment_dict['compound'] <= - 0.05:
        if prints:
            print("Positive")
        return "Negative"
    else:
        if prints:
            print("Neutral")
        return "Neutral"


def hashtags(df):
    print("(5/10) - Checking hashtags presence\n")
    df['hashtags'] = [has_hashtags(text) for text in df['text']]


def has_hashtags(text):
    return len(re.findall(r'\B#\w*[a-zA-Z]+\w*', text)) > 0


def popularity(df):
    print("(6/10) - Popularity classification\n")
    df['popularity'] = [0 if retweets == 0 else 1 for retweets in df['retweet_count']]


def tweet_popularity_label(retweet_count, quote_count):
    if retweet_count > 0 or quote_count > 0:
        return 1
    else:
        return 0


def merge_tweets_and_users(df_tweets, df_users):
    print("(7/10) - Merging tweets and corresponding users\n")
    df = pd.merge(df_tweets[
                      ['tweet_id', 'text', 'timestamp', 'user_id', 'like_count', 'retweet_count',
                       'quote_count', 'reply_count', 'reach', 'topics_ids', 'topics', 'topics_cleaned',
                       'sentiment', 'hashtags', 'popularity']],
                  df_users[['user_id', 'followers', 'following', 'tweet_count', 'verified', 'created_at']],
                  on="user_id").reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_day_phase(hour):
    if 0 <= hour < 7:
        return "Middle of the night"
    elif 7 <= hour < 13:
        return "Morning"
    elif 13 <= hour < 16:
        return "Afternoon"
    elif 16 <= hour < 20:
        return "Dusk"
    elif 20 <= hour < 24:
        return "Night"


def time_phases_transformation(df, df_retweeters):
    print("(8/10) - Doing variables transformation")
    df['year'] = df['timestamp'].apply(lambda x: x.year)
    df['month'] = df['timestamp'].apply(lambda x: x.strftime('%B'))
    df['day_of_week'] = df['timestamp'].apply(lambda x: x.strftime('%A'))
    df['day_phase'] = df['timestamp'].apply(lambda x: get_day_phase(int(x.hour)))
    df['week_idx'] = df['timestamp'].apply(lambda x: '%s-%s' % (x.year, '{:02d}'.format(x.isocalendar()[1])))
    time_phases_encoding(df)
    get_users_seniority(df, df_retweeters)


def time_phases_encoding(df):
    print("         Doing variables encoding")
    cols_to_transform = ['day_phase', 'day_of_week', 'month', 'year', 'sentiment', 'verified', 'hashtags']

    one_hot_encoder(df, cols_to_transform)

    for col in cols_to_transform:
        enc = LabelEncoder()
        enc.fit(df[col])
        df[col + '_enc'] = enc.transform(df[col])
    return df


def one_hot_encoder(df, cols_to_transform):
    ohc = OneHotEncoder(sparse=False, drop='first')
    ohc_feat = ohc.fit_transform(df[cols_to_transform])
    ohc_feat_names = ohc.get_feature_names_out()
    ohc_df = pd.DataFrame(ohc_feat, index=df.index, columns=ohc_feat_names)
    df[ohc_feat_names] = ohc_df[ohc_feat_names]


def get_users_seniority(df, df_retweeters):
    print("         Calculating accounts seniority\n")
    year = df.iloc[0]['timestamp'].year

    df['created_at'] = pd.to_datetime(df['created_at'], utc=True).dt.strftime("%Y-%m-%d")
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['seniority'] = [relativedelta(datetime.datetime(year, 12, 31), x).years for x in df['created_at']]

    df_retweeters['created_at'] = pd.to_datetime(df_retweeters['created_at'], utc=True).dt.strftime("%Y-%m-%d")
    df_retweeters['created_at'] = pd.to_datetime(df_retweeters['created_at'])
    df_retweeters['seniority'] = [relativedelta(datetime.datetime(year, 12, 31), x).years for x in df_retweeters['created_at']]

    return df


def outliers(df):
    print("(9/10) - Removing outliers")
    outliers_filter = ((df['followers'] < 100000) & (df['following'] < 30000) & (df['retweet_count'] < 1000) &
                       (df['like_count'] < 400) & (df['seniority'] < 17))
    return outliers_removal(df, outliers_filter)


def outliers_removal(df, outliers_filter):
    df_no_outliers = df[outliers_filter].copy()
    print('         Percentage of data kept after removing outliers:',
          np.round(df_no_outliers.shape[0] / df.shape[0], 4) * 100, '%')
    print('         Percentage of data removed:', np.round((1 - (df_no_outliers.shape[0] / df.shape[0])) * 100, 4), '%')
    print("         Size after outlier removal:", df_no_outliers.shape[0])
    df_rets = df[df['retweet_count'] > 0]
    df_rets_out = df_no_outliers[df_no_outliers['retweet_count'] > 0]
    print("         N. of tweets with atleast 1 retweet:", df_no_outliers[df_no_outliers['retweet_count'] > 0].shape[0])
    print('         Percentage of tweets with atleast 1 retweet removed:',
          np.round((1 - (df_rets_out.shape[0] / df_rets.shape[0])) * 100, 4), '%\n')
    return df_no_outliers


def save_to_csv(df, df_retweets_info, dataset_file, retweets_info_file):
    print("(10/10) - Saving datasets")
    year = df.iloc[0]['timestamp'].year
    location = dataset_file + str(year) + '.csv'
    location_retweets = retweets_info_file + str(year) + '.csv'
    print("         Saving final dataset in", location, "for year:", year)
    print("         Saving final retweets info dataset in", location_retweets, "for year:", year)
    df.to_csv(location, sep=',', date_format='%Y-%m-%d %H:%M:%S')
    df_retweets_info.to_csv(location_retweets, sep=',', date_format='%Y-%m-%d %H:%M:%S')


def process_and_transform(tweets_path, user_path, retweets_path, retweeters_path):
    df_tweets, df_users = load_datasets(tweets_path, user_path)

    df_retweets_info = get_tweet_reach(df_tweets, df_users, retweets_path, retweeters_path)

    process_topics(df_tweets)

    sentiment_classification(df_tweets)

    hashtags(df_tweets)

    popularity(df_tweets)

    df = merge_tweets_and_users(df_tweets, df_users)

    time_phases_transformation(df, df_retweets_info)

    df = outliers(df)

    save_to_csv(df, df_retweets_info, FINAL_DATASET, FINAL_DATASET_RETWEETS_INFO)
