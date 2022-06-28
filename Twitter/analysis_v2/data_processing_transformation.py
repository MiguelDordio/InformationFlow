import pandas as pd
import numpy as np
import json
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from dateutil.relativedelta import relativedelta


FILE_USERS = "../../data/raw_users.csv"
FILE_USERS_TWEETS = "../../data/raw_tweets.csv"
FILE_RETWEETERS = "../../data/retweets.csv"
FILE_RETWEETERS_USERS = "../../data/retweets_users.csv"
FINAL_DATASET = "../../data/tweets_2020_2021.csv"



def calculate_tweet_reach(df_tweets, df_users):
    df_retweeters = pd.read_csv(filepath_or_buffer=FILE_RETWEETERS_USERS, sep=",")
    df_retweets = pd.read_csv(filepath_or_buffer=FILE_RETWEETERS, sep=",")

    # remove duplicated retweeters
    df_retweeters = df_retweeters.drop_duplicates(subset=['user_id'], keep="first")

    # sort by timestamp
    df_retweets = df_retweets.sort_values(by='timestamp', ascending=True).reset_index()

    # merge retweeters datasets
    df_retweets_info = pd.merge(df_retweets[['tweet_id', 'user_id', 'topics', 'referenced_tweets']],
                                df_retweeters[['user_id', 'followers', 'following']], how='left', on="user_id")

    # remove NANs
    df_retweets_info = df_retweets_info[df_retweets_info['referenced_tweets'].notna()]

    df_retweets_info['ref_tweed_id'] = df_retweets_info['referenced_tweets'].apply(
        lambda ref_tweet: get_ref_tweet_id(ref_tweet))

    df_tweets['ref_tweed_id'] = df_tweets['referenced_tweets'].apply(lambda ref_tweet: get_ref_tweet_id(ref_tweet))
    # converting the reference ids that exist to int
    df_tweets['ref_tweed_id'] = pd.to_numeric(df_tweets['ref_tweed_id'], errors='coerce').fillna(0).astype(np.int64)

    df_tweets['reach'] = df_tweets.apply(
        lambda tweet_row: calculate_tweet_reach(tweet_row['tweet_id'], tweet_row['user_id'], tweet_row['topics'],
                                                tweet_row['retweet_count'], tweet_row['ref_tweed_id'], df_users,
                                                df_retweets_info), axis=1)


def get_ref_tweet_id(ref_tweets):
    g = str(ref_tweets)
    if isinstance(g, str) and g != 'nan' and not pd.isna(g):
        return int(g.split()[1].replace(',', ''))
    else: return None


def calculate_tweet_reach(tweet_id, user_id, topics, retweets_count, ref_tweed_id, df_users, df_retweets_info):
    total_reach = 0
    total_reach += df_users[df_users['user_id'] == user_id]['followers'].sum()
    if topics != '[]' and retweets_count > 0:
        if ref_tweed_id is not None and ref_tweed_id != 0:
            total_reach += df_retweets_info[(df_retweets_info['ref_tweed_id'] == tweet_id) | (df_retweets_info['ref_tweed_id'] == ref_tweed_id)]['followers'].sum()
        else:
            total_reach += df_retweets_info[df_retweets_info['ref_tweed_id'] == tweet_id]['followers'].sum()
    return total_reach


def process_topics(df_tweets):
    print("Topics: get topics")
    df_tweets['topics'] = df_tweets['topics'].apply(lambda topics: process_topic(topics))
    df_tweets['topics_ids'] = df_tweets['topics_ids'].apply(lambda topics: process_topic(topics))

    # grouping topics
    print("Topics: group topics by category")
    df_tweets['topics_cleaned'] = df_tweets['topics'].apply(lambda topic: group_topics(topic))

    print("Topics: removing NaNs and int conversion")
    df_tweets['topics_ids'].fillna(value=0, inplace=True)
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
    elif 'Sport' in topic or 'Athlete' in topic or 'Coach' in topic or 'Hockey' in topic or 'Football' in topic or 'NFL' in topic:
        return 'Sport'
    elif 'TV' in topic or 'Movie' in topic or 'Award' in topic or 'Actor' in topic or 'Fictional Character' in topic\
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
    sid_obj = SentimentIntensityAnalyzer()
    df_tweets['sentiment'] = df_tweets['text'].apply(lambda tweet_text: sentiment_scores(tweet_text, False, sid_obj))


def sentiment_scores(sentence, prints, sid_obj):
    if prints: print("\nSentence:", sentence)

    sentiment_dict = sid_obj.polarity_scores(sentence)

    if sentiment_dict['compound'] >= 0.05:
        if prints: print("Positive")
        return "Positive"
    elif sentiment_dict['compound'] <= - 0.05:
        if prints: print("Positive")
        return "Negative"
    else:
        if prints: print("Neutral")
        return "Neutral"


def tweet_popularity_label(retweet_count, quote_count):
    if retweet_count > 0 or quote_count > 0:
        return 1
    else:
        return 0


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


def time_phases_transformation(df):
    df['year'] = df['timestamp'].apply(lambda x: x.year)
    df['month'] = df['timestamp'].apply(lambda x: x.strftime('%B'))
    df['day_of_week'] = df['timestamp'].apply(lambda x: x.strftime('%A'))
    df['day_phase'] = df['timestamp'].apply(lambda x: get_day_phase(int(x.hour)))
    df['week_idx'] = df['timestamp'].apply(lambda x: '%s-%s' % (x.year, '{:02d}'.format(x.isocalendar()[1])))
    df = time_phases_encoding(df)
    return get_users_seniority(df)


def time_phases_encoding(df):
    cols_to_transform = ['day_phase', 'day_of_week', 'month', 'year', 'sentiment', 'verified']
    for col in cols_to_transform:
        enc = LabelEncoder()
        enc.fit(df[col])
        df[col + '_enc'] = enc.transform(df[col])
    return df


def get_users_seniority(df):
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True).dt.strftime("%Y-%m-%d")
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['seniority'] = df['created_at'].apply(lambda x: relativedelta(datetime.datetime.now(), x).years)
    return df


def outliers(df):
    outliers_filter = (((df['followers'] < 10000) & (df['following'] < 70000))
                      & ((df['retweet_count'] < 100) & (df['like_count'] < 4000) & (df['seniority'] < 17)))
    df_no_outliers = df[outliers_filter].copy()

    print('Percentage of data kept after removing outliers:', np.round(df_no_outliers.shape[0] / df.shape[0], 4) * 100,
          '%')
    print('Percentage of data removed:', np.round((1 - (df_no_outliers.shape[0] / df.shape[0])) * 100, 4), '%')

    return df_no_outliers


def process_and_transform():
    df_users = pd.read_csv(filepath_or_buffer=FILE_USERS, sep=",")
    df_tweets = pd.read_csv(filepath_or_buffer=FILE_USERS_TWEETS, sep=",")

    # sort by timestamp
    df_tweets = df_tweets.sort_values(by='timestamp', ascending=True).reset_index()

    # delete duplicate users
    df_users = df_users.drop_duplicates(subset=['user_id'], keep="first")

    calculate_tweet_reach(df_tweets, df_users)

    process_topics(df_tweets)

    sentiment_classification(df_tweets)

    df_tweets['popularity'] = df_tweets.apply(
        lambda row: tweet_popularity_label(row['retweet_count'], row['quote_count']), axis=1)

    # merging tweets data with respective users info
    df_complete = pd.merge(df_tweets[
                                ['tweet_id', 'text', 'timestamp', 'user_id', 'like_count', 'retweet_count',
                                 'quote_count',
                                 'reply_count', 'reach', 'topics_ids', 'topics', 'sentiment', 'popularity']],
                            df_users[['user_id', 'followers', 'following', 'tweet_count', 'verified', 'created_at']],
                            on="user_id").reset_index()

    df_complete['timestamp'] = pd.to_datetime(df_complete['timestamp'])

    df_complete = time_phases_encoding(df_complete)

    df = outliers(df_complete)

    df.to_csv('../../data/processed_' + str(df['year'].iloc[0]) + '.csv', sep=',', date_format='%Y-%m-%d %H:%M:%S')

