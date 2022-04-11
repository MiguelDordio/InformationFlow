import pandas as pd
import json

from models.Follow import Follow
from models.Tweet import csv_row_to_tweet
from models.User import csv_row_to_user


def users_to_csv(filename, users):
    columns = ['user_id', 'name', 'username', 'followers', 'following', 'tweet_count', 'verified', 'created_at']
    df = pd.DataFrame([user.__dict__ for user in users], columns=columns)
    df.to_csv(filename, sep=',', date_format='%Y-%m-%d %H:%M:%S')


def csv_to_users(filename):
    try:
        # df = pd.read_csv(filepath_or_buffer=filename, sep=",",
        #                 dtype={"user_id": int, "name": str, "username": str, 'followers': int, 'following': int,
        #                        'tweet_count': int, 'verified': bool, 'created_at': object})
        df = pd.read_csv(filepath_or_buffer=filename, sep=",", low_memory=False)
        users = [csv_row_to_user(row) for index, row in df.iterrows()]
        return users
    except OSError:
        return []


def tweets_to_csv(filename, tweets):
    columns = ['tweet_id', 'text', 'user_id', 'timestamp', 'tweet_type', 'like_count', 'reply_count',
               'retweet_count', 'quote_count', 'device', 'lang', 'topics_ids', 'topics', 'referenced_tweets']
    df = pd.DataFrame([tweet.__dict__ for tweet in tweets], columns=columns)
    df['text'] = df['text'].apply(lambda x: x.replace('\n', ' ').replace('\r', ''))
    df.to_csv(filename, sep=',', date_format='%Y-%m-%d %H:%M:%S')


def raw_tweets_to_csv(filename, tweets):
    columns = ['tweet_id', 'text', 'user_id', 'timestamp', 'tweet_type', 'like_count', 'reply_count',
               'retweet_count', 'quote_count', 'device', 'lang', 'topics_ids', 'topics', 'referenced_tweets']
    df = pd.DataFrame([tweet.__dict__ for tweet in tweets], columns=columns)
    df['text'] = df['text'].apply(lambda x: x.replace('\n', ' ').replace('\r', ''))
    df.to_csv(filename, sep=',', date_format='%Y-%m-%d %H:%M:%S')


def csv_to_tweets(filename):
    try:
        df = pd.read_csv(filepath_or_buffer=filename, sep=",")
        tweets = [(csv_row_to_tweet(row)) for index, row in df.iterrows()]
        return tweets
    except OSError:
        return []


def save_retweeters_tweets_to_csv(filename, tweets):
    df = pd.DataFrame(columns=['tweet_id', 'text', 'user_id', 'timestamp', 'conversation_id', 'tweet_type', 'lang',
                               'referenced_tweets', 'like_count', 'reply_count', 'retweet_count',
                               'quote_count', 'entities', 'device', 'reply_settings', 'context_annotations'])

    tweet_id, text, user_id, created_at, conversation_id, tweet_type, lang = [], [], [], [], [], [], []
    referenced_tweets, geo, like_count, reply_count, retweet_count = [], [], [], [], []
    quote_count, entities, source, reply_settings, context_annotations = [], [], [], [], []

    for tweet in tweets:
        tweet_id.append(tweet.tweet_id)
        text.append(tweet.text)
        user_id.append(tweet.user_id)
        created_at.append(tweet.created_at)
        conversation_id.append(tweet.conversation_id)
        tweet_type.append(tweet.tweet_type)
        lang.append(tweet.lang)
        if tweet.referenced_tweets is not None and len(tweet.referenced_tweets) > 0:
            referenced_tweets.append(json.dumps(tweet.referenced_tweets, default=obj_dict))
        else:
            referenced_tweets.append(None)
        if tweet.geo is not None:
            geo.append(json.dumps(tweet.geo.__dict__))
        else:
            geo.append(None)
        like_count.append(tweet.public_metrics.like_count)
        reply_count.append(tweet.public_metrics.reply_count)
        retweet_count.append(tweet.public_metrics.retweet_count)
        quote_count.append(tweet.public_metrics.quote_count)
        if tweet.entities is not None:
            entities.append(json.dumps(tweet.entities.__dict__))
        else:
            entities.append(None)
        source.append(tweet.source)
        reply_settings.append(tweet.reply_settings)
        if tweet.context_annotations is not None and len(tweet.context_annotations) > 0:
            context_annotations.append(json.dumps(tweet.context_annotations, default=obj_dict))
        else:
            context_annotations.append(None)


    df['tweet_id'] = tweet_id
    df['text'] = text
    df['user_id'] = user_id
    df['timestamp'] = created_at
    df['conversation_id'] = conversation_id
    df['tweet_type'] = tweet_type
    df['lang'] = lang
    df['referenced_tweets'] = referenced_tweets
    df['geo'] = geo
    df['like_count'] = like_count
    df['reply_count'] = reply_count
    df['retweet_count'] = retweet_count
    df['quote_count'] = quote_count
    df['entities'] = entities
    df['device'] = source
    df['reply_settings'] = reply_settings
    df['context_annotations'] = context_annotations

    df.to_csv(filename, sep=',', date_format='%Y-%m-%d %H:%M:%S')


def save_users_follows(filepath, users, users_followers, users_followings, connections, users_size):
    df = pd.DataFrame(columns=['account_id', 'follower_id'])
    accounts_ids, accounts_followings = process_users_follows(users, users_followers, users_followings, connections,
                                                              users_size)
    df['account_id'] = accounts_ids
    df['follower_id'] = accounts_followings
    df.to_csv(filepath, sep=',')


def process_users_follows(users, users_followers, users_followings, follows_connections, users_size):
    accounts_ids, accounts_followings = [], []

    for conn in follows_connections:
        accounts_ids.append(conn.account_id)
        accounts_followings.append(conn.follower_id)

    for i in range(users_size):
        user = users[i]
        for followers in users_followers[i]:
            accounts_ids.append(user.user_id)
            accounts_followings.append(followers.user_id)

        for followers in users_followings[i]:
            accounts_ids.append(followers.user_id)
            accounts_followings.append(user.user_id)

    return accounts_ids, accounts_followings


def load_data_from_follows_csv(filepath):
    df = pd.read_csv(filepath_or_buffer=filepath, sep=",")
    follows = [(Follow(row.account_id, row.follower_id)) for index, row in df.iterrows()]
    return follows


def obj_dict(obj):
    return obj.__dict__
