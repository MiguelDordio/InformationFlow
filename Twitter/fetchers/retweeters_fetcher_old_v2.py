import datetime
import socket
from time import sleep
import tweepy
import pandas as pd
import math
from helpers.Api import connect_tweepy, process_tweets_and_users, save_state, process_users

USERS_TWEETS = "../../data/tweets/raw_tweets_2019.csv"
RETWEETS_USERS = "../../data/true_retweets_users.csv"
QUERY_MAX_RESULTS = 500
MAX_QUERY_SIZE = 1024


def fetch_retweeters():
    tweepy_api = connect_tweepy()
    df_tweets = pd.read_csv(filepath_or_buffer=USERS_TWEETS, sep=",")

    df_retweets = df_tweets[(df_tweets['topics'] != '[]') & (df_tweets['retweet_count'] > 10)]
    df_retweets = df_retweets[:10]

    print("Retweeters will be collected for", len(df_retweets['tweet_id'].tolist()), "tweets\n")
    #get_tweets_retweets(tweepy_api, df_retweets)
    print("Job done, check files")


def get_tweets_retweets(tweepy_api, df_retweets):
    retweets = []
    retweets_users = []
    tweets_ids = df_retweets['tweet_id'].tolist()
    total_requests = len(tweets_ids)
    i = 0
    while i < total_requests:
        print("Retweeters will be collected from tweet", i)
        max_results, max_limit = tweets_to_fetch_count(df_retweets, i)
        users = api_fetch(tweepy_api.get_retweeters, tweets_ids[i], max_results, max_limit)
        f_users = [u for user in users for u in user]
        retweets, retweets_users = save_state([], f_users, retweets, retweets_users, i % 10000 == 0, '', RETWEETS_USERS)


def tweets_to_fetch_count(df_retweets, i):
    row = df_retweets.iloc[[i]]
    retweets_count = row['retweet_count'].tolist()
    quotes_count = row['quote_count'].tolist()
    total_count = sum(retweets_count) + sum(quotes_count)
    max_limit = math.ceil(total_count / QUERY_MAX_RESULTS)
    if total_count < 100:
        total_count = 100
    elif total_count > QUERY_MAX_RESULTS:
        total_count = QUERY_MAX_RESULTS
    print("Max limit of tweets to fetch", total_count, "x", max_limit, "=", total_count * max_limit)
    return total_count, max_limit


def api_fetch(endpoint, tweet_id, max_results, max_limit):
    users_collected = []
    while True:
        try:
            for response in tweepy.Paginator(endpoint,
                                             id=tweet_id,
                                             user_fields=['created_at', 'verified', 'public_metrics'],
                                             max_results=max_results, limit=max_limit):
                print("Response collected...\n")
                users = process_users(response)
                users_collected.append(users)
                sleep(1)
            break
        except ConnectionResetError:
            print("ConnectionResetError: [Errno 54] Connection reset by peer")
            sleep(60)
            continue
        except ConnectionError as e:  # This is the correct syntax
            print("ConnectionError:", e)
            sleep(60)
            continue
        except socket.timeout as e:
            print("Timeouterror:", e)
            sleep(60)
            continue
    return users_collected


if __name__ == '__main__':
    fetch_retweeters()
