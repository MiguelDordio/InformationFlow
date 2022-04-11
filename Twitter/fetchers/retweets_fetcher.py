import datetime
import socket
from time import sleep
import tweepy
import pandas as pd
import math
from helpers.Api import connect_tweepy, process_tweets_and_users, save_state

USERS_TWEETS = "../../data/test_tweets2.csv"
RETWEETS = "../../data/retweets_v2.csv"
RETWEETS_USERS = "../../data/retweets_users_v2.csv"
QUERY_MAX_RESULTS = 500
MAX_QUERY_SIZE = 1024


def fetch_retweeters():
    tweepy_api = connect_tweepy()
    df_tweets = pd.read_csv(filepath_or_buffer=USERS_TWEETS, sep=",")

    # df_tweets = df_tweets[:20]
    df_tweets = df_tweets[df_tweets['topics'] != '[]']
    df_retweets = df_tweets[df_tweets['retweet_count'] > 0]

    print("Retweeters will be collected for", len(df_retweets['tweet_id'].tolist()), "tweets\n")
    get_tweets_retweets(tweepy_api, df_retweets)
    print("Job done, check files")


def get_tweets_retweets(tweepy_api, df_retweets):
    retweets = []
    retweets_users = []
    tweets_ids = df_retweets['tweet_id'].tolist()
    total_requests = len(tweets_ids)
    i = 0
    while i < total_requests:
        query, j = prepare_fetch_query(tweets_ids, i)
        print("Retweeters will be collected from tweet", i, "to", j)
        max_results, max_limit = tweets_to_fetch_count(df_retweets, i, j)
        tweets, users = api_fetch(tweepy_api.search_all_tweets, query, max_results, max_limit)
        i = j
        f_tweets = [t for tweet in tweets for t in tweet]
        f_users = [u for user in users for u in user]
        retweets, retweets_users = save_state(f_tweets, f_users, retweets, retweets_users,
                                              (i % 20000 == 0) or (j == total_requests), RETWEETS, RETWEETS_USERS)



def tweets_to_fetch_count(df_retweets, i, j):
    retweets_count = df_retweets[i:j]['retweet_count'].tolist()
    quotes_count = df_retweets[i:j]['quote_count'].tolist()
    total_count = sum(retweets_count) + sum(quotes_count)
    max_limit = math.ceil(total_count / QUERY_MAX_RESULTS)
    if total_count < 100:
        total_count = 100
    elif total_count > QUERY_MAX_RESULTS:
        total_count = QUERY_MAX_RESULTS
    print("Max limit of tweets to fetch", total_count, "x", max_limit, "=", total_count * max_limit)
    return total_count, max_limit


def prepare_fetch_query(tweets_ids, i):
    query = "("
    next_item_separator = " OR "
    base_query_build = "url:"
    base_filters = ") (is:retweet OR is:quote)"
    tweet_id = ""
    while i < len(tweets_ids) and len(query) + len(base_query_build) + len(tweet_id) < MAX_QUERY_SIZE - len(
            base_filters):
        tweet_id = str(tweets_ids[i])
        query += base_query_build + tweet_id + next_item_separator
        i += 1

    query = query[:len(query)-4]
    query += base_filters
    print("Searching tweets for query:", query)
    return query, i


def api_fetch(endpoint, query, max_results, max_limit):
    tweets_collected = []
    users_collected = []
    while True:
        try:
            for response in tweepy.Paginator(endpoint,
                                             query=query,
                                             tweet_fields=['author_id', 'created_at', 'conversation_id',
                                                           'referenced_tweets', 'public_metrics', 'source', 'lang'],
                                             user_fields=['created_at', 'verified', 'public_metrics'],
                                             expansions=['author_id'],
                                             max_results=max_results, limit=max_limit):
                print("Response collected...\n")
                tweets, users = process_tweets_and_users(response)
                tweets_collected.append(tweets)
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
    return tweets_collected, users_collected


if __name__ == '__main__':
    fetch_retweeters()
