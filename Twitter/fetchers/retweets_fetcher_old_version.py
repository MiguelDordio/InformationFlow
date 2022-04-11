import datetime
import socket
from time import sleep
import tweepy
import pandas as pd
import math
from helpers.Api import connect_tweepy, process_tweets_and_users, save_state

USERS = "../../data/test_users2.csv"
USERS_TWEETS = "../../data/test_tweets2.csv"
RETWEETS = "../../data/retweets_v2.csv"
RETWEETS_USERS = "../../data/retweets_users_v2.csv"
QUERY_MAX_RESULTS = 500


def get_time_period(tweets_timetamps):
    return (datetime.datetime.strptime(min(tweets_timetamps), '%Y-%m-%d %H:%M:%S+00:00')
            - datetime.timedelta(hours=1)).isoformat() + "Z", \
           (datetime.datetime.strptime(max(tweets_timetamps), '%Y-%m-%d %H:%M:%S+00:00')
            + datetime.timedelta(days=30)).isoformat() + "Z"


def api_fetch(endpoint, tweets_by_user, username):
    tweets_collected = []
    users_collected = []
    max_results, max_limit = tweets_to_fetch_count(tweets_by_user)
    print("Max limit of tweets to fetch for user", username, "is", max_results)
    start, end = get_time_period(tweets_by_user['timestamp'].tolist())
    print("start date:", start, "\nend date:", end)
    while True:
        try:
            for response in tweepy.Paginator(endpoint,
                                             query=prepare_fetch_query(tweets_by_user['tweet_id'].tolist(), username),
                                             tweet_fields=['author_id', 'created_at', 'conversation_id',
                                                           'referenced_tweets', 'entities', 'public_metrics', 'source',
                                                           'reply_settings', 'lang'],
                                             user_fields=['created_at', 'verified', 'public_metrics'],
                                             expansions=['author_id'],
                                             start_time=start,
                                             end_time=end,
                                             max_results=50, limit=max_limit):
                print("Response collected...")
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


def api_fetchv2(endpoint):
    tweets_collected = []
    users_collected = []
    while True:
        try:
            for response in tweepy.Paginator(endpoint,
                                             query="(url:1505348333665079296) (is:retweet OR is:quote)",
                                             tweet_fields=['author_id', 'created_at', 'conversation_id',
                                                           'referenced_tweets', 'public_metrics', 'source', 'lang'],
                                             user_fields=['created_at', 'verified', 'public_metrics'],
                                             expansions=['author_id'],
                                             max_results=50, limit=1):
                print("Response collected...")
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


def get_tweets_retweets(tweepy_api, tweets_by_users, df_users):
    retweets = []
    retweets_users = []
    total_requests = len(tweets_by_users.items())
    count = 1
    for user_id, df_tweets_by_user in tweets_by_users.items():
        username = df_users[df_users['user_id'] == user_id]['username'].tolist()[0]
        print("Collecting retweeted/quoted tweets from user:", username)
        tweets, users = api_fetch(tweepy_api.search_all_tweets, df_tweets_by_user, username)
        f_tweets = [t for tweet in tweets for t in tweet]
        f_users = [u for user in users for u in user]
        retweets, retweets_users = save_state(f_tweets, f_users, retweets, retweets_users,
                                              (count % 20000 == 0) or (count == total_requests - 1),
                                              RETWEETS, RETWEETS_USERS)
        count += 1

        print("\n\n")


def tweets_to_fetch_count(df_retweets):
    retweets_count = df_retweets['retweet_count'].tolist()
    quotes_count = df_retweets['quote_count'].tolist()
    total_count = sum(retweets_count) + sum(quotes_count)
    max_limit = math.ceil(total_count / QUERY_MAX_RESULTS)
    if total_count < 100:
        total_count = 100
    elif total_count > QUERY_MAX_RESULTS:
        total_count = QUERY_MAX_RESULTS
    return total_count, max_limit


def get_tweets_by_user(df_retweets):
    unique_users = df_retweets['user_id'].unique()

    # create a data frame dictionary to store your data frames
    tweets_by_user = {elem: pd.DataFrame for elem in unique_users}
    print(len(tweets_by_user.keys()), "API call will be made\n")

    for key in tweets_by_user.keys():
        tweets_by_user[key] = df_retweets[:][df_retweets['user_id'] == key]

    return tweets_by_user


def prepare_fetch_query(tweets_ids, username):
    tweets_ids_str = ""  # ATENCAO TALVEZ DE PARA MELHORAR COM O OPERADOR: retweets_of ou o conversation_id ou ainda url:tweet_id
    base_query = "(RT @" + username + " is:retweet)"
    for tweets_id in tweets_ids:
        if (len(base_query) + len(tweets_ids_str + str(tweets_id) + " OR ")) < 1024:
            tweets_ids_str = tweets_ids_str + str(tweets_id) + " OR "
        else:
            break
    query = tweets_ids_str + base_query
    query = "url:"
    print("Searching tweets for query:", query)
    return query


def fetch_retweeters():
    tweepy_api = connect_tweepy()
    df_users = pd.read_csv(filepath_or_buffer=USERS, sep=",")
    df_users = df_users.drop_duplicates(subset=['user_id'], keep="first")
    df_tweets = pd.read_csv(filepath_or_buffer=USERS_TWEETS, sep=",")

    #df_tweets = df_tweets[:20]
    df_tweets = df_tweets[df_tweets['topics'] != '[]']
    df_retweets = df_tweets[df_tweets['retweet_count'] > 0]

    # test filter
    df_retweets = df_retweets[:2000]

    api_fetchv2(tweepy_api.search_all_tweets)

    print("Retweeters will be collected for", len(df_retweets['tweet_id'].tolist()), "tweets")
    # get_tweets_retweets(tweepy_api, get_tweets_by_user(df_retweets), df_users)
    print("Job done, check files")


if __name__ == '__main__':
    fetch_retweeters()
