import datetime
import math
import socket
from time import sleep

import requests
import tweepy
from helpers.Api import connect_tweepy, process_tweets_and_users, save_state

FILE_TWEETS = "../../data/tweets/raw_tweets"
FILE_USERS = "../../data/users/raw_users"
QUERY_MAX_RESULTS = 100


def get_tweets(endpoint, query, start, end, max_res, limit):
    tweets_collected = []
    users_collected = []
    while True:
        try:
            for response in tweepy.Paginator(endpoint, query=query,
                                             tweet_fields=['author_id', 'created_at', 'conversation_id',
                                                           'referenced_tweets', 'entities', 'public_metrics', 'source',
                                                           'reply_settings', 'lang', 'context_annotations'],
                                             user_fields=['created_at', 'verified', 'public_metrics'],
                                             expansions=['author_id'],
                                             start_time=start,
                                             end_time=end,
                                             max_results=max_res, limit=limit):
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
        except tweepy.errors.TwitterServerError as e:
            print("TwitterServerError:", e)
            sleep(60)
            continue
        except requests.exceptions.ConnectionError as e:
            print("TwitterServerError:", e)
            sleep(60)
            continue

    return tweets_collected, users_collected


def fetch_tweets(query, start, end, max_res, limit):
    tweepy_api = connect_tweepy()
    tweets, users = get_tweets(tweepy_api.search_all_tweets, query, start, end, max_res, limit)
    f_tweets = [t for tweet in tweets for t in tweet]
    f_users = [u for user in users for u in user]
    print("\n\nFinal results")
    print("Total of", len(f_tweets), "tweets fetched")
    print("Total of", len(f_users), "users fetched")
    return f_tweets, f_users


def get_daily_limit(num_days, total_tweets):
    tweets_per_day = total_tweets / num_days
    max_limit = math.ceil(tweets_per_day / QUERY_MAX_RESULTS)
    if tweets_per_day < 10:
        tweets_per_day = 10
        print("\n10 tweets per day will be collected, because that's the minimum per request\n")
    elif tweets_per_day > QUERY_MAX_RESULTS:
        tweets_per_day = QUERY_MAX_RESULTS
    return int(tweets_per_day), int(max_limit)


def get_timeline_tweets(query, start, end, total_tweets):
    days = [d1 + datetime.timedelta(days=x) for x in range((end - start).days + 1)]
    days_count = len(days)
    max_per_day, limit_per_day = get_daily_limit(days_count, total_tweets)
    print("max_res:", max_per_day, "| limit:", limit_per_day)
    all_tweets = []
    all_users = []
    current_year = 0
    tweets_file = ""
    users_file = ""
    for i in range(days_count):
        print("\nRequesting day:", i, "in:", days_count, "days")
        day = days[i]

        # save at every 30 days of data collected
        save = ((i+1) % 30 == 0 or i == days_count - 1)

        # check if the year changed, if yes, update the destination file
        if day.year != current_year:
            save = True
            tweets_file = FILE_TWEETS + "_" + str(day.year) + ".csv"
            users_file = FILE_USERS + "_" + str(day.year) + ".csv"
            print("Current year:", current_year, "New year:", tweets_file, users_file)

        current_year = day.year

        max_res, limit = get_daily_limit(5, max_per_day * limit_per_day)
        all_day_tweets = []
        all_day_users = []
        for hour in [1, 8, 14, 19, 22]:
            day_start = (day + datetime.timedelta(hours=hour-1)).isoformat() + "Z"
            day_end = (day + datetime.timedelta(hours=hour)).isoformat() + "Z"

            print("max_res:", max_res, "| limit:", limit)
            print('Fetching tweets starting at:', day_start)
            print("                      until:", day_end)

            f_tweets, f_users = fetch_tweets(query, day_start, day_end, max_res, limit)
            all_day_tweets += f_tweets
            all_day_users += f_users

        all_tweets, all_users = save_state(all_day_tweets, all_day_users, all_tweets, all_users, save, tweets_file,
                                           users_file)


def get_tweets_and_users(from_date, to_date):
    q2 = "lang:en place_country:US -url -is:retweet -is:reply -the the"
    max_tweets = 100
    get_timeline_tweets(q2, from_date, to_date, max_tweets)


if __name__ == '__main__':
    from_date = datetime.datetime(2019, 12, 31)
    to_date = datetime.datetime(2020, 1, 1)
    get_tweets_and_users(from_date, to_date)
