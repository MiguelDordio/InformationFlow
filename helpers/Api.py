import yaml
import tweepy

from helpers.csv import csv_to_tweets, csv_to_users, tweets_to_csv, users_to_csv
from models.Tweet import extract_tweet
from models.User import extract_user

FILE_CONFIGS_FETCHERS = "../../config.yaml"
FILE_CONFIGS = "../config.yaml"
FILE_BASE = "../data/"

def process_yaml():
    try:
        with open(FILE_CONFIGS_FETCHERS) as file:
            return yaml.safe_load(file)
    except OSError as e:
        with open(FILE_CONFIGS) as file:
            return yaml.safe_load(file)


def connect_tweepy():
    configs = process_yaml()
    twitter_bearer_token = configs["search_tweets_api"]["bearer_token"]
    return tweepy.Client(twitter_bearer_token, wait_on_rate_limit=True)


def connect_streaming_tweepy():
    configs = process_yaml()
    consumer_key = configs["search_tweets_api"]["consumer_key"]
    consumer_secret = configs["search_tweets_api"]["consumer_secret"]
    access_token = configs["search_tweets_api"]["access_token"]
    access_token_secret = configs["search_tweets_api"]["access_token_secret"]
    return tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)


def process_tweet_user(tweet_data, user_data):
    processed_user = None
    if user_data is not None:
        processed_user = extract_user(user_data)
    processed_tweet = extract_tweet(tweet_data)
    return processed_tweet, processed_user


def process_tweets_and_users(response):
    processed_tweets = []
    processed_users = []

    if response.data is not None:
        data_size = len(response.data)
        users_size = len(response.includes['users'])
        if not isinstance(response.data, list):
            return process_tweet_user(response.data, response.includes['users'][0])
        else:
            for i in range(len(response.data)):
                tweet = response.data[i]

                tweet_user = None
                if data_size == users_size:
                    tweet_user = response.includes['users'][i]
                else:
                    for user in response.includes['users']:
                        if user.id == tweet.author_id:
                            tweet_user = user

                processed_tweet, processed_user = process_tweet_user(tweet, tweet_user)

                processed_tweets.append(processed_tweet)
                processed_users.append(processed_user)
    else:
        print("Fetch had no data")

    return processed_tweets, processed_users


def save_state(current_tweets, current_users, all_tweets, all_users, save, tweets_filename, users_filename):
    all_tweets += current_tweets
    all_users += current_users

    if save:
        print("\nSaving state...")
        saved_tweets = csv_to_tweets(tweets_filename)
        print("previously saved_tweets size", len(saved_tweets))
        saved_users = csv_to_users(users_filename)
        print("previously saved_users size", len(saved_users))
        all_tweets += saved_tweets
        all_users += saved_users
        print("saving", len(all_tweets), "tweets")
        tweets_to_csv(tweets_filename, all_tweets)
        print("saving", len(all_users), "users")
        users_to_csv(users_filename, all_users)
        return [], []
    return all_tweets, all_users
