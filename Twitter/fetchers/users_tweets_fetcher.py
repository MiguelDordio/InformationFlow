import datetime
from time import sleep
import tweepy
from helpers.Api import connect_tweepy
from helpers.csv import load_data_from_users_csv, save_tweets_to_csv
from models.Tweet import extract_tweet

FILE_USERS = "../../data/users.csv"
FILE_FOLLOWS_USERS = "../../data/follows_users.csv"
FILE_USERS_CONNECTIONS = "../../data/users_follows.csv"


# fetch only tweets created by the user from 22/01/2022 to 29/01/2022 (1 week)
def api_fetch(endpoint, user_id):
    tweets_collected = []
    while True:
        try:
            for response in tweepy.Paginator(endpoint, user_id,
                                             tweet_fields=['author_id', 'created_at', 'conversation_id',
                                                           'referenced_tweets', 'entities', 'public_metrics', 'source',
                                                           'reply_settings', 'lang', 'geo', 'context_annotations'],
                                             exclude=['retweets', 'replies'],
                                             expansions=['geo.place_id'],
                                             start_time=datetime.datetime(2022, 1, 22).isoformat() + "Z",
                                             end_time=datetime.datetime(2022, 1, 30).isoformat() + "Z",
                                             max_results=100, limit=1):
                print(response)
                tweets_collected.append(response)
            break
        except ConnectionResetError:
            print("ConnectionResetError: [Errno 54] Connection reset by peer")
            sleep(60)
            continue
    return tweets_collected


def get_users_tweets(tweepy_api, users):
    tweets = []
    for user in users:
        print("Collecting", user.user_id, "tweets...")
        user_tweets = process_tweets(api_fetch(tweepy_api.get_users_tweets, user.user_id))
        tweets.append(user_tweets)
        print("For user:", user.user_id, "collected", len(user_tweets), "tweets\n")
    return tweets


def process_tweets(raw_tweets):
    tweets = []
    for raw_tweet in raw_tweets:
        if raw_tweet.data is not None:
            if 'places' in raw_tweet.includes:
                places = raw_tweet.includes['places']
            else:
                places = []
            for data in raw_tweet.data:
                tweet = extract_tweet(places, data)
                tweets.append(tweet)
        else:
            print("Fetch had no data, moving to next...")

    return tweets


def fetch_users_tweets():
    tweepy_api = connect_tweepy()
    users = load_data_from_users_csv(FILE_USERS)
    print("Tweets will be collected for", len(users), "users")
    tweets = get_users_tweets(tweepy_api, users)
    print("Tweets collected:", tweets)
    save_tweets_to_csv('../../data/users_tweets.csv', tweets)


if __name__ == '__main__':
    fetch_users_tweets()
