import datetime

import tweepy

import helpers.csv as csv
from helpers.Api import connect_tweepy
from models.Tweet import sanitize_tweets

FILE_USERS = '../../data/users.csv'


def get_random_accounts(tweepy_api, theme, max_accounts, limit):
    parsed_tweets_users = []

    query = theme

    max_time = datetime.datetime.fromisoformat('2022-02-12')
    while len(parsed_tweets_users) < max_accounts:
        print(len(parsed_tweets_users), "accounts collected. The goal is:", max_accounts)
        for response in tweepy.Paginator(tweepy_api.search_all_tweets,
                                         query=query,
                                         start_time=datetime.datetime.fromisoformat('2021-01-01'),
                                         end_time=max_time,
                                         tweet_fields=['author_id', 'created_at', 'in_reply_to_user_id',
                                                       'referenced_tweets'],
                                         user_fields=['created_at', 'verified', 'public_metrics', 'protected',
                                                      'profile_image_url'], expansions='author_id',
                                         max_results=max_accounts, limit=limit):
            print("raw response size", len(response.data))

            sanitized_tweets, sanitized_users = sanitize_tweets(response)
            print("users collected: \n\n", sanitized_users)
            remove_duplicate_users(parsed_tweets_users, sanitized_users, max_accounts)
        max_time = max_time + datetime.timedelta(weeks=+4)

    csv.save_users_to_csv(FILE_USERS, parsed_tweets_users)

    print("\n\n", len(parsed_tweets_users), "parsed tweets users")
    print(parsed_tweets_users)

    return parsed_tweets_users


def remove_duplicate_users(parsed_tweets_users, sanitized_users, max_accounts):
    for user in sanitized_users:
        if user not in parsed_tweets_users and len(parsed_tweets_users) < max_accounts:
            parsed_tweets_users.append(user)


def fetch_accounts(theme, max_accounts):
    tweepy_api = connect_tweepy()
    get_random_accounts(tweepy_api, theme, max_accounts, 2)


if __name__ == '__main__':
    fetch_accounts("vodafone place_country:PT lang:pt", 250)
