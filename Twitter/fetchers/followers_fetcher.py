from time import sleep

import tweepy
import copy

from helpers.Api import connect_tweepy
from helpers.csv import save_users_follows, load_data_from_users_csv, save_users_to_csv, load_data_from_follows_csv
from models.User import extract_user

FILE_USERS = "../../data/users.csv"
FILE_FOLLOWS_USERS = "../../data/follows_users.csv"
FILE_USERS_CONNECTIONS = "../../data/users_follows.csv"


def api_fetch(fetch_item, user_id, follows):
    backoff_counter = 1
    data_collected = []
    while True:
        try:
            for response in tweepy.Paginator(fetch_item, user_id,
                                             user_fields=['created_at', 'verified', 'public_metrics', 'protected',
                                                          'profile_image_url'],
                                             max_results=1000, limit=10):
                print(response)
                data_collected.append(response)
            break
        except ConnectionResetError:
            print("ConnectionResetError: [Errno 54] Connection reset by peer")
            sleep(60 * backoff_counter)
            backoff_counter += 1
            continue
    follows.append(data_collected)


def obtain_users_follows(tweepy_api, users, follows_users, follows_connections, intermediate_save, start):
    user_followers = []
    user_followings = []

    new_followers_users = []
    new_followings_users = []

    seen_users = copy.deepcopy(users)
    save_counter = 15
    for i in range(start, len(users)):
        if i == save_counter and intermediate_save:
            print("\nNow saving current state at index:", i, "\n")
            save_counter += 15
            save_fetched_follows(users, seen_users, user_followers, user_followings, new_followers_users,
                                 new_followings_users, follows_users, follows_connections, i - start)

        print("\nFetching followers for user:", users[i].user_id, "index:", i)
        api_fetch(tweepy_api.get_users_followers, users[i].user_id, user_followers)
        print("\nFetching followings for user:", users[i].user_id, "index:", i)
        api_fetch(tweepy_api.get_users_following, users[i].user_id, user_followings)

    print("\nData fetched, now processing...")
    save_fetched_follows(users, seen_users, user_followers, user_followings, new_followers_users,
                         new_followings_users, follows_users, follows_connections, len(users) - start)


def save_fetched_follows(users, seen_users, user_followers, user_followings, new_followers_users, new_followings_users,
                         follows_users, follows_connections, iterations):
    u_followers = process_follows(seen_users, user_followers, new_followers_users, iterations)
    u_followings = process_follows(seen_users, user_followings, new_followings_users, iterations)
    save_users_to_csv(FILE_FOLLOWS_USERS, follows_users + new_followers_users + new_followings_users)
    save_users_follows(users, u_followers, u_followings, follows_connections, iterations)


def process_follows(seen_users, raw_follows, new_users, size):
    follows = []

    for i in range(size):
        raw_follow = raw_follows[i]
        user_follow = []
        for users_follows in raw_follow:
            if users_follows.data is not None:
                for follow in users_follows.data:
                    user = extract_user(follow)
                    if user not in seen_users:
                        seen_users.append(user)
                        new_users.append(user)
                    user_follow.append(user)
            else:
                print("Fetch had no data, moving to next...")
        follows.append(user_follow)
    return follows


def get_last_fetched_user(users, follows_connections: type([])):
    if len(follows_connections) == 0:
        return 0

    last_account = follows_connections[len(follows_connections) - 1]
    for i in range(len(follows_connections) - 2, -1, -1):
        if follows_connections[i].account_id == last_account.account_id:
            for j in range(len(users)):
                if users[j].user_id == follows_connections[i].account_id:
                    return j + 1
        elif follows_connections[i].follower_id == last_account.follower_id:
            for j in range(len(users)):
                if users[j].user_id == follows_connections[i].follower_id:
                    return j + 1


def get_followers_and_followings(tweepy_api, users, follows_users, follows_connections, intermediate_save):
    start_fetch_index = get_last_fetched_user(users, follows_connections)
    print("Starting fetch from Users list index:", start_fetch_index)
    obtain_users_follows(tweepy_api, users, follows_users, follows_connections, intermediate_save, start_fetch_index)


def fetch_followers_and_followings(intermediate_save):
    tweepy_api = connect_tweepy()
    users = load_data_from_users_csv(FILE_USERS)
    follows_users = load_data_from_users_csv(FILE_FOLLOWS_USERS)
    follows_connections = load_data_from_follows_csv(FILE_USERS_CONNECTIONS)
    get_followers_and_followings(tweepy_api, users[:10], follows_users, follows_connections, intermediate_save)


if __name__ == '__main__':
    fetch_followers_and_followings(False)
