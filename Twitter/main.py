import datetime
from os import walk

from Twitter.analysis_v2.data_analysis import analysis
from Twitter.analysis_v2.data_prediction import predict
from Twitter.analysis_v2.data_processing_transformation import process_and_transform
from Twitter.fetchers.retweets_fetcher import fetch_retweeters
from Twitter.fetchers.tweets_fetchers import get_tweets_and_users


PATH_RAW_TWEETS_FILES = "../data/tweets/"
PATH_RAW_USERS_FILES = "../data/users/"
PATH_RAW_RETWEETS_FILES = "../data/retweets/"
PATH_RAW_RETWEETERS_FILES = "../data/retweeters"

PATH_PROCESSED_TWEETS_FILES = "../data/processed_tweets"
PATH_PROCESSED_RETWEETS_FILES = "../data/processed_retweets"


def main():
    define_program_usage()


def define_program_usage():
    print("Welcome to InformationFlow!\n")
    print("Please select a number to indicate how you would like to use the program:")
    print("1 - Full run (collect, process, analyze and predict)")
    print("2 - collect")
    print("3 - process")
    print("4 - analyze")
    print("5 - predict")
    option_selected = input("Enter the desired option: ")

    if option_selected == 1:
        run_all()
    elif option_selected == 2:
        get_raw_tweets_retweets()
    elif option_selected == 3:
        process_data()
    elif option_selected == 4:
        analyze()
    elif option_selected == 5:
        predict()
    else:
        print("Program closed!")


def run_all():
    get_raw_tweets_retweets()
    process_data()
    analyze()
    do_prediction()


def get_raw_tweets_retweets():
    print("Starting to collect data...")
    start_str = input("Enter the date from when to collect the data (yyyy-mm-dd format):")
    start_time = datetime.datetime.strptime(start_str, "%Y-%m-%d")

    end_str = input("Enter the date until when to collect the data (yyyy-mm-dd format):")
    end_time = datetime.datetime.strptime(end_str, "%Y-%m-%d")

    print(start_time)
    print(end_time)

    get_tweets_and_users(start_time, end_time)

    tweets_filenames = next(walk(PATH_RAW_TWEETS_FILES), (None, None, []))[2]
    for filename in tweets_filenames:
        fetch_retweeters(PATH_RAW_TWEETS_FILES + filename)


def process_data():
    print("Processing previously collected data...")
    tweets_filenames = next(walk(PATH_RAW_TWEETS_FILES), (None, None, []))[2]
    users_filenames = next(walk(PATH_RAW_USERS_FILES), (None, None, []))[2]
    retweets_filenames = next(walk(PATH_RAW_RETWEETS_FILES), (None, None, []))[2]
    retweeters_filenames = next(walk(PATH_RAW_RETWEETERS_FILES), (None, None, []))[2]

    for filenames in zip(tweets_filenames, users_filenames, retweets_filenames, retweeters_filenames):
        process_and_transform(filenames[0], filenames[1], filenames[2], filenames[3])


def analyze():
    print("Analyzing previously collected and processed data...")
    processed_filenames = next(walk(PATH_PROCESSED_TWEETS_FILES), (None, None, []))[2]
    filenames = [PATH_PROCESSED_TWEETS_FILES + "/" + filename for filename in processed_filenames]
    analysis(filenames)


def do_prediction():
    print("Doing predictions with previously collected and processed data...")
    predict()


if __name__ == '__main__':
    main()