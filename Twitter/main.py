import datetime
from os import walk

from Twitter.analysis.data_analysis import analysis
from Twitter.analysis.data_prediction_ml import train_test_model
from Twitter.analysis.data_processing_transformation import process_and_transform
from Twitter.analysis.retweets_analysis import retweets_analysis
from Twitter.data_fetchers.retweets_fetcher import fetch_retweeters
from Twitter.data_fetchers.tweets_fetchers import get_tweets_and_users


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
    print("5 - create, train and test model")
    print("6 - predict")
    option_selected = int(input("Enter the desired option: "))

    if option_selected == 1:
        run_all()
    elif option_selected == 2:
        get_raw_tweets_retweets()
    elif option_selected == 3:
        process_data()
    elif option_selected == 4:
        print("Please specify the type of analysis")
        print("1 - Full analysis")
        print("2 - Tweets analysis")
        print("3 - Retweets analysis")
        option_selected = int(input("Enter the desired option: "))
        analyze(option_selected)
    elif option_selected == 5:
        print("Please specify if you want to do hyper parameter optimization")
        print("0 - No")
        print("1 - Yes")
        option_selected = int(input("Enter the desired option: "))
        if option_selected == 0:
            create_train_test_model(False)
        else:
            create_train_test_model(True)
    else:
        print("Program closed!")


def run_all():
    get_raw_tweets_retweets()
    process_data()
    analyze(1)
    create_train_test_model(True)


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
        process_and_transform(PATH_RAW_TWEETS_FILES + '/' + filenames[0],
                              PATH_RAW_USERS_FILES + '/' + filenames[1],
                              PATH_RAW_RETWEETS_FILES + '/' + filenames[2],
                              PATH_RAW_RETWEETERS_FILES + '/' + filenames[3])


def analyze(analysis_type):
    print("Analyzing previously collected and processed data...")

    processed_filenames = next(walk(PATH_PROCESSED_TWEETS_FILES), (None, None, []))[2]
    filenames = [PATH_PROCESSED_TWEETS_FILES + "/" + filename for filename in processed_filenames]

    processed_retweets_filenames = next(walk(PATH_PROCESSED_RETWEETS_FILES), (None, None, []))[2]
    retweets_filenames = [PATH_PROCESSED_RETWEETS_FILES + "/" + filename for filename in processed_retweets_filenames]

    if analysis_type == 1:
        analysis(filenames, False)
        retweets_analysis(filenames, retweets_filenames)
    elif analysis_type == 2:
        print("Please specify if you want only covid-19 analysis")
        print("1 - Full analysis")
        print("2 - Only covid-19 analysis")
        option_selected = int(input("Enter the desired option: "))
        if option_selected == 1:
            analysis(filenames, False)
        else:
            analysis(filenames, True)
    elif analysis_type == 3:
        retweets_analysis(filenames, retweets_filenames)


def create_train_test_model(parameter_optimization):
    print("Creating, training and testing new model...")
    train_test_model("../data/processed_tweets/", parameter_optimization)


if __name__ == '__main__':
    main()
