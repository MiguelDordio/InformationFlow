import pymongo
import yaml


def process_yaml():
    with open("../config.yaml") as file:
        return yaml.safe_load(file)


def connect_mongodb():
    configs = process_yaml()
    mongo_connection = configs["mongodb"]["connection"]
    client = pymongo.MongoClient(mongo_connection)
    db = client.dissertacao  # connect to existing 'dissertacao' data base
    return db.tweet_collection  # use or create 'tweet_collection' collection


def mongo_id_index(collection):
    collection.create_index([('id', pymongo.ASCENDING)], unique=True)  # ensure the collected tweets are unique


def mongodb_text_index(collection):
    collection.create_index([('id', pymongo.ASCENDING)], unique=True)  # ensure the collected tweets are unique

def check_mongodb_status(tweet_collection):
    print(tweet_collection.estimated_document_count())  # number of tweets collected
    user_cursor = tweet_collection.distinct("id")
    print(len(user_cursor))  # number of unique Twitter users