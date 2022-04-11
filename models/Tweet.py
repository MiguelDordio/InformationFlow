TWEET_QUOTE_TYPE = "Quote Tweet"
TWEET_REPLY_TYPE = "Reply Tweet"
TWEET_RETWEET_TYPE = "Retweet Tweet"


class Tweet:

    def __init__(self, tweet_id, text, user_id, timestamp, tweet_type, device, lang,
                 like_count, reply_count, retweet_count, quote_count, topics_ids, topics, referenced_tweets):
        self.tweet_id = tweet_id
        self.text = text
        self.user_id = user_id
        self.timestamp = timestamp
        self.tweet_type = tweet_type

        self.like_count = like_count
        self.reply_count = reply_count
        self.retweet_count = retweet_count
        self.quote_count = quote_count

        self.device = device
        self.lang = lang

        self.topics_ids = topics_ids
        self.topics = topics
        self.referenced_tweets = referenced_tweets

    @classmethod
    def from_api(cls, tweet_id, text, user_id, created_at, referenced_tweets, source, lang,
                 like_count, reply_count, retweet_count, quote_count, context_annotations):

        tweet_type = cls.determine_tweet_type(referenced_tweets)
        device = cls.process_source(source)
        topics_ids, topics = cls.process_context_annotations(context_annotations)
        ref_tweets = cls.process_referenced_tweets(referenced_tweets)

        return cls(tweet_id=tweet_id, text=text, user_id=user_id, timestamp=created_at, tweet_type=tweet_type,
                   device=device, lang=lang,
                   like_count=like_count, reply_count=reply_count, retweet_count=retweet_count,
                   quote_count=quote_count,
                   topics_ids=topics_ids, topics=topics,
                   referenced_tweets=ref_tweets)

    def __repr__(self):
        return "Tweet(id: {0}, text: {1}, user_id: {2}, timestamp: {3}, tweet_type: {4}\n" \
            .format(self.tweet_id, self.text, self.user_id, self.timestamp, self.tweet_type)

    def __str__(self):
        return "id: {0}, text: {1}, user_id: {2}, timestamp: {3}, tweet_type: {4}" \
            .format(self.tweet_id, self.text, self.user_id, self.timestamp, self.tweet_type)

    @staticmethod
    def process_context_annotations(context_annotations):
        if context_annotations is None:
            return None

        topics_ids, topics = [], []
        for ca in context_annotations:
            topics_ids.append(ca['domain']['id'])
            topics.append(ca['domain']['name'])

        return topics_ids, topics

    @staticmethod
    def process_referenced_tweets(referenced_tweets):
        if referenced_tweets is None:
            return None
        return [ReferencedTweet(t.id, t.type) for t in referenced_tweets]

    @staticmethod
    def process_source(source):
        if 'iPhone' in source or ('iOS' in source):
            return 'iPhone'
        elif 'Android' in source:
            return 'Android'
        elif 'Mobile' in source:
            return 'Mobile device'
        elif 'Mac' in source:
            return 'Mac'
        elif 'Windows' in source:
            return 'Windows'
        elif 'Bot' in source:
            return 'Bot'
        elif 'Web' in source:
            return 'Web'
        elif 'Instagram' in source:
            return 'Instagram'
        elif 'Blackberry' in source:
            return 'Blackberry'
        elif 'iPad' in source:
            return 'iPad'
        elif 'Foursquare' in source:
            return 'Foursquare'
        else:
            return ''

    @staticmethod
    def determine_tweet_type(referenced_tweets):
        tweet_type = ""
        if referenced_tweets is None:
            tweet_type = "Original Tweet"
        else:
            for tweet_ref in referenced_tweets:
                if tweet_ref.type == "quoted":
                    tweet_type = TWEET_QUOTE_TYPE
                elif tweet_ref.type == "replied_to":
                    tweet_type = TWEET_REPLY_TYPE
                elif tweet_ref.type == "retweeted":
                    tweet_type = TWEET_RETWEET_TYPE
        return tweet_type


class ReferencedTweet:
    def __init__(self, id, type):
        self.id = id
        self.type = type

    def __repr__(self):
        return "ReferencedTweet(id: {0}, type: {1} \n".format(self.id, self.type)


def extract_tweet(data):
    return Tweet.from_api(tweet_id=data['id'], text=data['text'], user_id=str(data['author_id']),
                          created_at=data['created_at'],
                          source=data['source'], lang=data['lang'], referenced_tweets=data['referenced_tweets'],
                          like_count=data['public_metrics']['like_count'],
                          reply_count=data['public_metrics']['reply_count'],
                          retweet_count=data['public_metrics']['retweet_count'],
                          quote_count=data['public_metrics']['quote_count'],
                          context_annotations=data['context_annotations'])


def csv_row_to_tweet(row):
    return Tweet(tweet_id=row.tweet_id, text=row.text, user_id=row.user_id, timestamp=row.timestamp,
                 device=row.device, lang=row.lang, tweet_type=row.tweet_type,
                 like_count=row.like_count,
                 reply_count=row.reply_count,
                 retweet_count=row.retweet_count,
                 quote_count=row.quote_count,
                 topics_ids=row.topics_ids, topics=row.topics, referenced_tweets=row.referenced_tweets)
