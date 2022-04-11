class User:
    def __init__(self, user_id, name, username, followers, following, tweet_count, verified, created_at):
        self.user_id = user_id
        self.name = name
        self.username = username
        self.followers = followers
        self.following = following
        self.tweet_count = tweet_count
        self.verified = verified
        self.created_at = created_at

    def __repr__(self):
        return "User(id: {0}, name: {1}, username: {2}) \n".format(self.user_id, self.name, self.username)

    def __str__(self):
        return "id: {0}, name: {1}, username: {2}".format(self.user_id, self.name, self.username)

    def __eq__(self, other):
        return self.user_id == other.user_id


def extract_user(data):
    return User(user_id=str(data.id), name=str(data.name), username=str(data.username),
                followers=data['public_metrics']['followers_count'],
                following=data['public_metrics']['following_count'],
                tweet_count=data['public_metrics']['tweet_count'],
                verified=data['verified'],
                created_at=data['created_at'])


def csv_row_to_user(row):
    return User(user_id=row.user_id, name=row.name, username=row.username, followers=row.followers,
                following=row.following, tweet_count=row.tweet_count, verified=row.verified, created_at=row.created_at)
