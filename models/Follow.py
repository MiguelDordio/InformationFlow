class Follow:
    def __init__(self, account_id, follower_id):
        self.account_id = account_id
        self.follower_id = follower_id

    def __repr__(self):
        return "Follow(account_id: {0}, follower_id: {1}) \n".format(self.account_id, self.follower_id)

    def __str__(self):
        return "Follow(account_id: {0}, follower_id: {1}) \n".format(self.account_id, self.follower_id)

    def __eq__(self, other):
        return self.account_id == other.account_id and self.follower_id == other.follower_id