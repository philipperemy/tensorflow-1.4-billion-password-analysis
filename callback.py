class Callback:
    def __init__(self):
        pass

    def call(self, emails_passwords):
        # emails_passwords = list of tuples
        pass


class ReducePasswordsOnSimilarEmailsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def finalize_cache(self):
        keys = list(self.cache.keys())
        for key in keys:
            self.cache[key] = list(self.cache[key])
            if len(self.cache[key]) <= 1:
                del self.cache[key]

    def call(self, emails_passwords):
        for (email, password) in emails_passwords:
            if email not in self.cache:
                self.cache[email] = set()
            self.cache[email].add(password)

    def debug(self):
        pass
        # print('{0} total number of unique emails.'.format(len(self.cache)))
