import pickle


class Pickle:
    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def save(path, obj):
        with open(path, "wb") as file:
            pickle.dump(obj, file)
