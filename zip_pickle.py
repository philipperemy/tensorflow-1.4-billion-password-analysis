import gzip
import pickle


def save(filename, obj):
    # save objects into a compressed disk file
    fil = gzip.open(filename, 'wb')
    pickle.dump(file=fil, obj=obj)
    fil.close()


def load(filename):
    # reload objects from a compressed disk file
    fil = gzip.open(filename, 'rb')
    while True:
        try:
            yield pickle.load(fil)
        except EOFError:
            break
    fil.close()
