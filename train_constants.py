#
# train_constants.py: Contains the constants necessary for the encoding and the training phases.
#

# Maximum password length. Passwords greater than this length will be discarded during the encoding phase.
ENCODING_MAX_PASSWORD_LENGTH = 12

# Maximum number of characters for encoding. By default, we use the 80 most frequent characters and
# we bin the other ones in a OOV (out of vocabulary) group.
ENCODING_MAX_SIZE_VOCAB = 80
