
# re string for all punctuations except ' . and -
def get_vocab_removal_rules():
    return r'[^\w\'\s-]|\d'


# re string for sentence split rules
def get_sentence_split_rules():
    return r'[\n\r]+|[.!?:](?=\s+[0-9A-Z]|[\n\r]+|$)'
