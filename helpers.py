def extract_sentiment(results):
    k = 0
    for sentiment in results:
        if k == 0 or results[sentiment] > results[k]:
            k = sentiment

    if k == 'neu':
        return 'neutral'
    elif k == 'pos':
        return 'positive'
    elif k == 'neg':
        return 'negative'

    return 'n/a'


def tokenizer(text):
    import nltk
    words = nltk.word_tokenize(text)
    tokens = []
    for word in words:
        if len(word) >= 3:
            tokens.append(word)
    return tokens


def get_prevalent_element(elements):
    max_count_element = elements[0]
    for element in elements:
        if elements.count(element) > elements.count(max_count_element):
            max_count_element = element
    return max_count_element


