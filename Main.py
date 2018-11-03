import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import helpers

# all data processing in the following lines are exclusively for training data ONLY
# all training data are used for the partially supervised classification algorithm
# where sentiments are first labelled and fed into the algorithm then the classification
# algorithm runs on the actual data set (the ones below are preprocessed for initial training)
training_tweets_data = pd.read_json(os.path.dirname(os.path.realpath(__file__)) + '/dataset.json')
training_dataframe = pd.DataFrame.from_dict(training_tweets_data, orient='columns')

# read dataset from a json file
raw_tweets_data = pd.read_json(os.path.dirname(os.path.realpath(__file__)) + '/dataset.json')
tweets_dataframe = pd.DataFrame.from_dict(raw_tweets_data, orient='columns')

# convert tweets to a list
sentences = tweets_dataframe['tweet'].tolist()

# start of k means clustering
# creates instance of *Term Frequency Inverse Document Vectorizer*
vectorizer = TfidfVectorizer(tokenizer=helpers.tokenizer, stop_words='english')
# vectorize sentences
X = vectorizer.fit_transform(sentences)

MAX_CLUSTER = 20
optimum_cluster = {'best_n': 2, 'silhouette_score': 1}
silhouette_scores = []

# try every number of cluster possible to see optimal error from the silhouette score
for num_clusters in range(MAX_CLUSTER)[2:]:
    # use k means to cluster vectorized sentences
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=150)
    kmeans_model.fit(X)
    cluster_labels = kmeans_model.fit_predict(X)

    # get centroids for each cluster
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]

    # get { word, vectorized_word } or human-word per word-vector
    terms = vectorizer.get_feature_names()

    # dictionary - array of { cluster_number: [word-vector] }
    human_clusters = []
    cluster_sentiments = []

    # start of vader sentiment analyzer
    # initialize sentiment analyzer instance
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # populate human clusters with array of [human-words] with index as cluster label
    for i in range(num_clusters):
        cluster = []
        sentiments = []
        # in every cluster, get 8 nearest words
        for word_index in order_centroids[i, :6]:
            # reveal human word from terms[word_vector]
            human_word = terms[word_index]
            # append the word to the cluster
            cluster.append(human_word)

        for word_index in order_centroids[i, :]:
            human_word = terms[word_index]
            # get sentiment from VADER sentiment analyzer
            sentiment = sentiment_analyzer.polarity_scores(human_word)
            # append the resulting sentiment to the sentiment array
            sentiments.append(helpers.extract_sentiment(sentiment))

        human_clusters.append(cluster)
        # append only the prevalent element to the sentiment array
        cluster_sentiments.append(helpers.get_prevalent_element(sentiments))

    validation_score = silhouette_score(X, cluster_labels)

    # check if the silhouette score of this kmeans given this number of clusters
    # is the optimum cluster
    if abs(validation_score) < optimum_cluster['silhouette_score']:
        # if this is the optimum cluster
        # replace the most optimum cluster dictionary
        optimum_cluster['best_n'] = num_clusters
        optimum_cluster['silhouette_score'] = validation_score

    # create a in-memory worksheet containing Cluster, Words, Prevalent Sentiment, and Silhouette Score
    df = pd.DataFrame({'Cluster': range(2, len(human_clusters) + 2),
                       'Words': human_clusters,
                       'Prevalent Sentiment': cluster_sentiments,
                       'Silhouette Score': validation_score})

    # if ./result directory does not exist
    if not os.path.isdir('../result'):
        # create ./result directory
        os.mkdir('../result')

    # directory + filename of the output of this clustering algorithm
    directory_to_save = '../result/output-' + str(num_clusters) + '.xls'
    # save to an excel sheet
    df.to_excel(directory_to_save, sheet_name='Results', index=False)

    # append to silhouette score the current cluster's silhouette score
    silhouette_scores.append(validation_score)

# after having found the optimum number of cluster
# write a text file for it
output = open('../result/scores.txt', 'w')
for index, score in enumerate(silhouette_scores):
    output.write(str(index + 2) + ': ' + str(score) + '\n')
output.close()
