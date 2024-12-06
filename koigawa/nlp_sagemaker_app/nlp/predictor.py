# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

# This code was not written from scratch by Capstone project members
# The code copied from AWS code sample and then edited to allow for it to work on our code
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py


from __future__ import print_function

import io
import json
import os
import pickle
import signal
import logging
import sys
import traceback
import gensim
from gensim.models import KeyedVectors
from gensim.similarities.fastss import FastSS
import flask
from nltk.corpus import wordnet as wn
import nltk

nltk_data_path = "/usr/share/nltk_data"
os.environ["NLTK_DATA"] = nltk_data_path
nltk.download("wordnet", download_dir=nltk_data_path)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    def __init__(self):
        self.model = get_model()
        nltk.download("wordnet", download_dir=nltk_data_path)
        logger.info(f"Init method called to download nltk to {nltk_data_path}!!")

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            files = os.listdir(model_path)
            logger.info(f"Files in dir ({model_path}): {files}")
            cls.model = KeyedVectors.load(
                "/opt/ml/model/fasttext-wiki-news-subwords-300.model"
            )

        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        nltk.download("wordnet")

        model = cls.get_model()
        n_words = 5
        input = input.replace(" ", "").lower()

        try:
            word_cosine = model.most_similar(input, topn=n_words * 5)

        except KeyError:
            word_cosine = [(input, 1)]

        # convert all words to lowercase
        word_cosine = [(word[0].lower(), word[1]) for word in word_cosine]

        # extract the words
        words = [row[0] for row in word_cosine]

        # load similar words into fuzzy search query (levenshtein edit distance)
        fastss = FastSS(words)

        # create container for unique related words
        unique_words = []

        # for each word, check if it has similarities in the list
        # if any of the similar words have already been identified, continue
        for word in words:
            similar_words = fastss.query(word, max_dist=2)[0]
            similar_words.extend(fastss.query(word, max_dist=2)[1])
            similar_words.extend(fastss.query(word, max_dist=2)[2])

            if set(unique_words) & set(similar_words):
                continue
            elif word == input:
                continue
            else:
                unique_words.append(word)

        # remove any of the same words
        for word in list(fastss.query(input, max_dist=2)[1]):
            try:
                unique_words.remove(word)
            except:
                continue

        unique_words = list(unique_words)

        # refine list based on relevance using semantic similarity based on wordnet path distance
        path_similarities = [
            get_wordnet_path_similarity(input, t) for t in unique_words
        ]
        ranked_path_similarities = sorted(
            list(zip(path_similarities, unique_words)), reverse=True
        )
        ranked_words = [w[1] for w in ranked_path_similarities]

        related_words = (";").join(ranked_words[:n_words])

        return related_words


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = (
        ScoringService.get_model() is not None
    )  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    # Do the prediction
    try:
        data = flask.request.data.decode("utf-8")
        predictions = ScoringService.predict(data)
        return flask.Response(response=predictions, status=200, mimetype="text/plain")
    except Exception as e:
        logger.error(f"Invocations failed: {e}")
        return flask.Response(
            response=f"Failed with {data}", status=415, mimetype="text/plain"
        )


def get_wordnet_path_similarity(search_term, term):
    # this function will retrive the path symilarity of words using wordnet

    try:
        synset1 = wn.synsets(search_term)[0]
        synset2 = wn.synsets(term)[0]
        path_similarity = synset1.path_similarity(synset2)

        return path_similarity
    except IndexError:
        return 0
