# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

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

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            files = os.listdir(model_path)
            logger.info(f"Files in dir ({model_path}): {files}")
            cls.model = KeyedVectors.load("/opt/ml/model/fasttext-wiki-news-subwords-300.model")

        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        model = cls.get_model()
        n_words = 5

        words_vects = model.most_similar(input, topn=n_words*5)

        words = [row[0] for row in words_vects]

        # load similar words into fuzzy search query (levenshtein edit distance) 
        fastss = FastSS(words)

        # retrieve similar words with a max edit distance of 1
        matching_words = fastss.query(input, max_dist=1)[1]

        # filter out words that match too closely
        words = [row[0] for row in words_vects if row[0] not in matching_words]

        related_words = (";").join(words[:n_words])

        return related_words


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

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
        return flask.Response(response=f"Failed with {data}", status=415, mimetype="text/plain")
