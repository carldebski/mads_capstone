import gensim.downloader
import os
from gensim.models import KeyedVectors
from gensim.similarities.fastss import FastSS


def get_similar_words(search_word, n_words=5):
    """
    takes in a word and returns n related words based on their cosine similarity. words that are 
    too similar are identified using the levenshtein edit distance and removed.
    
    args:
        > word (txt): search term
        > n_words (int): number of related words to return

    returns:
        > related_words (dict): a dictionary in the format {search_word: related_words} 

    Other models:
    gensim pre-trained model descriptions: https://github.com/piskvorky/gensim-data 
        > fasttext-wiki-news-subwords-300
        > conceptnet-numberbatch-17-06-300 
        > word2vec-ruscorpora-300
        > word2vec-google-news-300
        > glove-wiki-gigaword-50
        > glove-wiki-gigaword-100
        > glove-wiki-gigaword-200
        > glove-wiki-gigaword-300
        > glove-twitter-25
        > glove-twitter-50
        > glove-twitter-100
        > glove-twitter-200
        > __testing_word2vec-matrix-synopsis
    """
    
    model_path = '../Mainline/models/wiki_word_embeddings'

    # remove spaces from phrases
    search_word = search_word.replace(" ","").lower()

    # retrive word embeddings model
    if os.path.exists(model_path):
        # load the model from local file
        print("Loading saved model...")
        model = KeyedVectors.load(model_path)

    else:
        # download and save the model if not available
        print("Downloading model...")
        model = gensim.downloader.load('fasttext-wiki-news-subwords-300')
        model.save(model_path)

    # safe retrival of the most similar words. Will return itself if the word is not in its vocabulary
    try:
        word_vects = model.most_similar(search_word, topn=n_words*5)

    except KeyError:
        word_vects = [(search_word, 1)]

    # convert all words to lowercase
    word_vects = [(word[0].lower(), word[1]) for word in word_vects]

    # extract the words
    words = [row[0] for row in word_vects]

    # load similar words into fuzzy search query (levenshtein edit distance) 
    fastss = FastSS(words)

    # retrieve similar words with a max edit distance of 2
    matching_words = fastss.query(search_word, max_dist=2)[1]

    # filter out words that match too closely
    words = [row[0] for row in word_vects if row[0] not in matching_words]

    # trim related words to top 50
    related_words = {search_word: words[:n_words]}

    return related_words


# def get_synonyms(term):
#     # retrive the synonyms of a term 
#     synonyms = wn.synonyms(term)

#     return synonyms


# def get_hyponyms(term):
#     # this function will retrive the hyponyms (subordinate synonyms - e.g. Horse is a hyponym for animal) 
#     # from the NLTK 
    
#     hyponyms = []
#     synset = wn.synsets(term)[0]
#     for w in synset.hyponyms():
#         hyponyms.append(w.lemma_names())

#     return hyponyms


# def get_related_queries(term):
#     # retrive the related queries in google trends for a term

#     pytrends = TrendReq(hl='en-US', tz=300) 
#     pytrends.build_payload([term], cat=0, timeframe='today 5-y', geo='', gprop='')
#     df = pytrends.related_queries()

#     return df


# def get_suggested_queries(term):
#     # retrive the related queries in google trends for a term

#     pytrends = TrendReq(hl='en-US', tz=300) 
#     pytrends.build_payload([term], cat=0, timeframe='today 5-y', geo='', gprop='')
#     df = pytrends.suggestions(term)

#     return df



