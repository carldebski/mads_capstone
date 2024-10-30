

# import nltk
# from nltk.corpus import wordnet as wn
# from pytrends.request import TrendReq
# nltk.download('wordnet')
import gensim.downloader
from gensim.similarities.fastss import FastSS


def get_similar_words(search_word):
    """
    takes in a word and returns fifty related words based on their cosine similarity. words that are 
    too similar are identified using the levenshtein edit distance and removed.
    
    parameters:
        > word (text): search term

    returns:
        > related_words (dict): a dictionary in the format {search_word: related_words} 
    """
    

    # download the embeddings
    vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')

    # retrieve the 100 most similar words 
    words_vects = vectors.most_similar(search_word, topn=100)
    
    # extract the words
    words = [row[0] for row in words_vects]

    # load similar words into fuzzy search query (levenshtein edit distance) 
    fastss = FastSS(words)

    # retrieve similar words with a max edit distance of 1
    matching_words = fastss.query(search_word, max_dist=1)[1]

    # filter out words that match too closely
    words = [row[0] for row in words_vects if row[0] not in matching_words]

    # trim related words to top 50
    related_words = {search_word: words[:5]}

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



