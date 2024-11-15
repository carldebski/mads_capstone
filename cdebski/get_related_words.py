import gensim.downloader
import os
from gensim.models import KeyedVectors
from gensim.similarities.fastss import FastSS
from nltk.corpus import wordnet as wn


def get_similar_words(search_word, n_words=5):
    """
    takes in a word and returns n related words based on their cosine similarity. words that are 
    too similar are identified using the levenshtein edit distance and removed. words are further ranked by their 
    wordnet path difference. 
    
    args:
        > word (txt): search term
        > n_words (int): number of related words to return

    returns:
        > related_words (dict): a dictionary in the format {search_word: related_words} 

    Other models available:
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
        word_cosine = model.most_similar(search_word, topn=n_words*5)

    except KeyError:
        word_cosine = [(search_word, 1)]

    # convert all words to lowercase
    word_cosine = [(word[0].lower(), word[1]) for word in word_cosine]
    print(word_cosine)
    # extract the words
    words = [row[0] for row in word_cosine]

    # load similar words into fuzzy search query (levenshtein edit distance) 
    fastss = FastSS(words)

    # create container for unique related words
    unique_words = set()

    # for each word, check if it has similarities in the list
    # if any of the similar words have already been identified, continue
    for word in words:
        similar_words = fastss.query(word, max_dist=2)[1]

        if unique_words & set(similar_words):
            continue
        else:
            unique_words.add(word)

    # remove any of the same words
    for word in list(fastss.query(search_word, max_dist=2)[1]):
        try:
            unique_words.remove(word)
        except:
            continue

    unique_words = list(unique_words)

    # refine list based on relevance using semantic similarity based on wordnet path distance
    path_similarities = [get_wordnet_path_similarity(search_word, t) for t in unique_words]
    ranked_path_similarities = sorted(list(zip(path_similarities, unique_words)), reverse=True)
    ranked_words = [w[1] for w in ranked_path_similarities]

    # trim related words to top 50
    related_words = {search_word: ranked_words[:n_words]}

    return related_words


def get_hypernyms(term):
    # this function will retrive the hypernyms

    try:
        synset = wn.synsets(term)[0]
        hypernyms = synset.hypernyms()[0].lemma_names()[0]
            
        return hypernyms
    
    except IndexError:
        pass


def get_wordnet_path_similarity(search_term, term):
    # this function will retrive the path symilarity of words using wordnet

    try:
        synset1 = wn.synsets(search_term)[0]
        synset2 = wn.synsets(term)[0]
        path_similarity = synset1.path_similarity(synset2)

        return path_similarity
    
    except IndexError:
        
        return 0
        
    
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


def get_synonyms(term):
    # retrive the synonyms of a term 
    synonyms = wn.synonyms(term)

    return synonyms


def get_hyponyms(term):
    # this function will retrive the hyponyms (subordinate synonyms - e.g. Horse is a hyponym for animal) 
    # from the NLTK 
    
    hyponyms = []
    synset = wn.synsets(term)[0]
    for w in synset.hyponyms():
        hyponyms.append(w.lemma_names())

    return hyponyms



