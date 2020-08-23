"""
Author: Dor Meir
"""
import datetime
import os
import pickle
from contextlib import redirect_stdout

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from sklearn.metrics import silhouette_samples
import re
import gensim
from collections import defaultdict
from gensim.parsing.preprocessing import preprocess_documents
import yaml

with open('configs.yaml') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)


def clean_search_query(query):
    """
    Receives a search query, returns it without punctuation, spaces in the sides and lower cased
    :param query: a search query
    :return: cleaned search query
    """
    # removing punctuation
    query = re.sub(configs['clean_search_query']['punctuation_to_remove'], ' ', query)
    # removing spaces from sides of text
    query = query.strip()
    # lower-casing
    query = query.lower()
    return query


def import_df(fname=configs['fname_for_import']):
    """
    Import raw queries df and report to user
    :param fname: the csv file name
    :return: the pandas data frame of the csv
    """
    print(f"Importing queries df '{fname}' . . .")
    return pd.read_csv(fname)


def pre_process_df(df, date, drop_cols=True, remove_na=True, remove_duplicate=True, date_feature=None,
                   group_by_feature=False,
                   remove_short_queries=True, drop_numeric_queries=True, clean_search_queries=True,
                   delete_stopwords=False, save_df=True, load_existing=False, fname=None,
                   query_feature=configs['df_features']['query'], remove_i_will=True):
    """
    Receives the search query df and returns preprocessed df cleaned from na's, duplicates, date converted
     or grouped df by a specific feature
    :param remove_i_will: If true, removes all queries starting in "i will".
    :param date: The date of the search queries (will be updated in each row of the 'search_date' column
    :param remove_short_queries: if true, remove queries with shorter characters than the minimal n_gram
    :param drop_cols: if true, removes the columns of
    :param fname: Name of the proccessed df file to save/load
    :param delete_stopwords: if true, removes stop words. Yuval said that might not be a good practice.
    :param load_existing: if true, imports existing preprocessed df
    :param save_df: if true, saves df
    :param clean_search_queries: removes punctuation, spaces in the sides and lower case everything
    :param query_feature: The name of the query feature, for drop_numeric_query
    :param drop_numeric_queries: Drop *only* numeric queries (i.e. 441, 160) rows
    :param group_by_feature: The feature to group by over the df, and than sum the grouped values.
    :param df: search queries df
    :param remove_na: True if will be removed
    :param remove_duplicate: True if will be removed
    :param date_feature: Name of feature if it should be converted to to time
    :return: the preprocessed df
    """
    if not fname:
        fname = 'df' + '_' + configs['fname_for_import']
    if not load_existing:
        print('Pre-processing df . . .')
        print(df.shape)
        if drop_cols:
            print('    Drop columns . . .')
            df = df.copy()
            df.drop(columns=configs['df_features']['cols_to_drop'], inplace=True)
        df.dropna(inplace=True) if remove_na else False
        if clean_search_queries:
            print(
                f'    clean_search_queries (removes punctuation, spaces in the sides and lower case everything) . . .')
            df = df.copy()
            df[query_feature] = df[query_feature].apply(clean_search_query)
        if remove_short_queries:
            print(f'    remove queries with very small number of chars (smaller than the number of n-grams')
            minimum_len_query = int(configs['bow']['smallest_ngram'])
            df = df.copy()
            df = df[df[query_feature].transform(lambda x: len(x)) >= minimum_len_query]
        if delete_stopwords:
            print(f'    delete stop words . . .')
            df = df.copy()
            df[query_feature] = df[query_feature].apply(remove_stopwords)
        if group_by_feature:
            print(f'    convert group. . .')
            df_sum = df.groupby(by=[configs['df_features']['query']]).sum()[
                [configs['df_features']['no_searches'],
                 configs['df_features']['search_with_clicks'],
                 configs['df_features']['listings_with_orders'],
                 configs['df_features']['orders'],
                 configs['df_features']['revenue']]]
            df_avg = df.groupby(by=[configs['df_features']['query']]).mean()[
                configs['df_features']['total_number_of_search_results']]
            df_cat = df.groupby(by=[configs['df_features']['query']]).first()[configs['df_features']['sub_category']]
            df = pd.concat([df_sum, df_avg, df_cat], axis=1).reset_index()
        if drop_numeric_queries:
            print(f'    drop numeric queries (i.e. 144) . . .')
            df = df[~df[query_feature].str.isnumeric()]
        if remove_i_will:
            print(f'    drop search queries starting with "i will" . . .')
            df = df[~df[query_feature].str.startswith("i will")]
        df[configs['df_features']['date']] = date
        print(df.shape)
        if save_df:
            df_save(df, fname)
    else:
        df = df_load(fname)
    return df


def nlp_process_words_counts_and_searches(df, fit_vectorizer=True, save_vectorizer=True, fname=None,
                                          query_feature=configs['df_features']['query'],
                                          no_searches_feature=configs['df_features']['no_searches']):
    """
    Receives the queries df and parameters, returns the words counts and word searches in the entire corpus
    (disregarding the sentence structure of the corpus).
    :param df: the queries df
    :param fit_vectorizer: if true, retrain the counter. if false, loads the counter from fname.
    :param save_vectorizer: if true, saves the pickles the vectorizer and the vectorized queries
    :param fname: if fit_vectorizer is False, loads the model from this file name
    :param query_feature: The name of the queries feature
    :param no_searches_feature: The name of the number of searches feature
    :return: a data frame with the queries word counts and number of searches
    """
    if fit_vectorizer:
        print('Fitting words_counts_and_searches . . .')
        words_counts_in_corpus = defaultdict(int)
        words_searches_in_corpus = defaultdict(int)
        if not stopwords:
            nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        for i, sentence in enumerate(df[query_feature]):
            for word in sentence.split():
                if word not in stop_words:
                    words_counts_in_corpus[word.lower()] += 1
                    words_searches_in_corpus[word.lower()] += df[no_searches_feature].iloc[i]
        res = pd.DataFrame([words_counts_in_corpus, words_searches_in_corpus],
                           index=[configs['df_features']['counts'], configs['df_features']['no_searches']]).T
        if save_vectorizer:
            print(f'Saving words counts: {fname} . . .')
            pickle_something(res, fname)
    else:  # load existing
        print(f'Load existing words counts: {fname} . . .')
        res = unpickle_something(fname)
    return res


def nlp_process_bow(df, fit_vectorizer=True, save_vectorizer=True, fname=None,
                    query_feature=configs['df_features']['query'], analyzer=configs['bow']['analyzer'],
                    min_df=configs['bow']['min_df'], smallest_ngram=configs['bow']['smallest_ngram'],
                    largest_ngram=configs['bow']['largest_ngram']):
    """
    Receives the queries df and parameters, returns bag of words (word counts) of the query_feature + dictionary
    :param df: the queries df
    :param min_df: A TF-IDF parameter. Ignore terms that have a document frequency strictly lower than this threshold.
    :param largest_ngram: largest ngram in the ngram range used for the analyzer
    :param smallest_ngram: smallest ngram in the ngram range used for the analyzer
    :param analyzer: The embeddings building blocks. default is 'word' (gensim default) for doc2vec
    :param fit_vectorizer: if true, retrain the bow. if false, loads the existing one from fname.
    :param save_vectorizer: if true, saves the pickles the vectorizer and the vectorized queries
    :param fname: if fit_vectorizer is False, loads the model from this file name
    :param query_feature: The name of the queries feature
    :return: the word counter bow model, the dictionary and the queries transformed
    """

    if fit_vectorizer:
        print('   Fitting CountVectorizer . . .')
        if len(df) <= 1000:
            min_df = 1
        cv = CountVectorizer(min_df=min_df,
                             analyzer=analyzer,
                             ngram_range=(smallest_ngram, largest_ngram))
        cv.build_analyzer()
        _ = cv.fit(df[query_feature])
        dictionary = cv.get_feature_names()
        queries_bow = cv.transform(df[query_feature])
        if save_vectorizer:
            print(f'Saving vectorizer: {fname} . . .')
            pickle_something(cv, fname)
            pickle_something(queries_bow, 'queries_' + fname)
    else:
        print(f'Load existing vectorizer: {fname} . . .')
        cv = unpickle_something(fname)
        dictionary = cv.get_feature_names()
        queries_bow = cv.transform(df[query_feature])
    return cv, dictionary, queries_bow


def nlp_process_tfidf(df, queries_bow, fit_vectorizer=True, save_vectorizer=True, fname=None,
                      query_feature=configs['df_features']['query']):
    """
    Receives the queries df and queries transformed by bow and parameters,
    returns tfidf of the query_feature
    :param queries_bow: the count vectorizer bow of the queries
    :param df: the queries df
    :param fit_vectorizer: if true, retrain the bow. if false, loads the existing one from fname.
    :param save_vectorizer: if true, saves the pickles the vectorizer and the vectorized queries
    :param fname: if fit_vectorizer is False, loads the model from this file name
    :param query_feature: The name of the queries feature
    :return: the tfidf model and the queries transformed
    """
    if fit_vectorizer:
        print('   Fitting TfidfTransformer . . .')
        tfidf = TfidfTransformer()
        _ = tfidf.fit(queries_bow)
        queries_tfidf = tfidf.transform(queries_bow)
        if save_vectorizer:
            print(f'Saving vectorizer: {fname} . . .')
            pickle_something(tfidf, fname)
            pickle_something(queries_tfidf, 'queries_' + fname)
    else:
        print(f'Load existing vectorizer: {fname} . . .')
        tfidf = unpickle_something(fname)
        queries_tfidf = tfidf.transform(df[query_feature])
    return queries_tfidf, tfidf


def sentences_tokenizer(queries, cv):
    """
    Receives pre-processed queries and a CountVectorizer, returns the queries tokenized
    :param queries: an array of queries, already pre-processed so that the minial character will be the minimal ngram
    range
    :param cv: the CountVectorizor
    :return: a list of lists representing each query as a list of n-grams in the query
    """
    tokenized_sentences = []
    for i, query in enumerate(queries):
        cv.fit_transform([query])
        tokenized_sentences.append(cv.get_feature_names())
    return tokenized_sentences


def nlp_process_doc2vec(df, fit_vectorizer=True, save_vectorizer=True, fname=None,
                        query_feature=configs['df_features']['query'], analyzer=configs['bow']['analyzer'],
                        windows=configs['doc2vec']['window'],
                        vector_size=configs['doc2vec']['vector_size'], min_count=configs['doc2vec']['min_count'],
                        epochs=configs['doc2vec']['epochs'], smallest_ngram=configs['bow']['smallest_ngram'],
                        largest_ngram=configs['bow']['largest_ngram']):
    """
    Receives the queries df and queries transformed by bow and parameters,
    returns tfidf of the query_feature
    :param windows: The maximum distance between the current and predicted char within a sentence.
    :param analyzer: The embeddings building blocks. default is 'char' (from sklearn), can be also 'word' (gensim default)
    :param epochs: Number of epochs for the doc2vec Neutral network
    :param min_count: consider only words with higher frequency than min_count
    :param vector_size: vector size for latent layer of the doc2vec Neutral network
    :param df: the queries df
    :param fit_vectorizer: if true, retrain the bow. if false, loads the existing one from fname.
    :param save_vectorizer: if true, saves the pickles the vectorizer and the vectorized queries
    :param fname: if fit_vectorizer is False, loads the model from this file name
    :param query_feature: The name of the queries feature
    :param smallest_ngram: smallest ngram in the ngram range used for the analyzer
    :param largest_ngram: largest ngram in the ngram range used for the analyzer
    :return: the doc2vec model and the queries transformed
    """
    if fit_vectorizer:
        print('Fitting doc2vec . . .')
        if analyzer == 'word':
            print("   Creating gensim's default word embeddings for the train data . . .")
            tokenized_sentences = preprocess_documents(df[query_feature])
        else:  # analyzer == 'char'
            print(
                f"   Creating sklearn char analyzer with ({smallest_ngram}-{largest_ngram}) ngrams for the train data "
                f". . .")
            cv = CountVectorizer(  # min_df=configs['bow']['min_df'],
                analyzer=configs['bow']['analyzer'],  # This is optimal
                ngram_range=(smallest_ngram, largest_ngram))
            tokenized_sentences = sentences_tokenizer(df[query_feature], cv)

        train_data = list(create_tagged_document(tokenized_sentences))
        model_doc2vec = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs,
                                                      windows=windows)
        model_doc2vec.build_vocab(train_data)
        if save_vectorizer:
            print(f'Saving vectorizer: {fname} . . .')
            model_doc2vec.save(fname)
    else:
        print(f'Load existing vectorizer: {fname} . . .')
        model_doc2vec = gensim.models.doc2vec.Doc2Vec.load(fname)
    return model_doc2vec, model_doc2vec.docvecs.vectors_docs


def nlp_process(df, words_counts_and_searches=False, bow=False, tfidf=False, doc2vec=True, fit_vectorizer=True,
                save_vectorizer=True, fname=None, query_feature=configs['df_features']['query'],
                no_searches_feature=configs['df_features']['no_searches'], analyzer='Default',
                smallest_ngram='Default', largest_ngram='Default', min_df='Default',
                windows=configs['doc2vec']['window'],
                vector_size=configs['doc2vec']['vector_size'], min_count=configs['doc2vec']['min_count'],
                epochs=configs['doc2vec']['epochs']):
    """
    Receives the df, returns one type of vectorized representation of the search queries
    :param save_vectorizer: if true, saves the pickles the vectorizer and the vectorized queries
    :param doc2vec: if true returns the sentences vectors trained on doc2vec algorithm
    :param fit_vectorizer: if true, retrain the vectorizer. if false, loads the vectorizer from fname, but fits the
    vector on the df provided by the user.
    :param fname: if fit_vectorizer is False, loads the model from this file name
    :param words_counts_and_searches: if True returns the words counts and word searches in the entire corpus
    (disregarding the sentence structure of the corpus).
    :param stop_words: None for not removing stop_words, default is removing english stop_words
    :param df: the queries df
    :param bow: bool for returning bag of words of the query_feature + dictionary
    :param tfidf:  bool for returning tfidf of the query_feature + dictionary
    :param query_feature: The name of the queries feature
    :param no_searches_feature: The name of the number of searches feature
    :param largest_ngram: largest ngram in the ngram range used for the analyzer
    :param smallest_ngram: smallest ngram in the ngram range used for the analyzer
    :param windows: A doc2vec parameter for the number of words left and right of each word to input the embedding model
    :param vector_size: A doc2vec parameter for the size of the vector of each query (size of the latent layer)
    :param epochs: A doc2vec parameter. Number of iterations (epochs) over the corpus.
    :param min_count: A doc2vec parameter.  Ignores all words with total frequency lower than this.
    :param min_df: A TF-IDF parameter. Ignore terms that have a document frequency strictly lower than this threshold.
    :param analyzer: The embeddings building blocks. default is 'word' (gensim default) for doc2vec
    :return: nlp processed query feature column, a dictionary for bow or tfidf, and the vectorizer for bow, tfidf,
    doc2vec
    """
    # create the file name for saving or loading a model
    vectorizer_name = 'not_defined'
    if not fname:
        if words_counts_and_searches:
            vectorizer_name = 'words_counts_and_searches'
        elif bow:
            vectorizer_name = 'bow'
        elif tfidf:
            vectorizer_name = 'tfidf'
        elif doc2vec:
            vectorizer_name = 'doc2vec'
        file_suffix = configs['fname_for_import'] + configs['PICKLE_SUFFIX']
        fname = vectorizer_name + '_' + file_suffix

    # word counts and searches processing
    if words_counts_and_searches:
        return nlp_process_words_counts_and_searches(df, fit_vectorizer=fit_vectorizer, save_vectorizer=save_vectorizer,
                                                     fname=fname, query_feature=query_feature,
                                                     no_searches_feature=no_searches_feature)
    # bow processing
    if bow or tfidf:
        if analyzer == 'Default':
            analyzer = configs['tfidf_analyzer']
        if smallest_ngram == 'Default':
            smallest_ngram = configs['bow']['smallest_ngram']
        if largest_ngram == 'Default':
            largest_ngram = configs['bow']['largest_ngram']
        if min_df == 'Default':
            min_df = configs['bow']['min_df']
        cv, dictionary, queries_bow = nlp_process_bow(df, fit_vectorizer=fit_vectorizer,
                                                      save_vectorizer=save_vectorizer,
                                                      fname=fname,
                                                      query_feature=query_feature,
                                                      analyzer=analyzer,
                                                      min_df=min_df,
                                                      smallest_ngram=smallest_ngram,
                                                      largest_ngram=largest_ngram)
        if bow:
            return queries_bow, dictionary, cv
        # tfidf processing (first part is bow pre-processing)
        elif tfidf:
            queries_tfidf, tfidf = nlp_process_tfidf(df, queries_bow, fit_vectorizer=fit_vectorizer,
                                                     save_vectorizer=save_vectorizer, fname=fname,
                                                     query_feature=query_feature)
            return queries_tfidf, dictionary, tfidf

    # doc2vec processing
    if doc2vec:
        if analyzer == 'Default':
            analyzer = configs['doc2vec_analyzer']
        if smallest_ngram == 'Default':
            smallest_ngram = configs['doc2vec']['smallest_ngram']
        if largest_ngram == 'Default':
            largest_ngram = configs['doc2vec']['largest_ngram']
        model_doc2vec, queries_doc2vec = nlp_process_doc2vec(df, fit_vectorizer=fit_vectorizer,
                                                             save_vectorizer=save_vectorizer, fname=fname,
                                                             query_feature=query_feature,
                                                             analyzer=analyzer,
                                                             windows=windows,
                                                             vector_size=vector_size,
                                                             min_count=min_count,
                                                             epochs=epochs,
                                                             smallest_ngram=smallest_ngram,
                                                             largest_ngram=largest_ngram)
        return queries_doc2vec, model_doc2vec


def create_tagged_document(list_of_list_of_words):
    """
    Create the tagged document needed for Doc2Vec using the list of list of words in the corpus
    :param list_of_list_of_words: tokenized_sentences - the output of gensim's preprocess_documents(df[query_feature])
    :return: Generator of the tagged sentences
    """
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


def feature_engineering(df, query_lens=True, clicks_per_search=True, price=True, query_feature='search_query',
                        save_df=True, load_existing=False, fname=None,
                        search_with_clicks_feature=configs['df_features']['search_with_clicks'],
                        no_searches_feature=configs['df_features']['no_searches'],
                        revenue_feature=configs['df_features']['revenue'],
                        orders_feature=configs['df_features']['orders']):
    """
    Engineer new features for the df
    :param revenue_feature: For bq-results-20200712-141041-tqwab845scx, revenue per listing
    :param orders_feature: For bq-results-20200712-141041-tqwab845scx, order per listing
    :param price: For bq-results-20200712-141041-tqwab845scx, if both orders and revenues exist, create price
    :param df: the search queries df
    :param fname: Name of the proccessed df file to save/load
    :param load_existing: if true, imports existing preprocessed df
    :param save_df: if true, saves df
    :param query_lens: no. of characters of the search query
    :param clicks_per_search: no. of clicks in a search listing of a keyword divided by no. of searches of a keyword
    :param query_feature: Name of the feature with the search queries' text
    :param search_with_clicks_feature: Name of the the no. of searches with clicks features
    :param no_searches_feature: Name number of searches feature
    :return: the df with the engineered features
    """
    if not fname:
        fname = 'df_engineered' + '_' + configs['fname_for_import']
    if not load_existing:
        if query_lens:
            df[configs['df_features']['char_lengths']] = df[query_feature].apply(lambda s: len(s))
        if clicks_per_search:
            df[configs['df_features']['clicks_per_search']] = df[search_with_clicks_feature] / df[no_searches_feature]
        if price:
            df[configs['df_features']['price']] = df[revenue_feature] / df[orders_feature]
        if save_df:
            df_save(df, fname)
    else:
        df = df_load(fname)
    return df


def get_separate_days_df_dict(df, date_feature=configs['df_features']['date']):
    """
    Receives queries df, returns a dict with each members as a data frame filtered by one day (in the order of days)
    :param df: the queries df
    :param date_feature: the date feature
    :return: A dict where each member is the df filtered by day
    """
    days = df[date_feature].value_counts().sort_index().index
    df_day = {}
    for i, x in enumerate(days):
        df_day[i] = df[df.search_date == x].reset_index(drop=True)
    return df_day


def plot_feature_to_keywords(df, keyword_feature=configs['df_features']['query'],
                             feature_to_plot=configs['df_features']['no_searches'],
                             return_df=False, n_search_queries=configs['plot_feature']['n_search_queries'],
                             figsize=configs['plot_feature']['figsize']):
    """
    Receives queries df, plots the feature_to_plot over the keyword feature, possibly return the plot's df
    (keyword feature as the index, feature to plot as the column)
    :param df: queries df
    :param keyword_feature: The feature in the x axis of the plot
    :param feature_to_plot: The feature in the y axis of the plot
    :param return_df: Bool for returning the df that was plotted
    :param n_search_queries: number of ticks in the x axis (biggest values for keywords in the feature_to_plot)
    :param figsize: size of the figure of the plot
    :return: plot keyword_feature over feature_to_plot, and possibly return the df of the plot
    """
    plt.figure(figsize=figsize)
    plotted_df = df[[keyword_feature, feature_to_plot]].set_index(keyword_feature)[feature_to_plot]. \
        sort_values(ascending=False)
    plotted_df[:n_search_queries].plot(rot=configs['plot_feature']['rot'])
    plt.ylabel(feature_to_plot)
    plt.title(f"""The {feature_to_plot} distribution over the top {n_search_queries} {keyword_feature}""")
    plt.tight_layout()
    plt.show()
    if return_df:
        return pd.DataFrame(plotted_df)


def show_labels_svd(queries, queries_vectorized, labels, load_existing=False, save=True, fname=None,
                    pct_queries_to_show=configs['svd']['pct_queries_to_show'],
                    n_components=configs['svd']['n_components'], title=None):
    """
    Receives the row queries, their vectors and their cluster labels and plots the cluster with annotations using
    svd two components of the vectors
    :param fname: the name of the saved model file
    :param save: if true, saves the model
    :param load_existing: if True, loads existing model and doesn't fit
    :param title: A title to show for the plot (e.g. f"WCSS is {kmeans.inertia_}")
    :param queries: list of search raw search queries to show in the annotations
    :param queries_vectorized: bow, tfidf, word embedding or hashings of the search queries
    :param labels: list the cluster labels for the search queries
    :param pct_queries_to_show: percentage of annotations to show (too many - can't read)
    :param n_components: no. of components for performing the svd dimensionality reduction
    """
    if not fname:
        file_suffix = configs['fname_for_import'] + configs['PICKLE_SUFFIX']
        fname = 'svd' + '_' + file_suffix
    if not load_existing:
        svd = TruncatedSVD(n_components=n_components, n_iter=configs['svd']['n_iter'],
                           random_state=configs['svd']['random_state'])
        Y = svd.fit_transform(queries_vectorized)
        if save:
            pickle_something(svd, fname)
    else:
        svd = unpickle_something(fname)
        Y = svd.fit_transform(queries_vectorized)

    fig = plt.figure(figsize=configs['svd']['figsize'])
    ax = fig.add_subplot(111)
    ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=configs['svd']['cmap'])
    ax.set_title(title, fontsize=configs['svd']['fontsize']) if title else False
    idx_to_show = np.random.choice(np.arange(len(queries)), size=round(pct_queries_to_show * len(queries)))
    for i in idx_to_show:
        ax.annotate(queries[i], xy=(Y[i, 0], Y[i, 1]), size=configs['svd']['annotate_size'])
    plt.show()


def get_top_words_dict(word_counts_searches,
                       top_words_searches_threshold=configs['top_words_dict']['searches_threshold']):
    """
    Receives a word_counts_searches (out of nlp_process(words_counts_and_searches=True) and top_words_counts_threshold,
    top words dictionary that fits the threshold condition.
    :param word_counts_searches: data frame of each word in the corpus, the number of time it appears in the corpus
    and the total number of searches for all the word's search queries.
    :param top_words_searches_threshold: The threshold we want our top words counts to cover - for threshold of 90%
    and search_gueries_new.csv, the number of words in top_n_words_dict is 10139
    :return: top_words_dict, containing keys of words and their values as their ranks of frequency in the corpus (0 is
    out of vocabulary, 1 is the most frequent word, 2 is the next frequent)
    """
    n_top_words = sum(
        word_counts_searches.counts.cumsum() / sum(word_counts_searches.counts) <= top_words_searches_threshold)
    top_words_dict = {'OOV': 0}
    for i, word in enumerate(word_counts_searches.counts.sort_values(ascending=False)[:n_top_words].index.tolist(), 1):
        top_words_dict[word] = i
    return top_words_dict


def words_frequencies_baseline_classifier(df, load_existing=False, save_model=True, fname=None,
                                          top_words_searches_threshold=configs['top_words_dict']['searches_threshold'],
                                          query_feature=configs['df_features']['query']):
    """
    Predicts labels using the naive 1-grams frequencies of single words in the corpus - each sentence will be classified to
    a cluster named after the word in it which is most frequent in the corpus.
    The function receives the queries df and a threshold for the top words in the corpus, returns the labels of each query, each
    label matches a value in a key in the top_words_dict that is also returned.
    :param df: the search queries data frame
    :param top_words_searches_threshold: the threshold of words in the corpus our dictionary will cover. For threshold of
    :param fname: the name of the saved model file
    :param save_model: if true, saves the model
    :param load_existing: if True, loads existing model and doesn't fit
    90% in search_gueries_new.csv, the number of words in top_words_dict is 10139.
    :param query_feature: The name of the query feature
    :return: the naive labels of the queries (as numbers tokens), and the dictionary for the labels.
    """
    if not fname:
        file_suffix = configs['fname_for_import'] + configs['PICKLE_SUFFIX']
        fname = 'words_frequencies_labels' + '_' + file_suffix
    fname_dict = fname + '_top_words_dict'
    if not load_existing:
        word_counts_searches = nlp_process(df, words_counts_and_searches=True)
        top_words_dict = get_top_words_dict(word_counts_searches,
                                            top_words_searches_threshold=top_words_searches_threshold)
        labels = []
        for search_query in df[query_feature]:
            search_query_top_words = []
            if any([word in top_words_dict for word in search_query.split()]):
                for word in search_query.split():
                    if word in top_words_dict:
                        search_query_top_words.append(top_words_dict[word])
                labels.append(min(search_query_top_words))
            else:
                labels.append(top_words_dict['OOV'])
        if save_model:
            pickle_something(labels, fname)
            pickle_something(top_words_dict, fname_dict)
    else:
        print(f'Load existing model {fname} . . .')
        labels = unpickle_something(fname)
        top_words_dict = unpickle_something(fname_dict)

    return labels, top_words_dict


def find_optimal_clusters_kmeans(queries_bow, weights_number_of_searches, load_existing=False, save_models=True,
                                 fname=None,
                                 n_samples=configs['find_clusters_kmeans']['n_samples'], show_plot=True,
                                 compute_silhouette=True,
                                 n_clusters_list=configs['find_clusters_kmeans']['n_clusters']):
    """
    Receives the queries vector and plots the wcss (Kmeans inertia) and silhouette coefficients over
    the list of n_clusters_list, in order to find the optimal number of clusters (the number of clusters
    in which the improvement in WCSS (lower) and silouette (higher) is the biggest improvement out of all possible
    number of clusters. It also returns the silhouette_per_n_clusters_per_point, a number between -1 and +1 that
    evaluates for each point in sample for each cluster, how much this point relates to the cluster it was assigned
    to and how much it is far away from the next closest cluster (bigger and positive is good fit, lower and negative
    means a bad fit).
    :param weights_number_of_searches: queries df number of searches, used as kmeans weights
    :param show_plot: bools for showing the plot, if False only returns the computed values for the n_clusters_list
    :param queries_bow: the queries vector (bow, tfidf, word embedding, hashing)
    :param fname: the name of the saved model file
    :param save_models: if true, saves the model
    :param load_existing: if True, loads existing model and doesn't fit
    :param n_samples: the number of samples to compute on the wcss and silhouette
    :param compute_silhouette: bool for computing and presenting silhouette.
    :param n_clusters_list: The number of clusters to check. If you change the default, pick values which won't mislead
    you (prefer constant interval like 100, 200, 300).
    :return: kmeans_models, kmeans_labels, n_clusters_list, wcss, silhouette_per_n_clusters_per_point,
            silhouette_per_n_clusters
    """
    if not fname:
        file_suffix = configs['fname_for_import'] + configs['PICKLE_SUFFIX']
        fname = 'kmeans_models' + '_' + file_suffix
    fname_n_clusters_list = fname + '_n_clusters_list'
    fname_wcss = fname + '_wcss'
    fname_silhouette_per_n_clusters = fname + '_fname_silhouette_per_n_clusters'

    if not load_existing:
        wcss = []
        kmeans_models = []
        kmeans_labels = []
        silhouette_per_n_clusters = []
        silhouette_per_n_clusters_per_point = []
        for n_clusters in n_clusters_list:
            print(f"Finding {n_clusters} clusters . . .")
            kmeans = KMeans(n_clusters=n_clusters,
                            random_state=configs['find_clusters_kmeans']['rand_state']).fit(queries_bow[:n_samples],
                                                                                            sample_weight=weights_number_of_searches[
                                                                                                          :n_samples])
            kmeans_models.append(kmeans)
            wcss.append(kmeans.inertia_)
            if compute_silhouette:
                print('Computing silhouette . . .')
                if n_clusters < 2:
                    return "Need at least 2 clusters to compute silhouette - quitting the function..."
                labels = kmeans.predict(queries_bow)
                kmeans_labels.append(labels)
                sample_silhouette_values = silhouette_samples(queries_bow, labels)
                silhouette_per_n_clusters_per_point.append(sample_silhouette_values)
                silhouette_per_n_clusters.append(np.mean(sample_silhouette_values))
                print('Finished computing silhouette.')

        fig, ax1 = plt.subplots()
        if show_plot:
            fig.suptitle(f"""Kmeans clusters' wcss (small=good) & Average Silhouette (big=good)
                             for {n_samples} samples VS. number of clusters""")
            color = 'tab:red'
            ax1.plot(n_clusters_list, wcss, color=color)
            ax1.set_xlabel("No. Kmeans clusters")
            ax1.set_ylabel("wcss", color=color)
            ax1.set_xticks(n_clusters_list)
            ax1.tick_params(axis='y', labelcolor=color)

        if compute_silhouette:
            if show_plot:
                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.plot(n_clusters_list, silhouette_per_n_clusters, color=color)
                ax2.set_ylabel("Average Silhouette", color=color)
                ax2.set_xticks(n_clusters_list)
                ax2.tick_params(axis='y', labelcolor=color)
            return kmeans_models, kmeans_labels, n_clusters_list, wcss, silhouette_per_n_clusters_per_point, \
                   silhouette_per_n_clusters
        if save_models:
            pickle_something(kmeans_models, fname)
            pickle_something(n_clusters_list, fname_n_clusters_list)
            pickle_something(wcss, fname_wcss)
            pickle_something(silhouette_per_n_clusters, fname_silhouette_per_n_clusters)
    else:
        print(f'Load existing models {fname} . . .')
        kmeans_models = unpickle_something(fname)
        n_clusters_list = unpickle_something(fname_n_clusters_list)
        wcss = unpickle_something(fname_wcss)
        if compute_silhouette:
            silhouette_per_n_clusters = unpickle_something(fname_silhouette_per_n_clusters)
            return kmeans_models, n_clusters_list, wcss, silhouette_per_n_clusters

    return kmeans_models, n_clusters_list, wcss


def get_clusters(df, cluster_labels, kmeans_model, vectorizer_model, idx_subcategory=None, vectorizer=None,
                 silhouette_rank=True, cosine_centroid_rank=True,
                 queries_vectors=None, load_existing=False, save_df=True, fname=None):
    """
    Receives a list of search queries and a list of their cluster labels numbers, returns a data frame with same search
    queries and the cluster information for each search query - label, most common query search.
    :param cosine_centroid_rank: If true, compute the cosine centroid similarity to most_common_word
    :param silhouette_rank: If true, computes the silhouette coefficient for every sample (high is good clustered)
    :param queries_vectors: The search queries vectorized (bow, tfidf or embeddings)
    :param idx_subcategory: the indices of the subcateogory (or other feature)
    :param vectorizer: The type of vectorizer used for the search queries (bow, tfidf or doc2vec)
    :param df: the queries df
    :param vectorizer_model: the doc2vec model used for vectorizing the searchqueries
    :param kmeans_model: The kmeans model used for classifying the lables
    :param cluster_labels: list of corresponding cluster labels in number
    :param fname: the name of the saved model file
    :param save_df: if true, saves the df
    :param load_existing: if True, loads existing model and doesn't fit
    these columns: search_query, cluster_label (number), most_common_query_in_cluster, cluster_closest_to_centroid_query
    if vectorizer=='doc2vec' and (NOT) cluster_size (number of search queries in the same cluster)
    """

    if not fname:
        file_suffix = configs['fname_for_import'] + configs['PICKLE_SUFFIX']
        fname = 'df_clusters' + '_' + file_suffix

    if not load_existing:
        search_query = configs['df_features']['query']
        number_of_searches = configs['df_features']['no_searches']
        labels = configs['df_features']['cluster']
        cluster_common_query = configs['df_features']['cluster_common_query']
        cluster_closest_to_centroid_query = configs['df_features']['cluster_closest_to_centroid_query']
        cluster_closest_to_centroid_rank = configs['df_features']['cluster_closest_centroid_rank']
        cluster_rank_silhouette = configs['df_features']['cluster_rank_silhouette']

        if len(idx_subcategory) > 0:
            queries = df[[search_query, number_of_searches]].iloc[idx_subcategory]
        else:
            queries = df[[search_query, number_of_searches]]

        queries[labels] = cluster_labels
        unique_clusters_labels = queries[labels].unique()
        labels_common_queries_dict = defaultdict(str)
        for label in unique_clusters_labels:
            labels_common_queries_dict[label] = \
                queries[queries[labels] == label].sort_values(by=number_of_searches, ascending=False).search_query.iloc[
                    0]
        print(f"   Creating  {cluster_common_query}. . .")
        queries[cluster_common_query] = queries[labels].transform(lambda x: labels_common_queries_dict[x])

        if cosine_centroid_rank:
            print("   Computing centroid cosine similarity rank . . .")
            if vectorizer == 'doc2vec':

                print("      Computing label_centroid_dict . . .")
                label_centroid_dict = {label:
                    most_cosine_similar_query_doc2vec(kmeans_model.cluster_centers_[label], vectorizer_model, df)
                                       for label in range(len(kmeans_model.cluster_centers_))}

                label_rank_dict = {label: most_cosine_similar_query_doc2vec_rank(kmeans_model.cluster_centers_[label],
                                                     vectorizer_model) for label in range(
                                                     len(kmeans_model.cluster_centers_))}

                print("      Assigning centroids using the label_centroid_dict . . .")
                queries[cluster_closest_to_centroid_query] = queries[labels].transform(lambda x: label_centroid_dict[x])
                queries[cluster_closest_to_centroid_rank] = queries[labels].transform(lambda x: label_rank_dict[x])

                '''
                queries[cluster_closest_to_centroid_query] = \
                    queries[labels].transform(
                        lambda x: most_cosine_similar_query_doc2vec(kmeans_model.cluster_centers_[x], vectorizer_model,
                                                                    df))
                queries[cluster_closest_to_centroid_rank] = \
                    queries[labels].transform(
                        lambda x: most_cosine_similar_query_doc2vec_rank(kmeans_model.cluster_centers_[x],
                                                                         vectorizer_model))
                '''
            else:  # bow or tfidf
                print("      Computing label_centroid_dict . . .")
                label_centroid_dict = {label: (
                    most_cosine_similar_vector(df, kmeans_model.cluster_centers_[label], queries_vectors.toarray())) for
                                              label in range(len(kmeans_model.cluster_centers_))}
                print("      Assigning centroids using the label_centroid_dict . . .")
                queries[cluster_closest_to_centroid_query] = queries[labels].transform(lambda x: label_centroid_dict[x])
                # split the tuple result (query, rank) into two columns:
                queries[cluster_closest_to_centroid_rank] = queries[cluster_closest_to_centroid_query].transform(
                    lambda x: x[1])
                queries[cluster_closest_to_centroid_query] = queries[cluster_closest_to_centroid_query].transform(
                    lambda x: x[0])

        if silhouette_rank:
            print("   Computing silhouette similarity rank . . .")
            n_clusters = len(queries[labels].unique())
            if n_clusters < 2:
                print(f"   There are only {n_clusters} cluster in this subcateogry, "
                      f"can't compute silhouette (rank as {configs['silhouette']['unranked_rank']}).")
                queries[cluster_rank_silhouette] = configs['silhouette']['unranked_rank']
            else:
                queries[cluster_rank_silhouette] = \
                    (silhouette_samples(queries_vectors[idx_subcategory], queries[labels]) + 1) / 2

        df_clusters = pd.concat([queries, df[df.columns.difference(queries.columns)].iloc[queries.index]], axis=1)

        if save_df:
            df_save(df_clusters, fname)
    else:
        df_clusters = df_load(fname)

    return df_clusters


def get_subcategories_indices(df, n_categories=None, sub_category_feature=configs['df_features']['sub_category']):
    """
    Receives the queries df, returns a list of indices for the queries belonging to each category (or sub category)
    :param n_categories: The number of categories (default is the maximum in bq-results-20200712-141041-tqwab845scx)
    :param df: the queries df :param n_categories: The number of sub categories to return, default is all of them
    :param sub_category_feature: the name of the sub_cateogry feature (you can also put the categories feature here)
    :return: a list in which each member is an index list of indices of the rows of the queries belonging to each
    category and a list of the sub category names, all orderd by sub category size
    """
    if not n_categories:
        n_categories = len(df[sub_category_feature].unique())
    subcategories_by_size = df[sub_category_feature].value_counts().index.tolist()[:n_categories]
    return [df[df[sub_category_feature] == subcategory].index for subcategory in
            subcategories_by_size], subcategories_by_size


def find_kmeans_clusters_per_subcategory(df, fit_vectorizer=False, save_vectorizer=False, save_df=False,
                                         n_clusters=configs['kmeans']['n_clusters'],
                                         vectorizer=configs['kmeans']['vectorizer'],
                                         n_categories=configs['kmeans']['n_categories'],
                                         idx_last_cluster=0,
                                         kmeans_weights=configs['df_features']['no_searches'],
                                         sub_category_feature=configs['df_features']['sub_category'],
                                         silhouette_rank=True, cosine_centroid_rank=True,
                                         searches_feature=configs['df_features']['no_searches'],
                                         analyzer='Default', min_df='Default',
                                         windows=configs['doc2vec']['window'],
                                         smallest_ngram='Default', vector_size=configs['doc2vec']['vector_size'],
                                         largest_ngram='Default', min_count=configs['doc2vec']['min_count'],
                                         epochs=configs['doc2vec']['epochs']):
    """
    Receives the queries df, returns a data frame containingg a kmeans cluster label for each query in the df,
    the most common search query in the cluster, and the cluster size
    :param save_df: If true, saves the origional queries df with the extra columns of the cluster data.
    :param analyzer: The embeddings building blocks. default is 'word' (gensim default) for doc2vec
    :param windows: A doc2vec parameter for the number of words left and right of each word to input the embedding model
    :param smallest_ngram: smallest ngram in the ngram range used for the analyzer
    :param largest_ngram: largest ngram in the ngram range used for the analyzer
    :param min_count: A doc2vec parameter.  Ignores all words with total frequency lower than this.
    :param epochs: A doc2vec parameter. Number of iterations (epochs) over the corpus.
    :param vector_size: A doc2vec parameter for the size of the vector of each query (size of the latent layer)
    :param min_df: A TF-IDF parameter. Ignore terms that have a document frequency strictly lower than this threshold.
    :param vectorizer: The type of queries vectorizer to use: bow, tfidf or doc2vec (as outputs of nlp_process)
    :param kmeans_weights: The samples weights used for the kmeans fitting. if number_of_searches, uses this feature.
    :param idx_last_cluster: The number of the last subcategory (for keeping labeling unique) searches_feature.
    :param searches_feature: The name of the searches feature (for the sample_weight parameter of kmeans)
    :param df: the queries df
    :param cosine_centroid_rank: If true, compute the cosine centroid similarity to most_common_word
    :param silhouette_rank: If true, computes the silhouette coefficient for every sample (high is good clustered)
    :param fit_vectorizer: if true, retrain the counter. if false, loads the counter from fname.
    :param save_vectorizer: if true, saves the pickles the vectorizer and the vectorized queries
    :param n_clusters: default number of clusters for the kmeans fitting is 200 (optimized using wcss and silhouette)
    :param n_categories: number of categories to take into account. Default is total number of subcategories
    :param sub_category_feature: The name of the sub category feature
    :return: clustered_queries_df, the df containing the cluster information for each search query, and i the number
             of total clusters the algorithm find for all subcategories combined
    """

    if n_categories == 'ALL':
        n_categories = len(df[sub_category_feature].unique())
    print(f'Getting {n_categories} subcategories indices . . .')
    idx_subcategories, subcategories_by_size = get_subcategories_indices(df, sub_category_feature=sub_category_feature,
                                                                         n_categories=n_categories)
    print(f'Creating a {vectorizer} representation of the queries . . .')
    if vectorizer == 'bow':
        queries_vector, _, vectorizer_model = nlp_process(df, bow=True, fit_vectorizer=fit_vectorizer,
                                                          save_vectorizer=save_vectorizer, analyzer=analyzer,
                                                          min_df=min_df, smallest_ngram=smallest_ngram,
                                                          largest_ngram=largest_ngram)
    elif vectorizer == 'tfidf':
        queries_vector, _, vectorizer_model = nlp_process(df, tfidf=True, fit_vectorizer=fit_vectorizer,
                                                          save_vectorizer=save_vectorizer)
    else:  # vectorizer == 'doc2vec':
        queries_vector, vectorizer_model = nlp_process(df, doc2vec=True, fit_vectorizer=fit_vectorizer,
                                                       save_vectorizer=save_vectorizer,
                                                       analyzer=analyzer,
                                                       windows=windows,
                                                       vector_size=vector_size,
                                                       min_count=min_count,
                                                       epochs=epochs,
                                                       smallest_ngram=smallest_ngram,
                                                       largest_ngram=largest_ngram)

    clustered_queries_df = pd.DataFrame()
    models_kmeans_subcategories = []
    queries_vectors_subcategories = []
    for i, (subcategory, idx_subcategory) in enumerate(zip(subcategories_by_size, idx_subcategories)):
        if n_clusters >= len(df[searches_feature][idx_subcategory]):
            n_clusters = len(idx_subcategory) // configs['kmeans']['divisor_n_clusters']
        if n_clusters == 0:
            n_clusters = 1
        if subcategory == 'ALL':
            break
        else:
            print(f"Finding {n_clusters} clusters for subcategory {i + 1}/{n_categories} '{subcategory}':")
            queries_sub_category = queries_vector[idx_subcategory]
            print('   Fitting clusters . . .')
            if kmeans_weights is configs['df_features']['no_searches']:
                kmeans_weights = df[searches_feature][idx_subcategory]
            kmeans_weights = df[searches_feature][idx_subcategory]
            kmeans = KMeans(n_clusters=n_clusters, random_state=configs['kmeans']['rand_state']).fit(
                queries_sub_category,
                sample_weight=kmeans_weights)
            models_kmeans_subcategories.append(kmeans)
            queries_vectors_subcategories.append(queries_sub_category)
            print('   Predicting clusters . . .')
            labels = kmeans.predict(queries_sub_category)
            print('   Getting cluster data . . .')
            subcategory_clustered_queries_df = get_clusters(df, labels, kmeans, vectorizer_model,
                                                            idx_subcategory=idx_subcategory,
                                                            vectorizer=vectorizer, queries_vectors=queries_vector,
                                                            silhouette_rank=silhouette_rank,
                                                            cosine_centroid_rank=cosine_centroid_rank,
                                                            save_df=save_df)
            labels = labels + idx_last_cluster + 1
            subcategory_clustered_queries_df[configs['df_features']['cluster']] = \
                subcategory_clustered_queries_df[configs['df_features']['cluster']] + idx_last_cluster + 1
            clustered_queries_df = pd.concat([clustered_queries_df, subcategory_clustered_queries_df])
            print(np.unique(labels))
            idx_last_cluster = np.max(labels)

    return clustered_queries_df, idx_last_cluster, pd.DataFrame(
        zip(subcategories_by_size, models_kmeans_subcategories),
        columns=[configs['df_features']['subcategory_name'], configs['df_features']['kmeans_models']]), vectorizer_model


def find_kmeans_clusters_per_subcategory_days_dict(df_day, fit_vectorizer=False, save_vectorizer=False,
                                                   n_days=configs['kmeans']['n_days'],
                                                   n_categories=configs['kmeans']['n_categories'],
                                                   n_clusters=configs['kmeans']['n_clusters'],
                                                   vectorizer=configs['kmeans']['vectorizer'],
                                                   silhouette_rank=True, cosine_centroid_rank=True,
                                                   kmeans_weights=configs['df_features']['no_searches'],
                                                   analyzer='Default', min_df='Default',
                                                   windows=configs['doc2vec']['window'],
                                                   smallest_ngram='Default',
                                                   vector_size=configs['doc2vec']['vector_size'],
                                                   largest_ngram='Default', min_count=configs['doc2vec']['min_count'],
                                                   epochs=configs['doc2vec']['epochs']):
    """
    Receives a dict of data frame with queries data from each day, returns n_categories kmeans clusters
     of all subcategories in each day
    :param kmeans_weights: The samples weights used for the kmeans fitting. if number_of_searches, uses this feature.
    :param vectorizer: The type of queries vectorizer to use: bow, tfidf or doc2vec (as outputs of nlp_process)
    :param save_vectorizer: if true, saves the model
    :param vector_size: A doc2vec parameter for the size of the vector of each query (size of the latent layer)
    :param min_df: A TF-IDF parameter. Ignore terms that have a document frequency strictly lower than this threshold.
    :param analyzer: The embeddings building blocks. default is 'word' (gensim default) for doc2vec
    :param windows: A doc2vec parameter for the number of words left and right of each word to input the embedding model
    :param smallest_ngram: smallest ngram in the ngram range used for the analyzer
    :param largest_ngram: largest ngram in the ngram range used for the analyzer
    :param min_count: A doc2vec parameter.  Ignores all words with total frequency lower than this.
    :param epochs: A doc2vec parameter. Number of iterations (epochs) over the corpus.
    :param cosine_centroid_rank: If true, compute the cosine centroid similarity to most_common_word
    :param silhouette_rank: If true, computes the silhouette coefficient for every sample (high is good clustered)
    :param fit_vectorizer: if False, loads existing model and doesn't fit
    :param n_clusters: default number of clusters for the kmeans fitting is 200 (optimized using wcss and silhouette)
    :param df_day: a dict of data frames, each one with data from different day
    :param n_days: number of days to run on (starts from the first)
    :param n_categories: the number of categories to search in each subcategory
    :return: clustered_queries_df_day, a dict of data frames containing the clusters data for each day:
             search_query, cluster_label, most_common_query_in_cluster, cluster_size
    """
    if n_days == 'ALL':
        n_days = len(df_day)
    clustered_queries_df_day = {}
    idx_last_cluster = 0
    models_kmeans_subcategories = []
    vectorizer_models = []
    for i in range(n_days):
        clustered_queries_df_day[i], idx_last_cluster, model_kmeans_subcategories, vectorizer_model = \
            find_kmeans_clusters_per_subcategory(df_day[i], n_categories=n_categories,
                                                 idx_last_cluster=idx_last_cluster, n_clusters=n_clusters,
                                                 vectorizer=vectorizer, kmeans_weights=kmeans_weights,
                                                 fit_vectorizer=fit_vectorizer, save_vectorizer=save_vectorizer,
                                                 silhouette_rank=silhouette_rank,
                                                 cosine_centroid_rank=cosine_centroid_rank,
                                                 analyzer=analyzer, min_df=min_df,
                                                 windows=windows,
                                                 smallest_ngram=smallest_ngram,
                                                 vector_size=vector_size,
                                                 largest_ngram=largest_ngram, min_count=min_count,
                                                 epochs=epochs)
        models_kmeans_subcategories.append(model_kmeans_subcategories)
        vectorizer_models.append(vectorizer_model)
    print(f"Finished finding cluster for {n_days} days for {n_categories} subcategories")
    return clustered_queries_df_day, models_kmeans_subcategories, vectorizer_models


def find_kmeans_clusters_per_subcategory_per_day(df, n_days=configs['kmeans']['n_days'],
                                                 fit_vectorizer=False, save_vectorizer=False, save_df_and_models=False,
                                                 load_existing=False, fname=None,
                                                 silhouette_rank=True, cosine_centroid_rank=True,
                                                 n_categories=configs['kmeans']['n_categories'],
                                                 n_clusters=configs['kmeans']['n_clusters'],
                                                 date_feature=configs['df_features']['date'],
                                                 vectorizer=configs['kmeans']['vectorizer'],
                                                 kmeans_weights=configs['df_features']['no_searches'],
                                                 analyzer='Default', min_df='Default',
                                                 windows=configs['doc2vec']['window'],
                                                 smallest_ngram='Default',
                                                 vector_size=configs['doc2vec']['vector_size'],
                                                 largest_ngram='Default', min_count=configs['doc2vec']['min_count'],
                                                 epochs=configs['doc2vec']['epochs']):
    """
    Receives a queries df, returns the df with the kmeans cluster unique labels and most common search queries for all
    days (using all function above)
    :param epochs: A doc2vec parameter. Number of iterations (epochs) over the corpus.
    :param min_df: A TF-IDF parameter. Ignore terms that have a document frequency strictly lower than this threshold.
    :param min_count: A doc2vec parameter.  Ignores all words with total frequency lower than this.
    :param vector_size: A doc2vec parameter for the size of the vector of each query (size of the latent layer)
    :param smallest_ngram: smallest ngram in the ngram range used for the analyzer
    :param largest_ngram: largest ngram in the ngram range used for the analyzer
    :param windows: A doc2vec parameter for the number of words left and right of each word to input the embedding model
    :param analyzer: The embeddings building blocks. default is 'word' (gensim default) for doc2vec
    :param save_df_and_models: if true, saves the clustered df and the kmeans models of each subcategory
    :param vectorizer: The type of queries vectorizer to use: bow, tfidf or doc2vec (as outputs of nlp_process)
    :param kmeans_weights: The samples weights used for the kmeans fitting. if number_of_searches, uses this feature.
    :param df: the queries df
    :param cosine_centroid_rank: If true, compute the cosine centroid similarity to most_common_word
    :param silhouette_rank: If true, computes the silhouette coefficient for every sample (high is good clustered)
    :param n_days: the number of days to take from the df (start by begining). Default is all days.
    :param save_vectorizer: if true, saves the aggregated df to csv
    :param fname: the name of the saved file
    :param load_existing: if True, loads existing model and doesn't fit
    :param fit_vectorizer: if False, loads existing model and doesn't fit
    :param n_clusters: the number of clusters to find in each sub category. default is 200.
    :param n_categories: the number of subcategories to find clusters in. default is all.
    :param date_feature: the name of the date feature.
    :return: the df with the kmeans cluster labels and most common search queries, ordered by days
    """
    if not fname:
        file_suffix = configs['fname_for_import'] + configs['PICKLE_SUFFIX']
        fname = 'df_clustered' + '_' + file_suffix
    fname_models_kmeans_subcategories = fname + '_' + 'models_kmeans_subcategories'
    fname_vectorizer_model = fname + '_' + 'models_kmeans_subcategories'
    if not load_existing:
        if n_days == 'ALL':
            n_days = len(df[date_feature].unique())
        print(f"Dividing df to {n_days} days. . .")
        df_day = get_separate_days_df_dict(df, date_feature=date_feature)
        print(f"Finding clusters for all {n_days} days . . .")
        clustered_queries_df_day, models_kmeans_subcategories, vectorizer_model = \
            find_kmeans_clusters_per_subcategory_days_dict(
                df_day, n_days=n_days, n_categories=n_categories, n_clusters=n_clusters, vectorizer=vectorizer,
                kmeans_weights=kmeans_weights, fit_vectorizer=fit_vectorizer, save_vectorizer=save_vectorizer,
                silhouette_rank=silhouette_rank, cosine_centroid_rank=cosine_centroid_rank, analyzer=analyzer,
                min_df=min_df, windows=windows, smallest_ngram=smallest_ngram, vector_size=vector_size,
                largest_ngram=largest_ngram, min_count=min_count, epochs=epochs)
        print(f"Adding cluster data to df . . .")
        df_clustered = pd.concat([clustered_queries_df_day[i] for i in range(n_days)])
        if save_df_and_models:
            pickle_something(models_kmeans_subcategories, fname_models_kmeans_subcategories)
            df_save(df_clustered, fname)
    else:
        print(f"Loading existing df {fname}, existing models {fname_models_kmeans_subcategories},\n"
              f"and vectorizer {fname_vectorizer_model}. . .")
        df_clustered = unpickle_something(fname)
        models_kmeans_subcategories = unpickle_something(fname_models_kmeans_subcategories)
        vectorizer_model = unpickle_something(fname_vectorizer_model)
    print(f"Done!")
    return df_clustered, models_kmeans_subcategories, vectorizer_model


def most_cosine_similar_query_doc2vec(sentence_vector, model_doc2vec, df,
                                      query_feature=configs['df_features']['query']):
    """
    Get the most cosine similar search query to the sentence_vector in the model_doc2vec vocabulary
    :param sentence_vector: a vector in the same size of the vectors of model_doc2vec, possibly a kmeans centroid
    :param model_doc2vec: the doc2vec model
    :param df: the queries df
    :param query_feature: the name of the search query feature
    :return: The text of the most similar search query
    """
    return df[query_feature][model_doc2vec.docvecs.most_similar([sentence_vector])[0][0]]


def most_cosine_similar_query_doc2vec_rank(sentence_vector, model_doc2vec):
    """
    Get the most cosine similar search query rank to the sentence_vector in the model_doc2vec vocabulary
    :param sentence_vector: a vector in the same size of the vectors of model_doc2vec, possibly a kmeans centroid
    :param model_doc2vec: the doc2vec model
    :return: The rank of the most similar search query
    """
    return model_doc2vec.docvecs.most_similar([sentence_vector])[0][1]


def cosine_similarity_two_vectors(vector1, vector2):
    """
    Compute cosine similarity of two vectors in the same size
    :param vector1: array of numbers of size n
    :param vector2: array of numbers of size n
    :return: The cosine similarity between the vectors, ranged 0-1
    """
    return 1 - cosine(vector1, vector2)


def most_cosine_similar_vector(df, centroid, vectors, query_feature=configs['df_features']['query']):
    """
    Receives a centroid vector and a matrix of all search qureis vectors, returns the search query most similair to
    the centroid and it's cosine similarity rank
    :param df: the queries df
    :param centroid: the vector to find most similair vector to
    :param vectors: the queries df vectorized (can be either bow, tfidf or any kind of embedding)
    :param query_feature: the name of the query feature
    :return: the most similair search query name and his cosine similarity rank
    """
    #idx_most_similar = np.argmax([cosine_similarity_two_vectors(centroid, vector) for vector in vectors])
    #maximal_similarity = np.max([cosine_similarity_two_vectors(centroid, vector) for vector in vectors])
    similarities_matrix = vectors.dot(centroid)
    idx_most_similar = np.argmax(similarities_matrix)
    maximal_similarity = np.max(similarities_matrix)
    return df[query_feature][idx_most_similar], maximal_similarity


def cluster_describe(df_one_label, columns_to_remove=configs['cluster_describe']['columns_to_remove']):
    """
    Receives a clustered df filtered on one cluster,
    return relevant statistics of relevant columns about the specific cluster
    :param columns_to_remove: The columns that are not relevant - to remove
    :param df_one_label: df_clustered[df_clustered.cluster_label==x]
    :return: mean	std	min	25%	50%	75%	max over numeric relavent columns, round for 2 digits
    """
    return df_one_label[df_one_label.columns.difference(columns_to_remove)].describe().iloc[1:].T.round(2)


def df_save(df, fname):
    """
    saves df to fname in csv format and report to user
    :param df: the df to save
    :param fname: the df file name
    :return: just prints the message to the user
    """
    df.to_csv(fname)
    print(f"Saved '{fname}'.")


def df_load(fname):
    """
    Loads csv df and reports to user.
    :param fname: the df file name to load
    :return: the loaded csv in df format
    """
    print(f"Load existing df '{fname}' . . .")
    return pd.read_csv(fname)


def pickle_something(something, pickle_filename):
    """
    saves something as pickle file named pickle_filename
    :param something: variable to save
    :param pickle_filename: file path to save to
    :return: nothing, just saves
    """
    if os.path.isfile(pickle_filename):
        os.remove(pickle_filename)
    with open(pickle_filename, 'wb') as file:
        pickle.dump(something, file)
    print(f"Saved '{pickle_filename}'")


def unpickle_something(pickle_filename):
    """
    Returns the Loaded saved pickle
    :param pickle_filename: the pickle file to load
    :return: the loaded filename as a python object
    """
    if not os.path.isfile(pickle_filename):
        return f"{pickle_filename} doesn't exist!"
    with open(pickle_filename, 'rb') as file:
        something_unpickled = pickle.load(file)
    print(f"Loaded '{pickle_filename}'")
    return something_unpickled


def import_process_cluster_output(fname_for_import=configs['fname_for_import'],
                                  fname_for_save=configs['fname_for_save'],
                                  dates_to_filter=configs['dates_to_filter'],
                                  n_samples=configs['n_samples'], vectorizer=configs['vectorizer'],
                                  fit_vectorizer=configs['fit_vectorizer'],
                                  save_df=configs['save_df'],
                                  save_clusters_aggregated=configs['save_clusters_aggregated'],
                                  save_vectorizers=configs['save_vectorizers'],
                                  save_models=configs['save_models'],
                                  load_existing_kmeans=configs['load_existing_kmeans'],
                                  silhouette_rank=configs['compute_silhouette_rank'],
                                  cosine_centroid_rank=configs['compute_cosine_centroid_rank'],
                                  n_categories=configs['n_categories'],
                                  n_clusters=configs['n_clusters'],
                                  date_feature=configs['date_feature'],
                                  kmeans_weights=configs['kmeans_weights_df_column'],
                                  analyzer='Default', min_df=configs['tfidf_min_df'],
                                  windows=configs['doc2vec_windows'],
                                  smallest_ngram='Default', vector_size=configs['doc2vec_vector_size'],
                                  largest_ngram='Default', min_count=configs['doc2vec_min_count'],
                                  epochs=configs['doc2vec_epochs'],
                                  more_than_n_days=configs['clusters_over_n_days'],
                                  plot_numeric_features_clusters_evaluated=configs['plot_clusters_evaluated'],
                                  n_top_clusters=configs['plot_n_top_clusters']):
    """
    Import a df, pre-process it, nlp process it, and return a clustered outputs.
    :param n_top_clusters: The number of top clusters (by score) to plot.
    :param plot_numeric_features_clusters_evaluated: if True, plot all evaulated features.
    :param more_than_n_days: Evaluate clusters with only than more this value of days data. default is 0.
    :param min_df: A TF-IDF parameter. Ignore terms that have a document frequency strictly lower than this threshold.
    :param load_existing_kmeans: If False, fit K-means. If true, looks for existing model.
    :param save_models: If True, saves pickle file of the kmeans model.
    :param save_vectorizers: If True, saves a pickle file of the vectorizers of the search quries.
    :param save_clusters_aggregated: if True, saves a pickle file of the aggregated clusters data.
    :param save_df: If true, saves the origional queries df with the extra columns of the cluster data.
    :param n_samples: the number of samples to consider. default is all.
    :param dates_to_filter: the number of dates to consider. default is all.
    :param fname_for_save: the base name of all the files saved. default is the import name.
    :param fname_for_import: the file name of the search queries df download from the BigQuery.
    :param epochs: A doc2vec parameter. Number of iterations (epochs) over the corpus.
    :param min_count: A doc2vec parameter.  Ignores all words with total frequency lower than this.
    :param vector_size: A doc2vec parameter for the size of the vector of each query (size of the latent layer)
    :param smallest_ngram: smallest ngram in the ngram range used for the analyzer
    :param largest_ngram: largest ngram in the ngram range used for the analyzer
    :param windows: A doc2vec parameter for the number of words left and right of each word to input the embedding model
    :param analyzer: The embeddings building blocks. default is 'word' (gensim default) for doc2vec
    :param vectorizer: The type of queries vectorizer to use: bow, tfidf or doc2vec (as outputs of nlp_process)
    :param kmeans_weights: The samples weights used for the kmeans fitting. if number_of_searches, uses this feature.
    :param cosine_centroid_rank: If true, compute the cosine centroid similarity to most_common_word
    :param silhouette_rank: If true, computes the silhouette coefficient for every sample (high is good clustered)
    :param fit_vectorizer: if False, loads existing model and doesn't fit
    :param n_clusters: the number of clusters to find in each sub category. default is 200.
    :param n_categories: the number of subcategories to find clusters in. default is all.
    :param date_feature: the name of the date feature.
    :return: 1. The original data frame with the cluster labeling, with one row for each unique search query.
             2. The aggregated clusters data, with one row for each cluster in each day.
             3. The vectorizors used for representing the search queries
             4. The kmeans models used for clustering the search queries<br>
             ----- For data of more than one day -----<br>
             5. The evaluated features over each cluster and over all days, with two scores of the clusters.
             6. The plots of the some top clusters data on each evaluated feature.
    """
    # pre-set parameters according to the vectorizer type
    if analyzer == 'Default':
        if vectorizer == 'doc2vec':
            analyzer = configs['doc2vec_analyzer']
        elif vectorizer == 'tfidf':
            analyzer = configs['tfidf_analyzer']
        else:  # bow
            analyzer = configs['bow']['analyzer']
    if smallest_ngram == 'Default':
        if vectorizer == 'doc2vec':
            smallest_ngram = configs['doc2vec_smallest_ngram']
        elif vectorizer == 'tfidf':
            smallest_ngram = configs['tfidf_smallest_ngram']
        else:  # bow
            smallest_ngram = configs['bow']['smallest_ngram']
    if largest_ngram == 'Default':
        if vectorizer == 'doc2vec':
            largest_ngram = configs['doc2vec_largest_ngram']
        elif vectorizer == 'tfidf':
            largest_ngram = configs['tfidf_largest_ngram']
        else:  # bow
            largest_ngram = configs['bow']['largest_ngram']

    start_time = datetime.datetime.now()
    # Fix display
    pd.set_option('display.max_rows', configs['display.max_rows'])
    pd.set_option('display.max_columns', configs['display.max_columns'])
    pd.set_option('display.width', configs['display.width'])
    # Import queries df
    df = import_df(fname=fname_for_import)
    # Prepare variables
    if n_samples == 'ALL':
        n_samples = len(df)
    if dates_to_filter == 'ALL':
        dates_to_filter = np.unique(df[date_feature])
    if fname_for_save == 'same as fname_for_import':
        fname_for_save = fname_for_import
    # report parameters to user
    print(f"""Running cluster process with these parameters:
    fname_for_import={fname_for_import}
    fname_for_save={fname_for_save}
    dates_to_filter={dates_to_filter}
    n_samples={n_samples}
    vectorizer={vectorizer}
    analyzer={analyzer}
    ngram_range=({smallest_ngram}, {largest_ngram}))
    min_df={min_df}
    fit_vectorizer={fit_vectorizer}
    save_df={save_df}
    save_clusters_aggregated={save_clusters_aggregated}
    save_vectorizers={save_vectorizers}
    save_models={save_models}
    load_existing_kmeans={load_existing_kmeans}
    silhouette_rank={silhouette_rank}
    cosine_centroid_rank={cosine_centroid_rank}
    n_categories={n_categories}
    n_clusters={n_clusters}
    date_feature={date_feature}
    kmeans_weights={kmeans_weights}
The process started at {start_time}""")
    '''
    if vectorizer == 'tfidf':
        print("Vectorizor is tfidf, won't compute centroid score since the runetime is too long.")
        cosine_centroid_rank = False
        print(f"cosine_centroid_rank={cosine_centroid_rank}")
    '''
    df = df.iloc[:n_samples]
    idx_max_label = 0
    cluster_label = configs['df_features']['cluster']
    df_clustered_unified = pd.DataFrame()
    kmeans_models = []
    vectorizers = []
    # Process, cluster and unify
    for date_to_filter in dates_to_filter:
        print(f'Finding clusters of search queries for {date_to_filter} . . .')
        df_day_filtered = df[df.search_date == date_to_filter]
        df_day_filtered = pre_process_df(df_day_filtered, date=date_to_filter, remove_duplicate=True, remove_na=True,
                                         drop_numeric_queries=True)
        print(df_day_filtered)

        df_clustered, kmeans_models_one_day, vectorizer_one_day = \
            find_kmeans_clusters_per_subcategory_per_day(
                df_day_filtered,
                vectorizer=vectorizer, fit_vectorizer=fit_vectorizer, save_vectorizer=False,  # vectorizer
                n_categories=n_categories, n_clusters=n_clusters, date_feature=date_feature,  # cluster
                load_existing=load_existing_kmeans, save_df_and_models=False,  # model
                fname=f'{fname_for_save}',  # fname to save model
                cosine_centroid_rank=cosine_centroid_rank, silhouette_rank=silhouette_rank,
                kmeans_weights=kmeans_weights, analyzer=analyzer, min_df=min_df,
                windows=windows, smallest_ngram=smallest_ngram, vector_size=vector_size,
                largest_ngram=largest_ngram, min_count=min_count, epochs=epochs)

        df_clustered[cluster_label] = df_clustered[cluster_label] + idx_max_label
        idx_max_label = int(df_clustered[cluster_label].max())
        df_clustered_unified = pd.concat([df_clustered_unified, df_clustered], axis=0)
        kmeans_models.append(kmeans_models_one_day)
        vectorizers.append(vectorizer_one_day)
    print('Aggregating clusters over relavent columns . . .')
    aggregated_clusters = aggregate_clusters(df_clustered_unified)
    print(f"Finished finding clusters for {dates_to_filter}")
    runtime_minutes = (datetime.datetime.now() - start_time).seconds / 60
    if len(dates_to_filter) > 1:
        print("Evaluating clusters . . .")
        clusters_evaluated = evaluate_aggregated_clusters_over_all_features(aggregated_clusters,
                                                                            more_than_n_days=more_than_n_days)
        if configs['df_features']['cluster_rank_silhouette'] in clusters_evaluated.keys():
            print('   Adding average silhouette score over all days for all features evaluated . . .')
            clusters_evaluated = add_silhouette_score_over_all_days(clusters_evaluated)

        if configs['df_features']['cluster_closest_centroid_rank'] in clusters_evaluated.keys():
            print('   Adding average centroid distance score over all days for all features evaluated . . .')
            clusters_evaluated = add_centroid_score_over_all_days(clusters_evaluated)

        if plot_numeric_features_clusters_evaluated:
            print(f"Plotting the cluster evaluations of the top {configs['plot_n_top_clusters']}"
                  f" over numeric features . . .")
            eval_clusters_all_days = get_and_plot_clusters_all_days(aggregated_clusters)
            if type(eval_clusters_all_days) is dict:
                pickle_something(eval_clusters_all_days, fname_for_save + '_clusters_evaluated_only_all_days'
                                 + configs['PICKLE_SUFFIX'])

        pickle_something(clusters_evaluated, fname_for_save + '_clusters_evaluated' + configs['PICKLE_SUFFIX'])
    # save
    if save_models:
        fname_for_save_kmeans_models = fname_for_save + '_kmeans_models' + configs['PICKLE_SUFFIX']
        pickle_something(kmeans_models, fname_for_save_kmeans_models)
    if save_vectorizers:
        fname_for_save_vectorizors = fname_for_save + '_vectorizors' + configs['PICKLE_SUFFIX']
        pickle_something(vectorizers, fname_for_save_vectorizors)
    if save_df:
        df_save(df_clustered_unified, fname_for_save + '_df')
        df_save(aggregated_clusters, fname_for_save + '_clusters')

    print(f'Done! The process ran for {runtime_minutes / 60} hours, or {runtime_minutes} minutes.')
    if len(dates_to_filter) > 1:
        # If true, clusters_evaluated exists and so return it
        if plot_numeric_features_clusters_evaluated:
            # If true, eval_clusters_all_days exists and return it
            return eval_clusters_all_days, clusters_evaluated, aggregated_clusters, df_clustered_unified, kmeans_models,\
                   vectorizers
        return clusters_evaluated, aggregated_clusters, df_clustered_unified, kmeans_models, vectorizers
    else:
        return aggregated_clusters, df_clustered_unified, kmeans_models, vectorizers


def add_silhouette_score_over_all_days(clusters_evaluated_dict, also_sort=True):
    """
    Receives the dict of clusters evaluated over each feature, computes the average silhouette over all days,
    adds this average for all features evaluated and sorts by this score.
    :param clusters_evaluated_dict: the output of evaluate_aggregated_clusters_over_all_features
    :param also_sort: if true, sorts by the average silhouette score of all days
    :return: clusters_evaluated_dict with silhouette_over_all_days column for all features
    """
    mean_silhouette_values = np.mean(clusters_evaluated_dict[configs['df_features']['cluster_rank_silhouette']],
                                     axis=1).values  # no real problem here
    for feature in clusters_evaluated_dict.keys():
        clusters_evaluated_dict[feature][configs['df_features']['silhouette_over_all_days']] = mean_silhouette_values
        if also_sort:
            clusters_evaluated_dict[feature].sort_values(by=configs['df_features']['silhouette_over_all_days'],
                                                         ascending=False, inplace=True)
    return clusters_evaluated_dict


def add_centroid_score_over_all_days(clusters_evaluated_dict, also_sort=True):
    """
    Receives the dict of clusters evaluated over each feature, computes the average centroid distance score
    over all days, adds this average for all features evaluated and sorts by this score.
    :param clusters_evaluated_dict: the output of evaluate_aggregated_clusters_over_all_features
    :param also_sort: if true, sorts by the average centroid score of all days
    :return: clusters_evaluated_dict with centroid_over_all_days column for all features
    """
    # If silhouette_over_all_days, compute mean centroid score while dropping the silhouette column
    if configs['df_features']['silhouette_over_all_days'] in \
            clusters_evaluated_dict[list(clusters_evaluated_dict.keys())[0]].columns:
        mean_centroid_values = np.mean(
            clusters_evaluated_dict[configs['df_features']['cluster_closest_centroid_rank']].drop(
                columns=configs['df_features']['silhouette_over_all_days']), axis=1).values  # no real problem here
    else:  # no need to drop the silhouette column
        mean_centroid_values = np.mean(
            clusters_evaluated_dict[configs['df_features']['cluster_closest_centroid_rank']], axis=1).values  # and here
    for feature in clusters_evaluated_dict.keys():
        clusters_evaluated_dict[feature][configs['df_features']['centroid_over_all_days']] = mean_centroid_values
        if also_sort:
            clusters_evaluated_dict[feature].sort_values(by=configs['df_features']['centroid_over_all_days'],
                                                         ascending=False, inplace=True)
    return clusters_evaluated_dict


def import_process_cluster_output_log(log=configs['log'], fname_for_import=configs['fname_for_import'],
                                      fname_for_save=configs['fname_for_import'],
                                      dates_to_filter=configs['dates_to_filter'],
                                      n_samples=configs['n_samples'], vectorizer=configs['vectorizer'],
                                      fit_vectorizer=configs['fit_vectorizer'],
                                      save_df=configs['save_df'],
                                      save_clusters_aggregated=configs['save_clusters_aggregated'],
                                      save_vectorizers=configs['save_vectorizers'],
                                      save_models=configs['save_models'],
                                      load_existing_kmeans=configs['load_existing_kmeans'],
                                      silhouette_rank=configs['compute_silhouette_rank'],
                                      cosine_centroid_rank=configs['compute_cosine_centroid_rank'],
                                      n_categories=configs['n_categories'],
                                      n_clusters=configs['n_clusters'],
                                      date_feature=configs['date_feature'],
                                      kmeans_weights=configs['kmeans_weights_df_column'],
                                      analyzer='Default', min_df=configs['tfidf_min_df'],
                                      windows=configs['doc2vec_windows'],
                                      smallest_ngram='Default', vector_size=configs['doc2vec_vector_size'],
                                      largest_ngram='Default', min_count=configs['doc2vec_min_count'],
                                      epochs=configs['doc2vec_epochs']):
    """
    Import a df, pre-process it, nlp process it, and return a clustered outputs. if log=True, the entire output
    is redirected to a log file instead of the screen.
    :param log: if True, the entire output is redirected to a log file instead of the screen.
    :param min_df: A TF-IDF parameter. Ignore terms that have a document frequency strictly lower than this threshold.
    :param load_existing_kmeans: If False, fit K-means. If true, looks for existing model.
    :param save_models: If True, saves pickle file of the kmeans model.
    :param save_vectorizers: If True, saves a pickle file of the vectorizers of the search queries.
    :param save_clusters_aggregated: if True, saves a pickle file of the aggregated clusters data.
    :param save_df: If true, saves the original queries df with the extra columns of the cluster data.
    :param n_samples: the number of samples to consider. default is all.
    :param dates_to_filter: the number of dates to consider. default is all.
    :param fname_for_save: the base name of all the files saved. default is the import name.
    :param fname_for_import: the file name of the search queries df download from the BigQuery.
    :param epochs: A doc2vec parameter. Number of iterations (epochs) over the corpus.
    :param min_count: A doc2vec parameter.  Ignores all words with total frequency lower than this.
    :param vector_size: A doc2vec parameter for the size of the vector of each query (size of the latent layer)
    :param smallest_ngram: smallest ngram in the ngram range used for the analyzer
    :param largest_ngram: largest ngram in the ngram range used for the analyzer
    :param windows: A doc2vec parameter for the number of words left and right of each word to input the embedding model
    :param analyzer: The embeddings building blocks. default is 'word' (gensim default) for doc2vec
    :param vectorizer: The type of queries vectorizer to use: bow, tfidf or doc2vec (as outputs of nlp_process)
    :param kmeans_weights: The samples weights used for the kmeans fitting. if number_of_searches, uses this feature.
    :param cosine_centroid_rank: If true, compute the cosine centroid similarity to most_common_word
    :param silhouette_rank: If true, computes the silhouette coefficient for every sample (high is good clustered)
    :param fit_vectorizer: if False, loads existing model and doesn't fit
    :param n_clusters: the number of clusters to find in each sub category. default is 200.
    :param n_categories: the number of subcategories to find clusters in. default is all.
    :param date_feature: the name of the date feature.
    :return: 1. The original data frame with the cluster labeling, with one row for each unique search query.
             2. The aggregated clusters data, with one row for each cluster in each day.
             3. The vectorizors used for representing the search queries
             4. The kmeans models used for clustering the search queries<br>
             ----- For data of more than one day -----<br>
             5. The evaluated features over each cluster and over all days, with two scores of the clusters.
             6. The plots of the some top clusters data on each evaluated feature.
    """
    if log:
        log_fname = f'{datetime.datetime.date(datetime.datetime.now())}_{fname_for_save}.log'
        with open(log_fname, 'w') as f:
            with redirect_stdout(f):
                return import_process_cluster_output(fname_for_import=fname_for_import,
                                                     fname_for_save=fname_for_save,
                                                     dates_to_filter=dates_to_filter,
                                                     n_samples=n_samples, vectorizer=vectorizer,
                                                     fit_vectorizer=fit_vectorizer,
                                                     save_df=save_df,
                                                     save_clusters_aggregated=save_clusters_aggregated,
                                                     save_vectorizers=save_vectorizers,
                                                     save_models=save_models,
                                                     load_existing_kmeans=load_existing_kmeans,
                                                     silhouette_rank=silhouette_rank,
                                                     cosine_centroid_rank=cosine_centroid_rank,
                                                     n_categories=n_categories,
                                                     n_clusters=n_clusters,
                                                     date_feature=date_feature,
                                                     kmeans_weights=kmeans_weights,
                                                     analyzer=analyzer, min_df=min_df,
                                                     windows=windows,
                                                     smallest_ngram=smallest_ngram, vector_size=vector_size,
                                                     largest_ngram=largest_ngram, min_count=min_count,
                                                     epochs=epochs)
    else:
        return import_process_cluster_output(fname_for_import=fname_for_import,
                                             fname_for_save=fname_for_save,
                                             dates_to_filter=dates_to_filter,
                                             n_samples=n_samples, vectorizer=vectorizer,
                                             fit_vectorizer=fit_vectorizer,
                                             save_df=save_df,
                                             save_clusters_aggregated=save_clusters_aggregated,
                                             save_vectorizers=save_vectorizers,
                                             save_models=save_models,
                                             load_existing_kmeans=load_existing_kmeans,
                                             silhouette_rank=silhouette_rank,
                                             cosine_centroid_rank=cosine_centroid_rank,
                                             n_categories=n_categories,
                                             n_clusters=n_clusters,
                                             date_feature=date_feature,
                                             kmeans_weights=kmeans_weights,
                                             analyzer=analyzer, min_df=min_df,
                                             windows=windows,
                                             smallest_ngram=smallest_ngram, vector_size=vector_size,
                                             largest_ngram=largest_ngram, min_count=min_count,
                                             epochs=epochs)


def aggregate_clusters(df_clustered_unified):
    """
    Receives the clustered df, returns the df's clustered aggregated over:
    cluster_name	rank_silhouette	cluster_query_closest_to_centroid	most_common_query_in_cluster
    most_common_sub_category_name	rank_cosine_centroid	search_date	queries	n_queries
    listings_with_orders	number_of_searches	orders	price	revenue	searches_with_clicks
    total_number_of_search_results
    :param df_clustered_unified: the output of find_kmeans_clusters_per_subcategory_per_day
    :return: a data frame of the aggregated columns of each cluster
    """
    number_of_searches = configs['df_features']['no_searches']
    searches_with_clicks = configs['df_features']['search_with_clicks']
    listings_with_orders = configs['df_features']['listings_with_orders']
    orders = configs['df_features']['orders']
    revenue = configs['df_features']['revenue']
    total_number_of_search_results = configs['df_features']['total_number_of_search_results']
    rank_silhouette = configs['df_features']['cluster_rank_silhouette']
    cluster_label = configs['df_features']['cluster']
    search_query = configs['df_features']['query']
    queries = configs['df_features']['queries']
    n_queries = configs['df_features']['n_queries']
    search_date = configs['df_features']['date']
    most_common_sub_category_name = configs['df_features']['sub_category']
    most_common_query_in_cluster = configs['df_features']['cluster_common_query']
    cluster_query_closest_to_centroid = configs['df_features']['cluster_closest_to_centroid_query']
    cluster_closest_centroid_rank = configs['df_features']['cluster_closest_centroid_rank']
    cluster_rank_silhouette = configs['df_features']['cluster_rank_silhouette']
    cluster_name = configs['df_features']['cluster_name']
    unnamed_value = 'Unnamed'
    print('Aggregate columns. . .')
    columns_to_sum = columns_in_df(df_clustered_unified,
                                   [number_of_searches, searches_with_clicks, listings_with_orders, orders, revenue])
    columns_to_mean = columns_in_df(df_clustered_unified, [total_number_of_search_results, rank_silhouette])
    columns_to_first = df_clustered_unified.columns.difference(columns_to_sum).difference(columns_to_mean).difference(
        [cluster_label, search_query])
    df_sum = df_clustered_unified.groupby(by=[cluster_label]).sum()[columns_to_sum]
    df_avg = df_clustered_unified.groupby(by=[cluster_label]).mean()[columns_to_mean]
    df_cat = df_clustered_unified.groupby(by=[cluster_label]).first()[columns_to_first]
    clusters_aggregated = pd.concat([df_sum, df_avg, df_cat], axis=1)
    print('Aggregate queries (takes a while). . .')
    clusters_aggregated[queries] = [
        df_clustered_unified[df_clustered_unified[cluster_label] == one_cluster][search_query].tolist()
        for one_cluster in df_clustered_unified[cluster_label].unique()]
    clusters_aggregated[n_queries] = clusters_aggregated[queries].transform(lambda x: len(x))
    print('Create cluster price column. . .')
    if len(columns_in_df(clusters_aggregated, [orders, revenue])) == 2:
        clusters_aggregated = feature_engineering(clusters_aggregated, price=True, query_lens=False,
                                                  clicks_per_search=False, save_df=False)
    print('Reorder columns. . .')
    first_columns = columns_in_df(clusters_aggregated, [search_date, n_queries, queries])
    second_columns = columns_in_df(clusters_aggregated,
                                   [most_common_sub_category_name, most_common_query_in_cluster,
                                    cluster_query_closest_to_centroid, cluster_closest_centroid_rank, rank_silhouette,
                                    cluster_rank_silhouette])
    third_columns = clusters_aggregated.columns.difference(first_columns).difference(second_columns)
    clusters_aggregated = pd.concat([clusters_aggregated[first_columns], clusters_aggregated[second_columns],
                                     clusters_aggregated[third_columns]], axis=1)

    if len(columns_in_df(clusters_aggregated, [most_common_query_in_cluster, cluster_query_closest_to_centroid])) == 2:
        clusters_aggregated[cluster_name] = np.where(
            clusters_aggregated[cluster_query_closest_to_centroid] == clusters_aggregated[most_common_query_in_cluster],
            clusters_aggregated[cluster_query_closest_to_centroid], unnamed_value)
    else:
        clusters_aggregated[cluster_name] = unnamed_value
    clusters_aggregated[cluster_name] = clusters_aggregated.apply(
        lambda x: x[queries][0] if (x[n_queries] == 1) & (x[cluster_name] == unnamed_value) else x[cluster_name],
        axis=1)
    if len(columns_in_df(clusters_aggregated, [most_common_query_in_cluster, cluster_query_closest_to_centroid])) == 2:
        clusters_aggregated[cluster_name] = clusters_aggregated.apply(
            lambda x: f"{x[cluster_query_closest_to_centroid]}_{x[most_common_query_in_cluster]}" if (
                    x[cluster_name] == unnamed_value) else x[cluster_name], axis=1)
    else:
        clusters_aggregated[cluster_name] = clusters_aggregated.apply(
            lambda x: f"{x[most_common_query_in_cluster]}" if (x[cluster_name] == unnamed_value) else x[cluster_name],
            axis=1)
    clusters_aggregated[cluster_name] = clusters_aggregated[most_common_sub_category_name] + '_' + clusters_aggregated[
        cluster_name]

    clusters_aggregated = pd.concat([clusters_aggregated[[cluster_name]], clusters_aggregated[
        clusters_aggregated.columns.difference([cluster_name], sort=False)]], axis=1)
    print('Done aggregating!')
    return clusters_aggregated


def columns_in_df(df, list_of_columns):
    """
    Receives a df and a list of potential columns, returns a list of columns that are indeed in the df
    :param df: a data frame object
    :param list_of_columns: list of columns to search in the df
    :return: a list of columns that are both in list_of_columns and df.columns
    """
    return df.columns[df.columns.isin(list_of_columns)].tolist()


def evaluate_aggregated_clusters(clusters_aggregated,
                                 feature_to_evaluate=configs['df_features']['cluster_closest_centroid_rank'],
                                 more_than_n_days=configs['clusters_over_n_days']):
    """
    Receives the aggregated clusters and returns a group by of the feature to evaluate over all clusters and
    all dates. if more_than_n>0, returns only clusters with data for more than number of days.
    :param clusters_aggregated: the data frame output of aggregate_clusters
    :param feature_to_evaluate: a feature to evaluate (don't use ['cluster_label','cluster_name', 'search_date'])
    :param more_than_n_days: the minimum number of days a cluster have data on
    :return: a data frame with indices of cluster names and columns of dates, the data is of the feature to evaluate.
    """
    cluster_name = configs['df_features']['cluster_name']
    search_date = configs['df_features']['date']
    df_filtered_to_feature_more_than_n = \
        clusters_aggregated[clusters_aggregated.cluster_name.isin(
            clusters_aggregated.cluster_name.value_counts()[clusters_aggregated.cluster_name.value_counts() >
                                                            more_than_n_days].index.tolist())]
    return df_filtered_to_feature_more_than_n.sort_values(by=[cluster_name]).groupby(
        by=[search_date, cluster_name]).first()[[feature_to_evaluate]].unstack().T


def evaluate_aggregated_clusters_over_all_features(clusters_aggregated,
                                                   more_than_n_days=configs['clusters_over_n_days']):
    """
    Receives the clusters aggregated, returns the evaluation of the clusters over all features and all dates
    :param clusters_aggregated: the data frame output of aggregate_clusters
    :param more_than_n_days: the minimum number of days a cluster have data on
    :return: a dict where the key is the feature evaluated and the value is the data frame of evaluation
    """
    features_to_disregard = [configs['df_features']['cluster'], configs['df_features']['cluster_name'],
                             configs['df_features']['date']]
    return {feature: evaluate_aggregated_clusters(clusters_aggregated, feature_to_evaluate=feature,
                                                  more_than_n_days=more_than_n_days) for feature in
            clusters_aggregated.columns.difference(features_to_disregard).tolist()}


def plot_cluster_evaluated_feature(clusters_evaluated_dict, n_dates_to_filter, feature='price', n_top_clusters=5,
                                   save_fig=True, fname_for_save=configs['fname_for_import']):
    """
    Receives the clusters evaluated dictionary and a feature to plot, plot the first n_top_clusters clusters
    (if sorted by score, they are the top clusters by score)
    :param n_dates_to_filter: The number of dates of the clusters.
    :param clusters_evaluated_dict: the output of evaluate_aggregated_clusters_over_all_features
    :param feature: a feature (key in clusters_evaluated_dict) to plot over time
    :param n_top_clusters: the number of first clusters to take
    :param figsize: the size of the plot
    :param save_fig: if true, saves the plot
    :param fname_for_save: the file name for plot's saving
    :return: plots the evaluated feature top clusters
    """
    plt.figure(figsize=(7, 4.8))
    data_to_plot = clusters_evaluated_dict[feature].loc[feature, :].iloc[:, :n_dates_to_filter][:n_top_clusters]
    plt.plot(data_to_plot.T)
    plt.legend(data_to_plot.index, loc='upper center', bbox_to_anchor=(0.5, -0.08),
               fancybox=True, shadow=True)
    plt.title(f'{feature} over time of the {n_top_clusters} clusters with highest score', fontsize=(12))
    plt.ylabel(feature)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{fname_for_save}_{feature}_{n_top_clusters}_clusters_with_highest_score.png')


def plot_cluster_evaluated_all_features(clusters_evaluated_dict, n_dates_to_filter, n_top_clusters=5, save_fig=True,
                                        fname_for_save=configs['fname_for_import']):
    """
    Receives the clusters evaluated dictionary, plot the first n_top_clusters clusters over all quantitive features
    :param clusters_evaluated_dict: the output of evaluate_aggregated_clusters_over_all_features
    :param n_top_clusters: the number of first clusters to take
    :param n_dates_to_filter: The number of dates of the clusters.
    :param save_fig: if true, saves the plots
    :param fname_for_save: the file name for plots' saving
    :return:
    """
    for feature in clusters_evaluated_dict.keys():
        if np.issubdtype(clusters_evaluated_dict[feature].iloc[:, 0].dtype, np.number):
            plot_cluster_evaluated_feature(clusters_evaluated_dict, n_dates_to_filter=n_dates_to_filter,
                                           feature=feature, n_top_clusters=n_top_clusters,
                                           save_fig=save_fig, fname_for_save=fname_for_save)


def plot_counts_and_highest_values(clusters_evaluated, feature=configs['df_features']['total_number_of_search_results'],
                                   n_top_clusters=configs['plot_n_top_clusters'],
                                   rotate_x_degrees=configs['plot_rotate_x_degrees'], day_highest_values=None,
                                   save_fig=True, fname_for_save=configs['fname_for_save']):
    """
    plot the n_top_clusters with most values in dates and with highest values in a specific day over a feature
    :param clusters_evaluated: the data frame output of evaluate_aggregated_clusters_over_all_features
    :param feature: the feature to plot on
    :param save_fig: if true, saves the plots
    :param fname_for_save: the file name for plots' saving
    :param n_top_clusters: number of top clusters to show
    :param rotate_x_degrees: degrees to rotate x ticks
    :param day_highest_values: the top clusters with highest values will be measured by the values of this day
    :return:
    """
    if not day_highest_values:
        day_highest_values = clusters_evaluated[feature].columns[-1]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), squeeze=False)
    fig.suptitle(f'{feature}', fontsize=18)
    data_with_most_counts = clusters_evaluated[feature].loc[clusters_evaluated[feature].count(axis=1).sort_values(
        ascending=False).index[:n_top_clusters]]
    counts_plot = data_with_most_counts.loc[feature, :].T.plot(ax=axes[0, 0])
    counts_plot.set_ylabel(feature)
    counts_plot.set_title(f"The {n_top_clusters} clusters with most data over time")
    counts_plot.tick_params(axis='x', labelrotation=rotate_x_degrees)
    data_with_highest_values = clusters_evaluated[feature].loc[clusters_evaluated[feature].sort_values(
        by=day_highest_values, ascending=False).index[:n_top_clusters]]
    data_with_highest_values = data_with_highest_values.loc[feature, :].T.plot(ax=axes[0, 1])
    data_with_highest_values.set_title(
        f"The {n_top_clusters} clusters with highest value in {day_highest_values}")
    data_with_highest_values.tick_params(axis='x', labelrotation=rotate_x_degrees)
    plt.tight_layout()
    plt.show()
    if save_fig:
        plt.savefig(f'{fname_for_save}_{feature}_top_{n_top_clusters}.png')


def plot_counts_and_highest_values_over_numeric_features(clusters_evaluated,
                                                         n_top_clusters=configs['plot_n_top_clusters'],
                                                         rotate_x_degrees=configs['plot_rotate_x_degrees'],
                                                         day_highest_values=None, save_fig=True,
                                                         fname_for_save=configs['fname_for_save']):
    """
    plot the n_top_clusters with most values in dates and with highest values in a specific day over all features
    :param save_fig: If true, saves all plots (one file for each plot).
    :param fname_for_save: The base file name for saving the plots.
    :param clusters_evaluated: the data frame output of evaluate_aggregated_clusters_over_all_features
    :param n_top_clusters: number of top clusters to show
    :param rotate_x_degrees: degrees to rotate x ticks
    :param day_highest_values: the top clusters with highest values will be measured by the values of this day
    :return:
    """
    for feature in clusters_evaluated.keys():
        if np.issubdtype(clusters_evaluated[feature].iloc[:, 0].dtype, np.number):  # is numeric
            plot_counts_and_highest_values(clusters_evaluated, feature=feature, n_top_clusters=n_top_clusters,
                                           rotate_x_degrees=rotate_x_degrees,
                                           day_highest_values=day_highest_values, save_fig=save_fig,
                                           fname_for_save=fname_for_save)


def get_all_days_clusters(aggregated_clusters):
    """
    Receive aggregated_clusters, returns the aggregated clusters of only the cluster that appear in all days exactly
    (deletes more than once a day clusters).
    :param aggregated_clusters: the data frame output of aggregate_clusters
    :return: aggregated_clusters over clusters with all days data only
    """
    cluster_name = configs['df_features']['cluster_name']
    search_date = configs['df_features']['date']
    aggregated_clusters['name_and_date']=(aggregated_clusters[cluster_name] + '_' + aggregated_clusters[search_date])
    temp = pd.concat([aggregated_clusters['name_and_date'],aggregated_clusters[aggregated_clusters.columns.difference(['name_and_date'])]],axis=1)
    temp_value_counts = temp[~temp.name_and_date.isin(temp.name_and_date.value_counts()[temp.name_and_date.value_counts()>1].index)][cluster_name].value_counts()
    agg_clus_7_days = temp[temp.cluster_name.isin(temp_value_counts[temp_value_counts==7].index)].sort_values(by=[cluster_name])
    return agg_clus_7_days


def get_and_plot_clusters_all_days(aggregated_clusters):
    """
    Receive aggregated_clusters, plots the evaluation of the aggregated clusters of only the cluster that appear in all
     days exactly (deletes more than once a day clusters).
    :param aggregated_clusters: the data frame output of aggregate_clusters
    :return: aggregated_clusters over clusters with all days data only
    """
    agg_clus_all_days = get_all_days_clusters(aggregated_clusters)
    if agg_clus_all_days.empty:  # Than plot highest values for all clusters
        print("No cluster was found by the algorithm in exactly all days of the data. plotting clusters by score . . .")
        clusters_evaluated = evaluate_aggregated_clusters_over_all_features(aggregated_clusters)
        clusters_evaluated = add_silhouette_score_over_all_days(clusters_evaluated)
        clusters_evaluated = add_centroid_score_over_all_days(clusters_evaluated)
        return plot_counts_and_highest_values_over_numeric_features(clusters_evaluated)

    eval_clus_all_days = evaluate_aggregated_clusters_over_all_features(agg_clus_all_days)
    eval_clus_all_days = add_silhouette_score_over_all_days(eval_clus_all_days)
    eval_clus_all_days = add_centroid_score_over_all_days(eval_clus_all_days)
    plot_cluster_evaluated_all_features(eval_clus_all_days,
                                        len(aggregated_clusters[configs['df_features']['date']].value_counts()))
    return eval_clus_all_days