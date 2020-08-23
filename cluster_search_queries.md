# Cluster Search Queries

Cluster Search Queries is a Python module for automated clustering search queries into meaningful groups,
 solving two main problem:
* Typos
* Different Phrasings for same buyer intent

You can Use the module's result to explore different groups of search queries and find new search query trends.

## The models

The process contain two main phases:
1. Representation of the search queries as vectors of numbers.
2. Clustering those vectors.

The current default models:
1. Gensim's **Doc2Vec** embeddings representation of the search queries with 1-gram word analyzer.
2. Scikit-Learn's 50 **K-means** clusters over each most_common_sub_category in each day, 
   with weights of the number of searches of each search query.

it's possible to change these default settings to TF-IDF model which gives not too bad results,
but the TF-IDF scoring method isn't good enough.

The maximum amount of clusters the module will find is 8500 clusters for a day,
if there are 170 most common sub-categories in the gigs in the listings of all search queries in a day. 
Usually some sub-categories are associated with very few search queries, so they will be clustered into less
than 50 clusters. The total number of clusters that were found for 7 days in March (25/3-31/3) is 53595,
but grouping by the cluster name and date gave the final number of 31088 unique clusters over all these 7 days. 
 
## Identification of Clusters over different days

The naming of the cluster is meant to help identify same clusters of different days.
Each cluster is named after the sub-category and a second part of the name, following these rules:
* In case the cluster query closest to the cluster centroid and the most common cluster in the query are equal, the second 
part of the clusters name is made of this name.
* In case the cluster has only one search query in it, it the second part of the cluster name will be this name.
* In all other cases, the second part of the cluster name will be comprised of both the query closest to centroid
and of the most common query in the cluster.

## Evaluation of the clusters

The results for the Doc2vec model contain both average silhouette score over all days, and the centroid distance
score of all days (TF-IDF model contains only silhouette score). The second score, which is the cosine similarity of the
query closest to the centroid to cluster centroid, is found to be a better metric of the goodness of the cluster 
than the silhouette score. That's the main reason why Doc2Vec was selected over TF-IDF.
If the data contains more than one day, the results will be evaluated for each relevant feature for all clusters and
all days.  

## Final Outputs

If all saving parameters are kept as default and we simply run the main file,
the following results will be saved as files in the file system:

1. The original data frame with the cluster labeling, with one row for each unique search query.
2. The aggregated clusters data, with one row for each cluster in each day.
3. The vectorizors used for representing the search queries
4. The kmeans models used for clustering the search queries<br> 
----- For data of more than one day -----<br>
5. The evaluated features over each cluster and over all days, with two scores of the clusters.
6. The plots of the features of either clusters that appear on all days or if there are none,
   the highest score clusters data on each evaluated feature 

* The process also reports the filenames of all models and data saved,
and the running time of the entire process.
* Use the unpickle_something function for loading a pickle file saved (and pandas.read_csv for csv).

## Installation

`pip install importlib`<br>
`pip install yaml`<br>
`pip install datetime`<br>
`pip install os`<br>
`pip install pickle`<br>
`pip install contextlib`<br>
`pip install pandas`<br>
`pip install numpy`<br>
`pip install matplotlib`<br>
`pip install -U scikit-learn`<br>
`pip install nltk`<br>
`pip install gensim`<br>
`pip install re`<br>
`pip install collections`<br>

## Usage

1. Download the search queries data csv file from BigQuery using this command:
`select search_date, search_query,
sum(number_of_searches) number_of_searches,
sum(searches_with_clicks) searches_with_clicks,
avg(total_number_of_search_results) total_number_of_search_results,
sum(orders) orders,
sum(revenue),
most_common_sub_category_name, 
most_common_category_name
from `itc-data-science.dwh.search_keywords`
where search_query is not null
and search_date = '2019-11-20'
group by search_date, search_query, most_common_sub_category_name,most_common_category_name
order by 2`

You can change the `search_date` to either one day (for daily analysis) or several days (for long term trends research).
2. update configs.yaml `fname_for_import` value to the absolute path of the downloaded csv (you can leave the rest of 
the values as default, or change them as you wish)
3. run `python main.py`, or load the function in the `utils.py` file and run them separately. 

### Changing Running Parameters (configs.yaml)

* The first bulk of values is of the *Production* values - 
the complete pipeline (aka last function of the complete pipeline `import_process_cluster_output_log`)
uses these values. change only if you already tested those values.
* The second bulk of values is of the *Research* values -
The functions below the highest functions use these values as default.
You can change those values and run the lower function for optimizing the algorithem.
* You can also overrule the yaml values when calling a function, while specifying values
for the function parameters (otherwise the parameters defaults will be used,
which are in most cases the yaml file values)

### The Pipeline Structure (utils.py)

The pipeline was built using functional programming, so the data passes
from function to function to function to get a final result (in comparison to OOP, where classes are instantiated).
This way, you can either run the top function and get the final result,
or run separately each function below to get intermediate results. Either way, whenever calling a function,
all functions below that function are called as well. 
While you can always automatically save the final results,
some lower functions contain saving and loading models and NLP vectorizors
 capabilities (see functions parameters and description). 

Hence, here's the functions order of running:
1. import_process_cluster_output_log: run entire pipeline logging capability
    2. import_process_cluster_output: run entire pipeline
        3. import_df: import the search queries df (as exported from BigQuery)
        4. pre_process_df: prepare the df for clustering
            5. clean_search_query: clean the language data
        6. find_kmeans_clusters_per_subcategory_per_day: clusters search queries and adds the cluster labels to df
            7. get_separate_days_df_dict: turn df into dict (runs, but obsolete)
            8. find_kmeans_clusters_per_subcategory_days_dict: clusters the queries df when it's in his dict form
                9. find_kmeans_clusters_per_subcategory: clusters search queries and adds the cluster labels to df
                    10. get_subcategories_indices: find the indices of queries for each subcategory
                    11. nlp_process: prepares the language data representation for clustering
                        12.  nlp_process_words_counts_and_searches/nlp_process_bow/nlp_process_tfidf/nlp_process_doc2vec:
                             functions for representation specific nlp processing, they also call lower functions
                        
                        ----- Here's where the K-means clustering is called! -----
                        13.  get_clusters: generates clusters meta-data  from the k-means labels - most common query, 
                             query closest to centroid and scoring
                             14. most_cosine_similar_query_doc2vec/most_cosine_similar_query_doc2vec_rank/most_cosine_similar_vector:
                                 scoring the clusters functions, some use cosine_similarity_two_vectors function below.
    15. aggregated_clusters: gives daily-unique name for each cluster and group the clusters
        over the original data frame features.
        16. columns_in_df: return all columns from a list that are in a data frame
        17. feature_engineering: adds the price feature for each cluster
    
    ----- if the queries df has several days, continue to the functions below -----
    16. evaluate_aggregated_clusters_over_all_features: creates a dict with a key for each feature data over all days in all clusters
        17. evaluate_aggregated_clusters: organize data over all days in all clusters for a specific feature
    18. add_silhouette_score_over_all_days: create an average silhouette score of all days for all features evaluated
    19. add_centroid_score_over_all_days: create an average centroid distance score of all days for all the features
    20. get_and_plot_clusters_all_days: plots the evaluated features over time and cluster, 
        for the highest score clusters.
        21. plot_cluster_evaluated_all_features: if there are cluster that appear in all days, plot them. else - 
        21. plot_counts_and_highest_values_over_numeric_features: plot the top scored clusters with highest values and
            counts.
    22. pickle_something: saves a pickle of the evaluated features over days over each cluster 
    
    ----- The functions below will run also for just one day -----
    23. pickle_something: saves a pickle of the kmeans models used  for labeling each search query (there's one model 
        for each sub category in each day), a pickle of the vectorizers used for representing the language data, and a
        pickle for dictionary 
    24. df_save: saves the queries df with the labeling data (a row for each search query),
        and the aggregate clusters data (a row for each cluster in each day) 
     
     
For more information, contact me: Dor.meir999@gmail.com.