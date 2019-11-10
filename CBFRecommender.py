"""
Created on 09/11/2019

@author: Matteo Carretta
"""

import numpy as np
from base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from utils.helper import Helper
from utils.run import RunRecommender


class CBFRecomender:

    def __init__(self, knn_artist = 100, knn_album = 100, shrink_artist = 2, shrink_album = 2, weight_artist = 0.4):
        self.knn_artist = knn_artist
        self.knn_album = knn_album
        self.shrink_artist = shrink_artist
        self.shrink_album = shrink_album
        self.weight_artist = weight_artist
        self.helper = Helper()

    def compute_similarity_cbf(self, ICM, top_k, shrink, normalize = True, similarity = "cosine"):
        #Compute similarities for weighted features recommender
        similarity_object = Compute_Similarity_Python(ICM.T, shrink = shrink, topK = top_k, normalize = normalize, similarity = similarity)
        w_sparse = similarity_object.compute_similarity()
        return w_sparse

    def fit(self, URM):

        # URM Loading
        self.URM = URM

        # Load ICMs from helper
        self.ICM_artist = self.helper.load_icm_artist()
        self.ICM_album = self.helper.load_icm_album()

        # Computing SMs
        self.SM_artist = self.compute_similarity_cbf(self.ICM_artist, self.knn_artist, self.shrink_artist)
        self.SM_album = self.compute_similarity_cbf(self.ICM_album, self.knn_album, self.shrink_album)

    def compute_scores(self, playlist_id):
        tracks_list_train = self.URM[playlist_id]
        scores_artist = tracks_list_train.dot(self.SM_artist).toarray().ravel()
        scores_album = tracks_list_train.dot(self.SM_album).toarray().ravel()

        weight_album = 1 - self.weight_artist
        scores = (scores_artist * self.weight_artist) + (scores_album * weight_album)
        return scores

    def recommend(self, playlist_id, at = 10, exclude_seen = True):
        # Compute scores of the recommendation
        scores = self.compute_scores(playlist_id)

        # Filter to exclude already seen items
        if exclude_seen:
            scores = self.filter_seen(playlist_id, scores)
        recommended_items = np.argsort(scores)
        recommended_items = np.flip(recommended_items, axis = 0)
        return recommended_items[:at]

    def filter_seen(self, playlist_id, scores):
        start_pos = self.URM.indptr[playlist_id]
        end_pos = self.URM.indptr[playlist_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

if __name__ == "__main__":
    cbf_recommender = CBFRecomender()
    runner = RunRecommender()
    runner.run(cbf_recommender)
