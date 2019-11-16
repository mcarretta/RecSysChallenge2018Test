import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from sklearn import feature_extraction
import os
from base.IR_feature_weighting import okapi_BM_25
# Put root project dir in a global constant
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Helper:
    def __init__(self):
        # Put root project dir in a constant
        # Loading of data using pandas, that creates a new object URM_data with the first line values as attributes
        # (playlist_id, track_id) as formatted in the .csv file
        self.URM_data = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/train.csv"))
        self.playlists_list = np.asarray(list(self.URM_data.playlist_id))
        self.tracks_list = np.asarray(list(self.URM_data.track_id))


    def convert_URM_data_to_csr(self):
        ratings_list = np.ones(len(self.playlists_list))
        URM = sps.coo_matrix((ratings_list, (self.playlists_list, self.tracks_list)))
        URM = URM.tocsr()
        return URM


    def load_tracks_matrix(self):
        tracks_matrix = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/tracks.csv"))
        return tracks_matrix

    def load_icm_album(self):
        tracks_matrix = self.load_tracks_matrix()
        track_ids = np.asarray(list(tracks_matrix.track_id))
        album_ids = np.asarray(list(tracks_matrix.album_id))
        ratings_list = np.ones(len(album_ids))
        icm_album = sps.coo_matrix((ratings_list, (track_ids, album_ids)))
        icm_album = icm_album.tocsr()
        return icm_album

    def load_icm_artist(self):
        tracks_matrix = self.load_tracks_matrix()
        track_ids = np.asarray(list(tracks_matrix.track_id))
        artist_ids = np.asarray(list(tracks_matrix.artist_id))
        ratings_list = np.ones(len(artist_ids))
        icm_artist = sps.coo_matrix((ratings_list, (track_ids, artist_ids)))
        icm_artist = icm_artist.tocsr()
        return icm_artist

    def bm25_normalization(self, matrix):
        matrix_BM25 = matrix.copy().astype(np.float32)
        matrix_BM25 = okapi_BM_25(matrix_BM25)
        matrix_BM25 = matrix_BM25.tocsr()
        return matrix_BM25

    def tfidf_normalization(self, matrix):
        matrix_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(matrix)
        return matrix_tfidf.tocsr()

    def sklearn_normalization(self, matrix, axis=0):
        return normalize(matrix, axis=axis, norm='l2').tocsr()
