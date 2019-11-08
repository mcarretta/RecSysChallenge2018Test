import pandas as pd
import numpy as np
import scipy.sparse as sps
import os

class Helper:
    def __init__(self):
        # Put root project dir in a constant
        ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Loading of data using pandas, that creates a new object URM_data with the first line values as attributes
        # (playlist_id, track_id) as formatted in the .csv file
        self.URM_data = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/train.csv"))
        self.playlists_list = np.asarray(list(self.URM_data.playlist_id))
        self.tracks_list = np.asarray(list(self.URM_data.track_id))

    def convert_URM_data_to_csr(self):
        ratings_list = np.ones(len(self.playlists_list))
        URM = sps.coo_matrix((ratings_list, (self.playlists_list, self.tracks_list)))
        URM.tocsr()
        return URM

