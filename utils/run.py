from tqdm import tqdm
from utils.helper import Helper
import os


class RunRecommender:

    @staticmethod
    def run(recommender):
        # Put root project dir in a constant
        ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Open up target_playlists file
        target_playlists_file = open(os.path.join(ROOT_PROJECT_PATH, "data/target_playlists.csv"), "r")

        # Helper contains methods to convert URM in CSR
        helper = Helper()
        URM_CSR = helper.convert_URM_data_to_csr()

        # Start recommendation
        recommender.fit(URM_CSR)

        # Instantiate a new submission file
        target_playlists = list(target_playlists_file)[1:] # Eliminate header
        recommendation_matrix_file_to_submit = open(os.path.join(ROOT_PROJECT_PATH, "data/recommendation_matrix_to_submit.csv"), "w")
        recommendation_matrix_file_to_submit.write("playlist_id,track_ids\n")
        for playlist in tqdm(target_playlists):
            songs_recommended = recommender.recommend()
            print(len(songs_recommended))
            songs_recommended = " ".join(str(x) for x in songs_recommended)
            playlist = playlist.replace("\n", "") # remove \n from playlist file
            recommendation_matrix_file_to_submit.write(playlist + "," + songs_recommended + "\n")
        recommendation_matrix_file_to_submit.close()




