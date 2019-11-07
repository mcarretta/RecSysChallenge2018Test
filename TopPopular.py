import csv
from scipy.sparse import csr_matrix
import numpy as np


class TopPopRecommender:

    def __init__(self):
        pass

    def fit(self, URM_train):
        """ --- CSR Matrix Computation --- """
        # Vector with as many ones as the number of interactions (the len of URM_train)

        total_interactions = np.array([1] * len(URM_train))

        # Create two vectors containing the list of the playlists and of the songs
        playlists_list, songs_list = [], []
        for tuple_in_URM_train in URM_train:
            playlists_list.append(int(tuple_in_URM_train[0]))
            songs_list.append(int(tuple_in_URM_train[1]))

        # Construct the CSR matrix (no need to construct the COO and then convert it into CSR with np.tocsr())
        self.URM_CSR = csr_matrix((total_interactions, (np.array(playlists_list), np.array(songs_list))))

        """ --- Actual Popularity computation --- """
        # Calculate item popularity by summing for each item the rating of every use
        # The most popular playlist is the one with more songs in it
        item_popularity = (self.URM_CSR > 0).sum(axis=0)
        # Squeeze term removes single-dimensional entries from the shape of an array.
        item_popularity = np.array(item_popularity).squeeze()

        # Ordering the items according to the popularity values
        self.popular_items = np.argsort(item_popularity)

        # Flip order high to lower
        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, at=10):
        recommended_items = self.popular_items[0:at]

        return recommended_items

# TODO: Evaluator and MAP class
def prepare_data(filename):
    # Opens file and appends all the tuples of Training data to the URM.
    # In this case, playlist i which has track j, k, l...
    URM_tuples = []

    with open(filename) as csv_input:
        input_reader = csv.reader(csv_input, delimiter=',')
        for row in input_reader:
            URM_tuples.append(tuple(row))
        csv_input.close()

    return URM_tuples[1:]  # First row contains columns headers, not useful


if __name__ == "__main__":
    URM_data = prepare_data("data/train.csv")
    topPopular = TopPopRecommender()
    topPopular.fit(URM_data)
    print(topPopular.recommend(10))
