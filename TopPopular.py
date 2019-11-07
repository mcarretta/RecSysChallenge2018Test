import csv
from scipy.sparse import csr_matrix
import numpy as np


class TopPopRecommender(object):

    def fit(self, URM_train):
        """ --- CSR Matrix Computation --- """
        # Vector with as many ones as the number of interactions (the len of URM_train)
        totalInteractions = np.ones([1] * len(URM_train))

        # Create two vectors containing the list of the playlists and of the songs
        playlistsList, songsList = zip(*URM_train)
        playlistsList = list(playlistsList)
        songsList = list(songsList)

        # Construct the CSR matrix (no need to construct the COO and then convert it into CSR with np.tocsr())
        self.URM_CSR = csr_matrix(totalInteractions, (np.array(playlistsList), np.array(songsList)))

        """ --- Actual Popularity computation --- """
        # Calculate item popularity by summing for each item the rating of every use
        # The most popular playlist is the one with more songs in it
        itemPopularity = (URM_train > 0).sum(axis=0)
        # Squeeze term removes single-dimensional entries from the shape of an array.
        itemPopularity = np.array(itemPopularity).squeeze()

        # Ordering the items according to the popularity values
        self.popularItems = np.argsort(itemPopularity)

        # Flip order high to lower
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, at=10):
        recommended_items = self.popularItems[0:at]

        return recommended_items


def prepare_data(filename):
    # Opens file and appends all the tuples of Training data to the URM.
    # In this case, playlist i which has track j, k, l...
    URM_tuples = []

    with open(filename) as csvinput:
        inputreader = csv.reader(csvinput, delimiter=',')
        for row in inputreader:
            URM_tuples.append(tuple(row))
        csvinput.close()

    return URM_tuples[1:]  # First row contains columns headers, not useful


if __name__ == "__main__":
    URM_data = prepare_data("data/train.csv")
    topPopular = TopPopRecommender()
    topPopular.fit(URM_data)
    print(topPopular.recommend(10))
