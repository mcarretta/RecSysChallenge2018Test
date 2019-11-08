import numpy as np
from utils.run import RunRecommender

class TopPopRecommender:

    def __init__(self):
        pass

    def fit(self, URM_CSR):
        """ --- CSR Matrix Computation ---
            Training data loading and its conversion to a CSR matrix is done in run.py with the functions provided by Helper
        """
        self.URM_CSR = URM_CSR
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



if __name__ == "__main__":
    # noinspection Pylint
    top_popular = TopPopRecommender()
    runner = RunRecommender()
    runner.run(top_popular)
