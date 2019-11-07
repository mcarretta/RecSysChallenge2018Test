import numpy as np
from  utils.helper import Helper


class TopPopRecommender:

    def __init__(self):
        self.helper = Helper()

    def fit(self):
        """ --- CSR Matrix Computation --- """
        # Vector with as many ones as the number of interactions (the len of URM_train)
        self.URM_CSR = self.helper.convert_URM_data_to_csr()
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
    top_popular.fit()
    print(top_popular.recommend(1000))
