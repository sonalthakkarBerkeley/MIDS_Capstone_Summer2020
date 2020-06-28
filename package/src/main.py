
class Irrigation30:

    def __init__(self, region_of_interest):

        # region_of_interest = {
        #     lat: 43.771114,
        #     lon: -116.736866,
        #     edge_len: 0.005,
        #     resolution: 30
        # }

        self.region_of_interest = region_of_interest
        _determine_subregions(region_of_interest)

        pass

    # Alternative name: get_labels
    def predict(self, region_of_interest):
        pass

    def fit_clustering(self):
        pass

    # helper
    def _determine_subregions(self, region_of_interest):
        self.subregions = None
        pass

    def get_subregions(self):
        return self.subregions
