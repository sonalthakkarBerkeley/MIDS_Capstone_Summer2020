
import ee
import package as p

# Opens terminal window, ask user to authenticate EE account
ee.Authenticate()  # only need to do once?
ee.Initialize()  # for every notebook/session

# Input
region_of_interest = {
    lat: 43.771114,
    lon: -116.736866,
    edge_len: 0.005,
    resolution: 30
}

# Instantiate Inference30 object
# Calculate subregions from input region_of_interest
# Throw error if region is unacceptable
# Optional: provide time estimates?
model = p.Inference30(region_of_interest)  # Size of subregions?

# find subregions to do clustering (on each)
# fit / label clusters on each subregion

# Extra method: return 30m subregion locations, calculated from init^
model.get_subregions()

# Fit unsupervised clustering model(s)
model.fit_clustering(region_of_interest)  # Optional argument?

# Return labels for subregions from region_of_interest
# Possible return formats:
# (1) Dataframe {lat1,lon1,lat2,lon2,label}
# (2) Dict {[(lat1,lon1),(lat2,lon2)]=label}
labels = model.predict(region_of_interest)  # Optional argument?
