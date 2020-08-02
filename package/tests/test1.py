import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import irrigation30 as irr

# irr.authenticate()

model = irr.Irrigation30(center_lat=15.8005146, center_lon=77.97976826,
                         edge_len=0.005, num_clusters=2)

model.fit_predict()

model.simple_label
