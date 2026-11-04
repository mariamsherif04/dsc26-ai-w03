from tawseel.data import make_parcels
from tawseel.model import *

for s in [0, 1, 2, 3]:
    df = add_history_features(make_parcels(4000, seed=s))
    tr, te = time_split(df)
    r = compare_to_baseline(tr, te)
    print(s, {k: round(v, 3) for k, v in r["model"].items()}, "lift", round(r["lift_pr_auc"], 3))