from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from tawseel.data import make_parcels, VAGUE
from tawseel.model import add_history_features, time_split

df = add_history_features(make_parcels(4000, seed=0))
tr, te = time_split(df)
hi = tr.cod_amount.quantile(0.75)

def points(d):
    vague = d.address_text.isin(VAGUE) | (d.address_text == "")
    return (vague.astype(int) * 2
            + (d.cod_amount.fillna(0) > hi).astype(int)
            + (d.promised_slot == "evening").astype(int))

rate = tr.assign(s=points(tr)).groupby("s").failed.mean()
prob = points(te).map(rate).astype(float)
y = te.failed
print(round(roc_auc_score(y, points(te)), 3),
      round(average_precision_score(y, prob), 3),
      round(brier_score_loss(y, prob), 3))