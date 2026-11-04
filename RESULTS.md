# Results: Tawseel failed-delivery baseline

Last 20% of parcels by time (800 parcels), `make_parcels(4000, seed=0)`. History features are computed point-in-time, and the model is fitted on the earlier 80% only.

| Model | ROC AUC | PR-AUC | Brier | Base rate |
|---|---|---|---|---|
| Dummy (prior) | 0.500 | 0.220 | 0.172 | 0.22 |
| Rule: points (vague/empty address ×2, high COD, evening slot) | 0.590 | 0.262 | 0.172 | 0.22 |
| My model (logistic regression + customer history) | 0.638 | 0.318 | 0.166 | 0.22 |

- A hand-written rule reaches PR-AUC 0.262 (lift 0.042), while the model reaches 0.318 (lift 0.098), so the model is worth having over a simple rule.

- PR-AUC lift over the dummy is 0.098 on seed 0. On seeds 1, 2 and 3 the lift was 0.135, 0.142 and 0.165, and the model's Brier score was better than the dummy's every time.

## Three sentences for Sherif

On the most recent 20% of parcels, the model ranks risky deliveries clearly better than guessing, with PR-AUC 0.32 against 0.22 for the no-skill baseline.
It uses only information known when the parcel is created (area, time slot, COD amount, address quality and the customer's earlier parcels), never courier notes or delivery results, and the gain held on three other random datasets I tried.
It is a modest model, not a 99% one: it can help decide which parcels to confirm or call ahead, it will still miss many failures, and it should be re-checked on real data before anyone relies on it.