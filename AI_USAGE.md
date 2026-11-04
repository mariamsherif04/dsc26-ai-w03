# AI usage

## Tools I used, and for what

I used Claude to draft the code in `tawseel/model.py` (history features, time split, pipeline, evaluation) and to suggest simple hand-written rules to compare against the model. I then read and revised the code myself and ran it to check it. I also suggested trying different rules, and we compared them with the model on the same last-20% split.

## AI-suggested code I rejected or fixed in add_history_features or time_split

I did not reject or change any of the AI-suggested logic in `add_history_features` or `time_split`. I reviewed both by hand against the task description (strictly earlier timestamps, ties not counted as prior, all rows at the boundary timestamp go to test) and the code matched it, so I kept it. I ran the public tests and the results check, and nothing showed a problem.

## How I checked that my model has no leak

FEATURE_COLUMNS excludes courier_notes, delivered_at, failed and parcel_id, history uses strictly earlier timestamps, and the test ROC AUC is about 0.64 (0.64–0.69 across seeds 0–3), far below the 0.9 leak threshold.