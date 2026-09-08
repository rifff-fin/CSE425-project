# MusicCaps Human Evaluation

## What is required

Five independent listeners should rate the retrieved audio clips against the query caption. Do not use model scores as human ratings.

## Collection method for this group

Use the provided CSV directly. The five group members are the five listeners, so no Google Form is required. Each member should use one listener ID (`1` through `5`) and complete the rows assigned to that ID.

## How to participate

1. Open `human_evaluation.csv` and filter the rows by your assigned `listener_id`.
2. For each row, read the query caption and open the retrieved audio clip.
3. Open the audio file whose ID is shown in the matching retrieval mapping:
   - `results/retrieval_examples/musiccaps_examples.json`
4. Listen to the retrieved audio clip for at least 10 seconds.
5. Enter one rating in the `rating_1_to_5` column:
   - `1`: does not match the caption
   - `2`: weak match
   - `3`: partial match
   - `4`: good match
   - `5`: very strong match
6. Optionally record a short explanation in the `notes` column.
7. Each listener should rate all 30 rows belonging to their listener ID.

There are 10 queries, 3 retrieved clips per query, and 5 listener IDs, so the sheet contains 150 rating rows.

## Important

Listeners should rate independently and should not see another listener's scores before finishing. Do not change the `query_caption`, retrieved IDs, ranks, or model similarity information. Only fill `rating_1_to_5` and, optionally, `notes`. Do not fill missing ratings with guesses.

## Aggregate the completed ratings

From the project root, run:

```powershell
python src/aggregate_human_evaluation.py `
  --input results/retrieval_examples/human_evaluation.csv `
  --output results/retrieval_examples/human_evaluation_summary.json
```

The summary reports the number of completed ratings, mean score, score distribution, and mean score by retrieval rank. If ratings are missing, the script reports the missing count instead of treating missing rows as zero.
