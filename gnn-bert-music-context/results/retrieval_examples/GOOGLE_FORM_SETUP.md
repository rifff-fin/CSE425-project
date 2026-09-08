# MusicCaps Human Evaluation Google Form

## Is a Google Form required?

The project requires five independent listeners to complete 150 human ratings for Task 4. It does not require a particular tool, but a Google Form linked to Google Sheets is recommended because listeners submit independently and cannot see other listeners' answers.

## Form design

Create a Google Form with:

- Title: `MusicCaps Human Evaluation`
- One required short-answer question: `Listener ID` with values `1` through `5`
- Ten sections, one for each query caption
- Three required linear-scale questions in each section: `Rank 1`, `Rank 2`, and `Rank 3`
- Each scale must run from `1` to `5`
- Scale labels: `Does not match` and `Very strong match`
- An optional paragraph question named `Notes` in each section

Each rank question must include a playable Google Drive link to the matching retrieved WAV file. The query caption should be shown in the section description. Listeners should hear each clip for at least 10 seconds before submitting a rating.

## Audio and link preparation

Upload the 30 retrieved WAV files to a Google Drive folder. Set each file to `Anyone with the link - Viewer`, if permitted by your course and dataset licensing requirements. Keep a private mapping of `query_audio_id`, `retrieved_rank`, `retrieved_audio_id`, and `drive_url`.

Do not use local paths such as `D:/425 project/...` in the form. Those paths work only on the owner's computer.

Before publishing the form, verify that every rank link points to the same retrieved audio ID used by `musiccaps_examples.json`. The CSV template and retrieval JSON must be regenerated from the same final retrieval output; otherwise listeners may rate the wrong clips.

## Responses and aggregation

In Google Forms, open `Responses` and select `Link to Sheets`. After all five listeners submit, export the response sheet as CSV. Rename the columns to match the project evaluator or convert the form responses into this schema:

```text
listener_id,query_audio_id,query_caption,retrieved_rank,retrieved_audio_id,rating_1_to_5,notes
```

Run from `gnn-bert-music-context`:

```powershell
python src/aggregate_human_evaluation.py `
  --input results/retrieval_examples/human_evaluation.csv `
  --output results/retrieval_examples/human_evaluation_summary.json
```

The expected total is 150 completed ratings: 5 listeners x 10 queries x 3 retrieved clips.

## Gemini prompt

Paste the following prompt into Gemini after providing the final retrieval JSON, the 30 WAV links, and the Google Sheet or Google Drive destination:

```text
Create a Google Form for the human evaluation of my CSE425 GNN-BERT Music Context project.

Use the attached retrieval JSON as the source of truth for the 10 query captions and the three retrieved clips for each query. Use the attached WAV-link table as the source of truth for playable audio links. Do not invent captions, audio IDs, ranks, scores, or links. If any query/rank has no matching WAV link, stop and list the missing mapping instead of creating an incorrect question.

Form title:
MusicCaps Human Evaluation

Form description:
Listen to every retrieved audio clip for at least 10 seconds. Compare each clip with the query caption and rate how well it matches. Use 1 for does not match, 2 for weak match, 3 for partial match, 4 for good match, and 5 for very strong match. Ratings must be independent. Do not show respondents other listeners' answers.

Create these fields and settings:
1. Required short-answer question: Listener ID. Explain that the assigned values are 1, 2, 3, 4, or 5.
2. Create one section for each of the 10 query captions.
3. In each section, show the exact query caption in the section description.
4. Add exactly three required linear-scale questions, one for each retrieved rank in the JSON. Use a 1-to-5 scale with labels "Does not match" and "Very strong match".
5. Put the matching playable WAV link and retrieved audio ID in each question title. The title must identify the query ID and rank.
6. Add an optional paragraph question named Notes after the three ratings.
7. Do not collect email addresses unless I explicitly enable that setting.
8. Do not show a results summary to respondents.
9. Link responses to a Google Sheet.

Before finalizing, verify:
- 10 query sections exist.
- Every query has exactly 3 rating questions.
- There are exactly 30 audio links.
- All ranks are 1, 2, and 3.
- No local Windows paths such as D:/425 project appear in the form.
- No model similarity score is presented as a human rating.

After creating the form, return:
- The edit URL and respondent URL.
- The linked response-sheet URL.
- A table mapping query_audio_id, rank, retrieved_audio_id, and question title.
- Any missing or ambiguous audio links.
```

Gemini may be unable to create files in your Drive unless you authorize the relevant Google account and Workspace integration. If it cannot create the Form directly, ask it to generate a Google Apps Script that creates the same Form and response Sheet, then review the script before running it.