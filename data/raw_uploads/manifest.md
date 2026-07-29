# Raw Uploads Manifest

| File | Type | Format | Notes | Suggested Use |
|------|------|--------|-------|---------------|
| `Consumer Interview_Pinky.docx` / `Consumer_Interview_Pinky.md` | `interview_transcript` | DOCX / Markdown | Consumer interview summary + detailed Q&A transcript (Speaker 1 & Speaker 2). Covers drinking habits, whisky occasions, brand perception, purchase channels, barriers. | Domain reference / few-shot examples for answer probing patterns. |
| `Young Drinker Interview_Mr.Cen.docx` / `Young_Drinker_Interview_Mr_Cen.md` | `interview_transcript` | DOCX / Markdown | Young-drinker interview summary + detailed Q&A transcript. Focus on whisky entry path, selection criteria, brand perception, emotional/social drivers. | Domain reference / few-shot examples for transition and probe decisions. |
| `Young Drinker Interview_Mr.Li.docx` / `Young_Drinker_Interview_Mr_Li.md` | `interview_transcript` | DOCX / Markdown | Young-drinker interview summary + detailed Q&A transcript. Covers drinking frequency, whisky preference development, social acceptance, barriers. | Domain reference / few-shot examples for end-of-interview and probe decisions. |

**Important:** These are human-conducted interview transcripts, not OW-Text session exports. They do not contain per-turn `action` labels (`ask`/`probe`/`transition`/`end`), so they cannot be used directly as a turn-level training set. They are useful for:
- Understanding the interview domain and vocabulary.
- Extracting high-quality question/answer pairs for few-shot prompting.
- Building rules/heuristics for what counts as a "vague" answer or a natural transition.
