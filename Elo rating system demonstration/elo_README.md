# Elo Rating System

> A LangChain-powered adaptive math quiz that uses the Elo Rating System to measure and update a student's ability in real time. Questions are AI-generated, difficulty adapts to the user's current score, and answers are verified through a layered checking pipeline.

---

## What Is the Elo Rating System?

The Elo Rating System is a method for measuring the **relative skill** between two participants. Originally designed for chess, it works by assigning every participant a rating number and updating both ratings after every encounter based on one principle:

> *The more surprising the result, the bigger the rating change.*

In this demo, the two "participants" are the **student** (their ability rating) and the **question** (its difficulty rating). A correct answer means the student wins — an incorrect answer means the question wins. Both ratings update after every round.

The core formulas:

```
# Step 1 — Predict the outcome
Expected = 1 / (1 + 10 ^ ((Question_Rating - Student_Rating) / 400))

# Step 2 — Update after the result
New_Rating = Old_Rating + K × (Actual - Expected)

Actual: 1.0 = correct, 0.0 = incorrect
K:      sensitivity factor (32 in this demo)
```

For a full explanation with worked examples, see [`docs/elo_rating_system.docx`](docs/elo_rating_system.docx).

---

## How This Demo Works

```
Student answers a question
        ↓
Elo expected score calculated (how likely were they to get it right?)
        ↓
Answer verified through layered pipeline (local solver → LLM fallback)
        ↓
Both student and question ratings updated
        ↓
Next question difficulty weighted by new student score
```

### Difficulty Ratings

| Difficulty | Question Rating | When Served |
|---|---|---|
| Easy | 80 | Weighted heavily when score < 90 |
| Medium | 100 | Weighted heavily when score 90–120 |
| Hard | 130 | Weighted heavily when score > 120 |

The demo intentionally uses a small rating scale (starting at 100) so Elo changes are clearly visible within just a few rounds — making it easy to observe the system's behaviour without a long session.

### Answer Verification Pipeline

Rather than relying solely on the LLM to judge answers, verification runs through four layers in order:

1. **Direct text match** — normalised comparison handles unicode, spacing, and operator symbols
2. **Equation match** — `x = 5` and `x = 5.0` treated as equivalent
3. **Numeric equivalence** — `0.75`, `3/4`, and `75%` all resolve to the same value; units like `cm` and `°` are stripped
4. **LLM fallback** — solves the question independently before judging, uses its own answer as ground truth rather than deferring to the generated key

Answers the system cannot confidently verify receive an **UNCERTAIN** status — no Elo change is applied, so the user is never penalised unfairly.

---

## Sample Output

See [`SAMPLE_OUTPUT.md`](SAMPLE_OUTPUT.md) for a full recorded test run with round-by-round commentary explaining the Elo changes.

---

## Getting Started

### Requirements

- Python 3.10+
- A HuggingFace account with API access

### Install Dependencies

```bash
pip install langchain-huggingface python-dotenv
```

### Set Up Your API Token

Create a `.env` file in this folder (use `.env.example` as a template):

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
HF_TOKEN=your_token_here
```

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

### Run the Demo

```bash
python Elo_rating_demo.py
```

Type `quit` or `exit` at any time to end the session.

---

## File Structure

```
elo/
├── Elo_rating_demo.py       ← Main demo script
├── SAMPLE_OUTPUT.md         ← Recorded test run with commentary
├── .env.example             ← Token setup template
└── docs/
    └── elo_rating_system.docx  ← Full written documentation
```

---

## Key Design Decisions

**Why a small Elo scale?** A starting score of 100 with question ratings between 80–130 means rating changes are immediately visible after just a few rounds. This is intentional for demo purposes — in a production system you would use a conventional scale (e.g. starting at 1000).

**Why is difficulty engine-selected, not model-selected?** The model returns a difficulty field in its JSON, but the demo ignores it and uses the engine's weighted selection instead. This ensures Elo fairness — if the model labels a question incorrectly, the wrong question rating would be used for the update.

**Why the `uncertain` status?** When the verifier cannot confidently judge an answer (e.g. symbolic forms the local solver can't handle), marking it wrong would be unfair. The `uncertain` state preserves the student's Elo until the next clear result.

---

*Part of the [Psychometric Models](../README.md) research repository.*
