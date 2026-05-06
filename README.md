# 📊 Psychometric Models — A Research & Learning Repository

> A growing collection of psychometric and assessment models used in adaptive learning systems — with working implementations, documentation, and plain-English explanations for each topic. No prior Machine Learning knowledge required.

---

## 🧠 What Are Psychometric Models?

Psychometric models are mathematical frameworks designed to **measure human ability, knowledge, and learning** in a rigorous and quantifiable way.

The word comes from the Greek *psyche* (mind) and *metron* (measure). In short — these are models that try to answer questions like:

- How skilled is this person right now?
- How difficult is this task relative to others?
- How quickly is this person learning?
- What should they practise next?

These models sit at the intersection of **statistics, cognitive science, and education technology**. They are the backbone of adaptive learning platforms, standardised testing systems, and intelligent tutoring systems used worldwide.

---

## 🏗️ How This Repo Is Organised

Each psychometric model lives in its own folder with its own README, code, and documentation. This repository acts as the central index.

```
psychometric-models/
│
├── README.md               ← You are here — the central index
├── docs/                   ← Shared reference documents
└── [topic-name]/
    ├── README.md           ← Topic-specific explanation & usage
    ├── basics.py           ← Core implementation
    ├── simulation.py       ← Demo / simulation
    └── docs/               ← Topic documentation (e.g. .docx)
```

New topics are added as self-contained folders. Each one is independent — you can read and run any topic without needing to understand the others first.

---

## 📂 Topics

| Topic | Folder | Status |
|---|---|---|
| *(More coming soon)* | — | — |

---

## ▶️ General Setup

All implementations are written in **Python 3.8+**. Each topic folder lists its own specific dependencies in its README, but the common ones are:

```bash
pip install numpy matplotlib scikit-learn
```

Clone the repo to get started:

```bash
git clone https://github.com/YOUR_USERNAME/psychometric-models.git
cd psychometric-models
```

Then navigate into any topic folder and follow its README.

---

## 🗺️ Scope of This Repository

This repo covers models across several categories of psychometric and educational research. As new topics are added they will fall into one or more of these areas:

**Assessment & Ability Measurement** — Models that estimate a learner's skill level from their responses.

**Learning & Knowledge Tracing** — Models that track how knowledge changes over time with practice.

**Memory & Retention** — Models that predict how well information is retained and when it should be reviewed.

**Adaptive & Instructional Systems** — Models that decide what a learner should do or see next.

---

## 📚 Background Reading

If you are new to this area, these are good starting points before diving into any specific model:

- Baker, R. & Inventado, P. (2014). *Educational Data Mining and Learning Analytics.* Springer.
- Embretson, S. & Reise, S. (2000). *Item Response Theory for Psychologists.* Lawrence Erlbaum.
- [Carnegie Learning Research](https://www.carnegielearning.com/research/) — applied psychometrics in real classrooms
- [PSLC DataShop](https://pslcdatashop.web.cmu.edu) — open dataset repository for educational interaction data

---

*This repository is part of an ongoing research study into psychometric and assessment models used in adaptive learning systems.*
