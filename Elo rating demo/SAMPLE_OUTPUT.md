# Sample Output — Elo Math Quiz Demo

This is a recorded test run of the Elo Math Quiz agent. It shows how the system
generates questions, adapts difficulty based on the user's current Elo score, validates
answer keys before serving them, and updates the rating after every round.

---

## Session Overview

| | |
|---|---|
| **Starting Elo** | 100.0 |
| **Final Elo** | 147.3 |
| **Rounds played** | 10 |
| **Model** | google/gemma-4-26B-A4B-it via HuggingFace |

---

## Round-by-Round Breakdown

### Round 1 — Ratio `MEDIUM`
> **Q:** The ratio of the number of red marbles to blue marbles in a bag is 3:5.
> If there are 40 more blue marbles than red marbles, how many marbles are there in the bag in total?

```
Answer   : 160
Result   : ✅ CORRECT
Elo      : 100.0 → 116.0  (+16.0)
```

---

### Round 2 — Linear Equations `MEDIUM`
> **Q:** Solve for x in the equation: 3(x - 4) = 2x + 5

```
Answer   : 17
Result   : ✅ CORRECT
Elo      : 116.0 → 131.3  (+15.3)
```

---

### Round 3 — Algebraic Expansion & Factorisation `HARD`
> **Q:** Expand and simplify the expression: (2x + 3)(x - 5) - (x² - 4x + 2).
> Then, factorise the resulting quadratic expression completely.

```
Answer   : x² - 3x - 17
Result   : ❌ INCORRECT  (expected: x - 7)
Feedback : Incorrect algebraic expression.
Elo      : 131.3 → 115.2  (-16.1)
```

---

### Round 4 — Percentages `MEDIUM`
> **Q:** A shopkeeper bought a jacket for $80. He wants to sell it at a profit of 25%.
> However, during a sale, he offers a 10% discount on the marked selling price.
> What is the final selling price of the jacket?

```
Answer   : 90
Result   : ✅ CORRECT
Elo      : 115.2 → 130.5  (+15.3)
```

---

### Round 5 — Fractions `EASY`
> **Q:** Calculate the value of 3/4 + 1/8.

```
Answer   : 7/8
Result   : ✅ CORRECT
Elo      : 130.5 → 144.2  (+13.7)
```

> 💡 **Note the smaller gain (+13.7 vs +16.0 in Round 1):** As the user's Elo rises above the
> question's rating, the expected score increases — so winning gives fewer points.
> This is Elo's self-balancing mechanism in action.

---

### Round 6 — Geometry `EASY`
> **Q:** A rectangle has a length of 8 cm and a width of 5 cm.
> Calculate the perimeter of the rectangle.

```
Answer   : 26
Result   : ✅ CORRECT  (numerically equivalent — "26 cm" accepted as "26")
Elo      : 144.2 → 157.3  (+13.1)
```

---

### Round 7 — Algebraic Manipulation `HARD`
> **Q:** Given that x + y = 10 and x² + y² = 58, find the value of the product xy.

```
Answer   : 21
Result   : ✅ CORRECT
Elo      : 157.3 → 172.0  (+14.7)
```

---

### Round 8 — Ratio `MEDIUM`
> **Q:** The ratio of the number of red marbles to blue marbles in a bag is 3:5.
> If there are 40 more blue marbles than red marbles, how many marbles are there in the bag in total?

```
Answer   : 180
Result   : ❌ INCORRECT  (expected: 160)
Feedback : Incorrect numerical value.
Elo      : 172.0 → 152.7  (-19.3)
```

> 💡 **Note the larger penalty (-19.3):** At Elo 172.0 against a medium question rated 100,
> the user was heavily favoured. Losing an "expected" match costs more points.
> This mirrors how an upset loss in chess costs a strong player heavily.

---

### Round 9 — Linear Equations `MEDIUM`
> **Q:** Solve for x in the following equation: 4(x - 3) = 2x + 8

```
Answer   : 10
Result   : ✅ CORRECT
Elo      : 152.7 → 166.3  (+13.6)
```

---

### Round 10 — Expansion & Factorisation `MEDIUM`
> **Q:** Expand and simplify the expression: (2x + 3)(x - 4)

```
Answer   : 2x² + 6x - 12
Result   : ❌ INCORRECT  (expected: 2x² - 5x - 12)
Feedback : Incorrect algebraic expression.
Elo      : 166.3 → 147.3  (-19.0)
```

---

## Final Score

```
=======================================================
  Final Elo Score : 147.3  (started at 100.0)
  Rounds played   : 10
=======================================================
```

---

## What This Demonstrates

**Adaptive difficulty** — The system weighted question difficulty based on the user's
current score. Once the score exceeded 120, hard questions appeared more frequently
(Rounds 3, 7) while easy questions were still occasionally served (Rounds 5, 6).

**Proportional Elo changes** — Early wins against equal-rated questions gave +16 points.
Later wins against easier questions gave only +13 points. Losses against easier questions
cost more (-19) than losses against hard questions would. The system is always proportional
to how surprising the result was.

**Answer equivalence** — In Round 6, the answer `26` was accepted against an expected
answer of `26 cm`. The verifier treats numerically equivalent forms as correct.

---

*For the full source code and documentation see [`elo_demo.py`](elo_demo.py) and [`docs/elo_rating_system.docx`](docs/elo_rating_system.docx).*
