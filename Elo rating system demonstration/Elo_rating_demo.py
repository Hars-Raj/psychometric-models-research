import json
import math
import random
import re
from fractions import Fraction
from typing import List, Set

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ====================== 1. TOKEN SETUP ======================
load_dotenv()

# ====================== 2. LOAD MODEL ======================
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-26B-A4B-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3,
    do_sample=True,
)
chat_model = ChatHuggingFace(llm=llm)

# ====================== 3. ELO CONFIG ======================
DEFAULT_USER_ELO = 100.0
K_FACTOR = 32
RECENT_MEMORY = 6


def difficulty_to_rating(difficulty: str) -> int:
    mapping = {"easy": 80, "medium": 100, "hard": 130}
    return mapping.get(difficulty.lower(), 100)


def expected_score(player_rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_rating - player_rating) / 400.0))


def update_elo(player_rating: float, opponent_rating: float, actual_score: float, k: int = K_FACTOR) -> float:
    exp = expected_score(player_rating, opponent_rating)
    return player_rating + k * (actual_score - exp)


def extract_json_object(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def normalize_math_text(value: str) -> str:
    text = value.strip().lower()

    frac_match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    if frac_match:
        text = f"({frac_match.group(1)})/({frac_match.group(2)})"

    text = text.replace("^", "**")
    text = text.replace("\u2212", "-")
    text = text.replace("\u00d7", "*")
    text = text.replace("\u00f7", "/")
    text = re.sub(r"\s+", "", text)
    return text


def parse_numeric_value(value: str) -> float | None:
    text = normalize_math_text(value)
    if not text:
        return None

    text = text.replace(",", "")
    text = re.sub(r"^[\$\u00a3\u20ac]", "", text)
    text = text.replace("cm2", "").replace("m2", "").replace("mm2", "").replace("km2", "")
    text = text.replace("cm\u00b2", "").replace("m\u00b2", "").replace("mm\u00b2", "").replace("km\u00b2", "")
    text = re.sub(r"[a-z\u00b0]+$", "", text).strip()

    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1]

    if not re.fullmatch(r"[0-9+\-*/().]+", text):
        return None

    try:
        if "/" in text and re.fullmatch(r"[0-9]+/[0-9]+", text):
            numeric = float(Fraction(text))
        else:
            numeric = float(eval(text, {"__builtins__": {}}, {}))
    except Exception:
        return None

    if is_percent:
        numeric = numeric / 100.0
    return numeric


def parse_linear_equation_value(value: str) -> tuple[str, float] | None:
    raw = value.strip().lower()
    m = re.fullmatch(r"\s*([a-z])\s*=\s*([0-9+\-*/().]+)\s*", raw)
    if not m:
        return None
    var = m.group(1)
    rhs = parse_numeric_value(m.group(2))
    if rhs is None:
        return None
    return (var, rhs)


def question_signature(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return " ".join(normalized.split())


def is_repetitive(candidate_question: str, recent_sigs: Set[str]) -> bool:
    sig = question_signature(candidate_question)
    if sig in recent_sigs:
        return True

    cand_head = " ".join(sig.split()[:7])
    for old_sig in recent_sigs:
        old_head = " ".join(old_sig.split()[:7])
        if cand_head and cand_head == old_head:
            return True
    return False


def choose_difficulty_for_score(score: float) -> str:
    if score < 90:
        choices, weights = ["easy", "medium", "hard"], [0.70, 0.20, 0.10]
    elif score <= 120:
        choices, weights = ["easy", "medium", "hard"], [0.20, 0.60, 0.20]
    else:
        choices, weights = ["easy", "medium", "hard"], [0.25, 0.25, 0.50]
    return random.choices(choices, weights=weights, k=1)[0]


def format_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.10g}"


def prepare_linear_expr(expr: str) -> str:
    expr = expr.lower()
    expr = expr.replace("^", "**")
    expr = expr.replace("\u2212", "-")
    expr = expr.replace("\u00d7", "*")
    expr = expr.replace("\u00f7", "/")
    expr = expr.replace(" ", "")
    expr = re.sub(r"(\d)(x)", r"\1*\2", expr)
    expr = re.sub(r"(x)(\d)", r"\1*\2", expr)
    expr = re.sub(r"(\))(\d|x)", r"\1*\2", expr)
    expr = re.sub(r"(\d|x)(\()", r"\1*\2", expr)
    return expr


def eval_linear_expr(expr: str, x_value: float) -> float | None:
    expr = prepare_linear_expr(expr)
    if not re.fullmatch(r"[0-9x+\-*/().]+", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {"x": x_value}))
    except Exception:
        return None


def expressions_equivalent_in_x(expr_a: str, expr_b: str) -> bool | None:
    # Compare expression values at multiple points.
    # Returns:
    # - True if equivalent
    # - False if confidently different
    # - None if either expression cannot be evaluated
    test_points = [1.0, 2.0, 3.0]
    for x_val in test_points:
        va = eval_linear_expr(expr_a, x_val)
        vb = eval_linear_expr(expr_b, x_val)
        if va is None or vb is None:
            return None
        if not math.isclose(va, vb, rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def solve_linear_equation_from_question(question: str) -> float | None:
    matches = re.findall(r"([0-9xX+\-*/().\s]+)=([0-9xX+\-*/().\s]+)", question)
    if not matches:
        return None

    left_raw, right_raw = matches[0]
    f0_left = eval_linear_expr(left_raw, 0.0)
    f1_left = eval_linear_expr(left_raw, 1.0)
    f0_right = eval_linear_expr(right_raw, 0.0)
    f1_right = eval_linear_expr(right_raw, 1.0)
    if None in (f0_left, f1_left, f0_right, f1_right):
        return None

    slope = (f1_left - f0_left) - (f1_right - f0_right)
    intercept = f0_left - f0_right
    if math.isclose(slope, 0.0, rel_tol=1e-9, abs_tol=1e-9):
        return None
    return -intercept / slope


def canonical_entity_token(text: str) -> str:
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())
    words = [w for w in cleaned.split() if w not in {"the", "number", "of", "in", "a", "an", "club", "box", "basket", "to"}]
    if not words:
        return ""
    token = words[0]
    return token[:-1] if token.endswith("s") else token


def deterministic_answer_from_question(question: str) -> str | None:
    q = question.lower()

    # 1) Solve linear equations containing x.
    if "solve" in q and "x" in q and "=" in q:
        root = solve_linear_equation_from_question(question)
        if root is not None:
            return format_number(root)

    # 2) Fraction-of form: "2/3 of 45"
    m = re.search(r"([0-9]+)\s*/\s*([0-9]+)\s+of\s+([0-9]+(?:\.[0-9]+)?)", q)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        c = float(m.group(3))
        if not math.isclose(b, 0.0):
            return format_number((a / b) * c)

    # 3) Direct arithmetic prompt.
    m = re.search(r"(?:evaluate|calculate)[^:]*:\s*([0-9+\-*/().\s]+)", q)
    if m:
        expr = m.group(1).strip()
        if re.fullmatch(r"[0-9+\-*/().\s]+", expr):
            try:
                return format_number(float(eval(expr, {"__builtins__": {}}, {})))
            except Exception:
                pass

    # 4) Ratio apples:oranges with difference, ask total.
    m = re.search(
        r"ratio.*?(\d+)\s*:\s*(\d+).*?(\d+)\s+more\s+oranges?\s+than\s+apples?.*?(?:total|pieces of fruit|fruits)",
        q,
    )
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        diff = float(m.group(3))
        if not math.isclose(b - a, 0.0):
            unit = diff / (b - a)
            return format_number((a + b) * unit)

    # 5) Ratio apples:oranges with known oranges, ask apples.
    m = re.search(r"ratio.*?(\d+)\s*:\s*(\d+).*?there\s+(?:are|is)\s+(\d+(?:\.\d+)?)\s+oranges?.*?how many apples", q)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        oranges = float(m.group(3))
        if not math.isclose(b, 0.0):
            return format_number(oranges * a / b)

    # 6) Discount, optional sales tax.
    if "discount" in q and "cost" in q:
        price_m = re.search(r"costs?\s*\$?\s*(\d+(?:\.\d+)?)", q)
        disc_m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*discount", q)
        if price_m and disc_m:
            base = float(price_m.group(1))
            disc = float(disc_m.group(1)) / 100.0
            discounted = base * (1.0 - disc)
            tax_m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:sales\s*)?tax", q)
            if tax_m:
                discounted *= 1.0 + (float(tax_m.group(1)) / 100.0)
            return format_number(discounted)

    # 6b) Cost price with profit, then discount on marked selling price.
    m_cost = re.search(r"bought.*?for\s*\$?\s*(\d+(?:\.\d+)?)", q)
    m_profit = re.search(r"profit of\s*(\d+(?:\.\d+)?)\s*%", q)
    m_discount = re.search(r"(\d+(?:\.\d+)?)\s*%\s*discount", q)
    if m_cost and m_profit and m_discount:
        cost = float(m_cost.group(1))
        profit = float(m_profit.group(1)) / 100.0
        discount = float(m_discount.group(1)) / 100.0
        marked = cost * (1.0 + profit)
        final_price = marked * (1.0 - discount)
        return format_number(final_price)

    # 7) Triangle angle ratio, ask largest angle.
    m = re.search(r"ratio\s*(\d+)\s*:\s*(\d+)\s*:\s*(\d+).*?largest angle", q)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        c = float(m.group(3))
        s = a + b + c
        if not math.isclose(s, 0.0):
            return format_number((max(a, b, c) / s) * 180.0)

    # 8) Triangle where A=2B and C is k more/larger/greater than B.
    if "angle a is twice" in q and "angle c is" in q and ("more than angle b" in q or "larger than angle b" in q or "greater than angle b" in q):
        m = re.search(r"angle c is\s*(\d+(?:\.\d+)?)\s*(?:degrees?|\\u00b0)?\s*(?:more|larger|greater)\s+than angle b", q)
        if m:
            inc = float(m.group(1))
            b = (180.0 - inc) / 4.0
            a = 2.0 * b
            c = b + inc
            if re.search(r"find.*angle a|value of angle a|size of angle a", q):
                return format_number(a)
            if re.search(r"find.*angle c|value of angle c|size of angle c", q):
                return format_number(c)
            return format_number(b)

    # 9) Regular polygon interior angle -> sides.
    m = re.search(r"regular polygon.*?interior angle.*?(\d+(?:\.\d+)?)", q)
    if m and "how many sides" in q:
        interior = float(m.group(1))
        denom = 180.0 - interior
        if not math.isclose(denom, 0.0):
            return format_number(360.0 / denom)

    # 10) Ratio a:b, after N more of first/second, ratio becomes c:d.
    m = re.search(
        r"ratio of (.+?) to (.+?) .*? is\s*(\d+)\s*:\s*(\d+).*?"
        r"after\s*(\d+(?:\.\d+)?)\s*more\s+(.+?)\s+(?:join|joins|are added|is added|added).*?"
        r"ratio .*? becomes\s*(\d+)\s*:\s*(\d+).*?how many\s+(.+?)[\?\.]",
        q,
    )
    if m:
        first_raw, second_raw = m.group(1), m.group(2)
        a = float(m.group(3))
        b = float(m.group(4))
        n_added = float(m.group(5))
        added_raw = m.group(6)
        c_new = float(m.group(7))
        d_new = float(m.group(8))
        asked_raw = m.group(9)

        t_first = canonical_entity_token(first_raw)
        t_second = canonical_entity_token(second_raw)
        t_added = canonical_entity_token(added_raw)
        t_asked = canonical_entity_token(asked_raw)

        k = None
        if t_added == t_first:
            denom = (b * c_new) - (a * d_new)
            if not math.isclose(denom, 0.0):
                k = (n_added * d_new) / denom
        elif t_added == t_second:
            denom = (a * d_new) - (b * c_new)
            if not math.isclose(denom, 0.0):
                k = (n_added * c_new) / denom

        if k is not None:
            first_count = a * k
            second_count = b * k
            if t_asked == t_first:
                return format_number(first_count)
            if t_asked == t_second:
                return format_number(second_count)
            if "total" in asked_raw or "pieces of fruit" in asked_raw or "fruits" in asked_raw:
                return format_number(first_count + second_count)

    # 11) Sum of two numbers + twice smaller minus larger.
    m_sum = re.search(r"sum of two numbers is\s*(\d+(?:\.\d+)?)", q)
    m_rel = re.search(r"twice the smaller number,? the result is\s*(\d+(?:\.\d+)?)", q)
    if m_sum and m_rel and "larger number is subtracted" in q and "value of the larger number" in q:
        total = float(m_sum.group(1))
        result = float(m_rel.group(1))
        small = (total + result) / 3.0
        large = total - small
        return format_number(large)

    # 11b) Given x^2 - y^2 and x - y, find x + y and/or x^2 + y^2.
    m1 = re.search(r"x\^2\s*-\s*y\^2\s*=\s*(-?\d+(?:\.\d+)?)", q)
    m2 = re.search(r"x\s*-\s*y\s*=\s*(-?\d+(?:\.\d+)?)", q)
    if m1 and m2:
        diff_sq = float(m1.group(1))
        diff_xy = float(m2.group(1))
        if not math.isclose(diff_xy, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            sum_xy = diff_sq / diff_xy
            x_val = (sum_xy + diff_xy) / 2.0
            y_val = (sum_xy - diff_xy) / 2.0
            if "x^2 + y^2" in q:
                return format_number((x_val * x_val) + (y_val * y_val))
            if "x + y" in q:
                return format_number(sum_xy)

    # 11c) Simplify (x/a) +/- (x/b) ... to a single fraction.
    if "single fraction" in q and "x /" in q:
        terms = re.findall(r"([+\-]?)\s*\(?\s*x\s*/\s*(\d+(?:\.\d+)?)\s*\)?", q)
        if terms:
            coeff = 0.0
            for sign, denom_text in terms:
                denom = float(denom_text)
                if math.isclose(denom, 0.0, rel_tol=1e-9, abs_tol=1e-9):
                    return None
                term_coeff = 1.0 / denom
                if sign == "-":
                    coeff -= term_coeff
                else:
                    coeff += term_coeff

            frac = Fraction(coeff).limit_denominator(360)
            num, den = frac.numerator, frac.denominator
            if den == 1:
                return f"{num}x"
            return f"{num}x/{den}"

    # 12) Evaluate expression at x=value.
    m_x = re.search(r"when\s*x\s*=\s*(-?\d+(?:\.\d+)?)", q)
    m_expr = re.search(r"expression\s*[: ]\s*([0-9xX+\-*/^().\s]+)", question)
    if m_x and m_expr:
        x_val = float(m_x.group(1))
        expr_val = eval_linear_expr(m_expr.group(1), x_val)
        if expr_val is not None:
            return format_number(expr_val)

    # 13) Rectangle perimeter with length more than width, ask area.
    m = re.search(r"rectangle.*?perimeter.*?(\d+(?:\.\d+)?)\s*cm.*?length.*?(\d+(?:\.\d+)?)\s*cm\s*more than.*?width.*?area", q)
    if m:
        p = float(m.group(1))
        d = float(m.group(2))
        w = (p - 2.0 * d) / 4.0
        l = w + d
        return format_number(l * w)

    return None


# ====================== 4. AGENT TASKS ======================
def generate_math_question(
    score: float,
    round_no: int,
    target_difficulty: str,
    recent_questions: List[str],
    recent_topics: List[str],
) -> dict:
    if target_difficulty == "easy":
        difficulty_hint = "Set difficulty to easy."
        topic_hint = "Prefer integers, fractions, decimals, percentages, and simple one-variable equations."
    elif target_difficulty == "hard":
        difficulty_hint = "Set difficulty to hard."
        topic_hint = "Prefer multi-step Secondary 2 items: algebra, expansion/factorisation, and geometry reasoning."
    else:
        difficulty_hint = "Set difficulty to medium."
        topic_hint = "Prefer ratio, percentages, linear equations, and 2-step geometry questions."

    recent_questions_text = " | ".join(recent_questions[-RECENT_MEMORY:]) if recent_questions else "none"
    recent_topics_text = ", ".join(recent_topics[-RECENT_MEMORY:]) if recent_topics else "none"

    prompt = f"""
You are an Elo Math Quiz Agent.
Generate exactly one Secondary 2 math question and provide the answer key.

Round: {round_no}
User Elo score: {score:.1f}
Required difficulty: {target_difficulty}
Difficulty guidance: {difficulty_hint}
Topic guidance: {topic_hint}

Recent questions to avoid repeating:
{recent_questions_text}

Recent topics to avoid repeating:
{recent_topics_text}

Return only valid JSON with this exact schema:
{{
  "intro": "short encouraging line referencing their current score of {score:.1f}",
  "difficulty": "easy or medium or hard",
  "topic": "one short topic label like linear equations or percentages",
  "question": "math question text",
  "answer_key": "short final answer"
}}

Rules:
- Question must match Secondary 2 capabilities.
- Allowed topics: integers, fractions, decimals, percentages, ratio, simple algebra, linear equations,
  expansion/factorisation at basic level, basic geometry (angles, perimeter, area), and simple statistics.
- Not allowed: calculus, trigonometric identities, logarithms, matrices, probability trees, or advanced algebra.
- Must be solvable without a calculator (mental math or short written steps).
- Use clean numbers to keep arithmetic manageable by hand.
- Question must have a short numeric or symbolic final answer.
- Use plain-text math only (no LaTeX like \\frac or $...$).
- Keep question concise and clear.
- Difficulty must be exactly: {target_difficulty}
- Do not reuse the same wording or same topic as recent rounds.
""".strip()

    response = chat_model.invoke(prompt)
    return extract_json_object(response.content)


def validate_answer_key(question: str, answer_key: str) -> dict:
    deterministic = deterministic_answer_from_question(question)
    if deterministic is not None:
        proposed_num = parse_numeric_value(answer_key)
        deterministic_num = parse_numeric_value(deterministic)
        if proposed_num is not None and deterministic_num is not None:
            valid = math.isclose(proposed_num, deterministic_num, rel_tol=1e-9, abs_tol=1e-9)
        else:
            valid = normalize_math_text(answer_key) == normalize_math_text(deterministic)
        return {
            "my_answer": deterministic,
            "is_valid": valid,
            "correct_answer": deterministic,
            "reasoning": "Deterministic rule-based solver used.",
            "deterministic": True,
        }

    prompt = f"""
You are a Secondary 2 math teacher. Solve this question yourself with full working,
then check if the proposed answer key is correct.

Question: {question}
Proposed answer key: {answer_key}

Return only valid JSON with this exact schema:
{{
  "my_answer": "your independently computed answer",
  "is_valid": true or false,
  "correct_answer": "your computed answer",
  "reasoning": "one short paragraph"
}}

Rules:
- Solve independently.
- If your answer differs from proposed key, is_valid must be false.
- correct_answer must be your computed answer.
""".strip()

    try:
        response = chat_model.invoke(prompt)
        return extract_json_object(response.content)
    except Exception:
        return {
            "my_answer": answer_key,
            "is_valid": False,
            "correct_answer": answer_key,
            "reasoning": "Validation unavailable.",
            "uncertain": True,
        }


def verify_user_answer(question: str, answer_key: str, user_answer: str) -> dict:
    deterministic = deterministic_answer_from_question(question)
    expected = (deterministic if deterministic is not None else answer_key).strip()
    ua = user_answer.strip()

    if normalize_math_text(ua) == normalize_math_text(expected):
        return {"is_correct": True, "status": "correct", "expected_answer": expected, "feedback": "Correct."}

    eq_user = parse_linear_equation_value(ua)
    eq_key = parse_linear_equation_value(expected)
    if eq_user and eq_key and eq_user[0] == eq_key[0]:
        if math.isclose(eq_user[1], eq_key[1], rel_tol=1e-9, abs_tol=1e-9):
            return {"is_correct": True, "status": "correct", "expected_answer": expected, "feedback": "Correct. Equivalent equation value."}
        return {"is_correct": False, "status": "incorrect", "expected_answer": expected, "feedback": "Incorrect equation value."}

    num_user = parse_numeric_value(ua)
    num_key = parse_numeric_value(expected)
    if num_user is not None and num_key is not None:
        if math.isclose(num_user, num_key, rel_tol=1e-9, abs_tol=1e-9):
            return {"is_correct": True, "status": "correct", "expected_answer": expected, "feedback": "Correct. Numerically equivalent answer."}
        return {"is_correct": False, "status": "incorrect", "expected_answer": expected, "feedback": "Incorrect numerical value."}

    # If deterministic expected is numeric but user answer is non-numeric,
    # treat as incorrect instead of uncertain.
    if deterministic is not None and num_key is not None and num_user is None:
        return {"is_correct": False, "status": "incorrect", "expected_answer": expected, "feedback": "Expected a numeric final answer."}

    # Symbolic in x (e.g. 5x/12 vs 7x/12, or equivalent expanded/factorised forms).
    if "x" in normalize_math_text(ua) and "x" in normalize_math_text(expected):
        eq_symbolic = expressions_equivalent_in_x(ua, expected)
        if eq_symbolic is True:
            return {"is_correct": True, "status": "correct", "expected_answer": expected, "feedback": "Correct. Equivalent algebraic expression."}
        if eq_symbolic is False:
            return {"is_correct": False, "status": "incorrect", "expected_answer": expected, "feedback": "Incorrect algebraic expression."}

    prompt = f"""
You are a strict Secondary 2 math checker.
Solve the question yourself first, then decide if the user's answer is correct.

Question: {question}
Official answer key: {answer_key}
User answer: {ua}

Return only valid JSON with this exact schema:
{{
  "my_answer": "your independently computed answer",
  "is_correct": true or false,
  "expected_answer": "your computed answer",
  "feedback": "one short sentence"
}}

Rules:
- Solve first.
- Treat equivalent forms as correct.
- Do not blindly trust the official answer key.
""".strip()

    try:
        judged = extract_json_object(chat_model.invoke(prompt).content)
        if bool(judged.get("is_correct", False)):
            return {
                "is_correct": True,
                "status": "correct",
                "expected_answer": str(judged.get("expected_answer", expected)).strip(),
                "feedback": str(judged.get("feedback", "Correct.")).strip(),
            }
        return {
            "is_correct": False,
            "status": "uncertain",
            "expected_answer": expected,
            "feedback": "Uncertain grading. No Elo penalty applied.",
        }
    except Exception:
        return {
            "is_correct": False,
            "status": "uncertain",
            "expected_answer": expected,
            "feedback": "Could not confidently verify this answer. No Elo penalty applied.",
        }


# ====================== 5. INTERACTIVE ELO SESSION ======================
print("=" * 55)
print("        ELO MATH QUIZ - Powered by LangChain")
print("=" * 55)
print("Answer Secondary 2 math questions to grow your Elo rating.")
print("All questions are no-calculator friendly.")
print("Correct answers raise it. Wrong answers lower it.")
print("Type 'exit' or 'quit' at any time to end.\n")

score = DEFAULT_USER_ELO
round_no = 1
recent_question_texts: List[str] = []
recent_topics: List[str] = []
recent_signatures: Set[str] = set()

print(f"Starting Elo: {score:.1f}\n")

while True:
    print(f"--- Round {round_no} {'-' * 38}")

    try:
        selected_difficulty = choose_difficulty_for_score(score)
        generated = None
        candidate = None

        for _ in range(3):
            candidate = generate_math_question(
                score=score,
                round_no=round_no,
                target_difficulty=selected_difficulty,
                recent_questions=recent_question_texts,
                recent_topics=recent_topics,
            )
            candidate_question = str(candidate.get("question", "")).strip()
            candidate_answer = str(candidate.get("answer_key", "")).strip()

            if not candidate_question or is_repetitive(candidate_question, recent_signatures):
                continue

            validation = validate_answer_key(candidate_question, candidate_answer)
            if not validation.get("is_valid", True):
                corrected = str(validation.get("correct_answer", candidate_answer)).strip()
                candidate["answer_key"] = corrected

            generated = candidate
            break

        if generated is None:
            generated = candidate or {
                "topic": "linear equations",
                "question": "Solve 3x + 5 = 20.",
                "answer_key": "5",
                "intro": f"Your score is {score:.1f}. Here is a fallback question.",
            }

        difficulty = selected_difficulty
        topic = str(generated.get("topic", "secondary 2 math")).strip().lower()
        question = str(generated.get("question", "Solve 3x + 5 = 20.")).strip()
        answer_key = str(generated.get("answer_key", "5")).strip()
        intro = str(generated.get("intro", f"Your score is {score:.1f}. Here is your next question.")).strip()

    except Exception:
        difficulty = "medium"
        topic = "linear equations"
        question = "Solve 3x + 5 = 20."
        answer_key = "5"
        intro = f"Your score is {score:.1f}. Here is a fallback question."

    opp_rating = difficulty_to_rating(difficulty)

    print(f"Agent : {intro}")
    print(f"Topic : {topic}")
    print(f"Q ({difficulty.upper()}, question rating: {opp_rating}): {question}")

    user_input = input("Your answer: ").strip()
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\nSession ended early.")
        break

    try:
        judged = verify_user_answer(question=question, answer_key=answer_key, user_answer=user_input)
        status = str(judged.get("status", "incorrect")).lower()
        expected = str(judged.get("expected_answer", answer_key)).strip()
        feedback = str(judged.get("feedback", "Checked.")).strip()
    except Exception:
        status = "uncertain"
        expected = answer_key
        feedback = "Could not confidently verify this answer. No Elo penalty applied."

    old_score = score

    if status == "correct":
        score = update_elo(score, opp_rating, 1.0, K_FACTOR)
    elif status == "incorrect":
        score = update_elo(score, opp_rating, 0.0, K_FACTOR)
    else:
        status = "uncertain"

    delta = score - old_score

    recent_question_texts.append(question)
    recent_topics.append(topic)
    recent_question_texts = recent_question_texts[-RECENT_MEMORY:]
    recent_topics = recent_topics[-RECENT_MEMORY:]
    recent_signatures = {question_signature(q) for q in recent_question_texts}

    if status == "correct":
        result_label = "[CORRECT]"
    elif status == "incorrect":
        result_label = f"[INCORRECT] (expected: {expected})"
    else:
        result_label = "[UNCERTAIN] (no Elo change)"

    print(f"Result  : {result_label}")
    print(f"Feedback: {feedback}")
    print(f"Elo     : {old_score:.1f} -> {score:.1f} ({delta:+.1f})")

    round_no += 1
    print()

print("=" * 55)
print(f"  Final Elo Score : {score:.1f}  (started at {DEFAULT_USER_ELO:.1f})")
print(f"  Rounds played   : {round_no - 1}")
print("=" * 55)

