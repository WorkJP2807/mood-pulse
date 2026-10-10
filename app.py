"""
AI Manager Mood Pulse Dashboard - Flask backend
------------------------------------------------
Setup:
    pip install flask google-genai

Run:
    python app.py
Then open http://127.0.0.1:8080
"""

import json
import os
import re
import traceback

from flask import Flask, jsonify, request, send_file
from google import genai
from google.genai import types

# =============================================================================
# API KEY
# On Vercel:  set an Environment Variable named GEMINI_KEY (Project Settings).
# Locally:    either set the same env var, or replace the placeholder below.
# Never commit a real key to GitHub.
# =============================================================================
GEMINI_KEY = os.environ.get("GEMINI_KEY", "AIzaSyYourActualKeyHere")

client = genai.Client(api_key=GEMINI_KEY)

MODEL_NAME = "gemini-2.5-flash"
MAX_INPUT_CHARS = 5000
ALLOWED_COLORS = ("emerald", "indigo", "amber", "rose")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# -----------------------------------------------------------------------------
# System instruction: forensic linguistic + psychological reading of a message
# -----------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are an elite organizational psychologist and forensic linguist embedded in an
enterprise "Manager Mood Pulse" platform. You receive a message written by a
manager (email, chat, meeting note, ticket comment, etc.) together with a block of
pre-computed linguistic signals. Your job is to read the emotional state, intent and
psychological temperature behind the message, then coach the employee on how to
pitch their next request.

ANALYSIS METHOD - weigh ALL of the following, and look beneath the literal words:
1. Word length and sentence rhythm: very short, clipped sentences and one-word
   replies ("Fine." "Noted." "Ok.") signal withdrawal, irritation or control. Long,
   warm, elaborated sentences usually signal openness and psychological safety.
   Compare average word length and average sentence length against the message
   length - a short message that is also curt is different from a short message
   that is simply efficient and friendly.
2. Capitalization: ALL-CAPS words, shouted emphasis, or abrupt Capitalized Demands
   signal escalation or urgency. Consistent sentence-case with normal greetings
   signals composure. Absence of any capitalization can signal fatigue or
   disengagement.
3. Aggressive punctuation density: stacked or trailing punctuation such as "...",
   "?!", "!!", "??" indicates suppressed frustration, incredulity, or sarcasm.
   Ellipses used as trailing-off often hide resentment. Heavy exclamation marks can
   be either genuine enthusiasm or hostility - decide from the surrounding wording.
4. Passive-aggressive keywords and constructions: "per my last email", "as
   previously discussed", "going forward", "friendly reminder", "noted", "fine",
   "whatever", "no worries" (when contradicting the context), "kindly", "I guess",
   "obviously", "clearly", "again", "apparently", "with all due respect",
   "thanks in advance" used as a closing command. The pre-computed keyword hits are
   only SIGNALS, never a verdict: judge them in context, since some phrases are
   perfectly neutral in a friendly message.
5. Pragmatics: blame framing, deadline pressure, withheld pleasantries, missing
   greetings or sign-offs, rhetorical questions, sudden formality shifts,
   praise-then-criticism sandwiches, and implied consequences.

SCORING:
- "score" is an integer from 0 to 100 describing how emotionally positive, open and
  approachable the manager currently is. 100 means warm, relaxed and receptive.
  0 means hostile, volatile or shut down.
- Colour bands (choose "color" to match the score):
    75-100  -> "emerald"  (receptive, good moment to pitch)
    50-74   -> "indigo"   (neutral, businesslike, stable)
    30-49   -> "amber"    (tense, guarded, passive-aggressive undertones)
    0-29    -> "rose"     (hostile or highly stressed, high risk)
- "label" is a short 1-3 word mood name (for example "Warm & Receptive",
  "Guarded", "Quietly Irritated", "Escalating").

OUTPUT FIELDS:
- "pitchHeading": a punchy headline of at most 8 words naming the best pitching
  strategy for the employee right now.
- "pitchText": 2 to 4 sentences of concrete, practical advice on timing, framing,
  tone and what to lead with when the employee brings their next request. Be
  specific to the evidence in this message; avoid generic platitudes.
- "keyIndicators": 3 to 5 short strings (each under 70 characters) naming the
  concrete evidence you used, e.g. "Ellipsis density: 3 (elevated)", "Passive-
  aggressive phrase: 'per my last email'", "Terse sentences: avg 4.2 words".

Be nuanced and calibrated. Do not over-diagnose neutral or friendly messages, and do
not under-diagnose subtle hostility. Respond ONLY with the JSON object that matches
the required schema. No markdown, no commentary.
""".strip()

# -----------------------------------------------------------------------------
# Strict response schema enforced on the model
# -----------------------------------------------------------------------------
RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "score": types.Schema(type=types.Type.INTEGER),
        "label": types.Schema(type=types.Type.STRING),
        "color": types.Schema(type=types.Type.STRING, enum=list(ALLOWED_COLORS)),
        "pitchHeading": types.Schema(type=types.Type.STRING),
        "pitchText": types.Schema(type=types.Type.STRING),
        "keyIndicators": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
        ),
    },
    required=["score", "label", "color", "pitchHeading", "pitchText", "keyIndicators"],
)

# -----------------------------------------------------------------------------
# Local linguistic signal extraction (fed to the model as hard evidence)
# -----------------------------------------------------------------------------
PASSIVE_AGGRESSIVE_PATTERNS = [
    r"per my last (?:e-?mail|message)",
    r"as (?:previously )?(?:mentioned|discussed|stated|noted)",
    r"as i (?:said|mentioned)",
    r"just (?:circling|following) (?:back|up)",
    r"friendly reminder",
    r"no worries",
    r"\bfine\b",
    r"\bwhatever\b",
    r"\bnoted\b",
    r"going forward",
    r"with all due respect",
    r"\bkindly\b",
    r"thanks in advance",
    r"not sure why",
    r"\bi guess\b",
    r"if you (?:have|get) (?:the )?time",
    r"\bobviously\b",
    r"\bclearly\b",
    r"\bagain\b",
    r"\bapparently\b",
    r"as expected",
]


def extract_signals(text: str) -> dict:
    """Compute measurable linguistic features of the raw message."""
    words = re.findall(r"[\w'\u2019-]+", text)
    word_count = len(words)
    avg_word_len = (
        round(sum(len(w) for w in words) / word_count, 2) if word_count else 0
    )

    letters = [c for c in text if c.isalpha()]
    upper_letters = sum(1 for c in letters if c.isupper())
    caps_ratio = round(100 * upper_letters / len(letters), 1) if letters else 0
    caps_words = len([w for w in words if len(w) >= 2 and w.isalpha() and w.isupper()])

    ellipses = len(re.findall(r"\.{2,}|\u2026", text))
    interrobangs = len(re.findall(r"[?!]{2,}", text))
    exclamations = text.count("!")
    questions = text.count("?")

    sentences = [s for s in re.split(r"[.!?]+(?:\s|$)", text) if s.strip()]
    avg_sentence_len = round(word_count / len(sentences), 1) if sentences else 0

    lowered = text.lower()
    pa_hits = []
    for pattern in PASSIVE_AGGRESSIVE_PATTERNS:
        for match in re.finditer(pattern, lowered):
            pa_hits.append(match.group(0))

    return {
        "characters": len(text),
        "words": word_count,
        "avgWordLength": avg_word_len,
        "sentences": len(sentences),
        "avgSentenceLength": avg_sentence_len,
        "capsRatioPercent": caps_ratio,
        "capsWords": caps_words,
        "ellipses": ellipses,
        "interrobangs": interrobangs,
        "exclamations": exclamations,
        "questions": questions,
        "passiveAggressiveHits": pa_hits,
    }


def build_user_prompt(text: str, source: str, signals: dict) -> str:
    return (
        f"PIPELINE SOURCE: {source}\n\n"
        f"PRE-COMPUTED LINGUISTIC SIGNALS (JSON):\n"
        f"{json.dumps(signals, indent=2)}\n\n"
        f"MANAGER MESSAGE:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Analyze this message and return the JSON object."
    )


def parse_model_json(raw: str) -> dict:
    """Parse model output defensively (handles stray code fences)."""
    if not raw:
        raise ValueError("The model returned an empty response.")
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def color_for_score(score: int) -> str:
    if score >= 75:
        return "emerald"
    if score >= 50:
        return "indigo"
    if score >= 30:
        return "amber"
    return "rose"


def normalize_result(data: dict) -> dict:
    """Guarantee the response matches the strict schema the frontend expects."""
    try:
        score = int(round(float(data.get("score", 50))))
    except (TypeError, ValueError):
        score = 50
    score = max(0, min(100, score))

    color = str(data.get("color", "")).lower().strip()
    if color not in ALLOWED_COLORS:
        color = color_for_score(score)

    indicators = data.get("keyIndicators", [])
    if not isinstance(indicators, list):
        indicators = [str(indicators)]
    indicators = [str(i).strip() for i in indicators if str(i).strip()][:6]

    return {
        "score": score,
        "label": str(data.get("label", "Unclassified")).strip() or "Unclassified",
        "color": color,
        "pitchHeading": str(data.get("pitchHeading", "Proceed with care")).strip(),
        "pitchText": str(data.get("pitchText", "")).strip(),
        "keyIndicators": indicators,
    }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    candidates = (
        os.path.join(BASE_DIR, "templates", "index.html"),
        os.path.join(BASE_DIR, "index.html"),
    )
    for path in candidates:
        if os.path.exists(path):
            return send_file(path)
    looked = "<br>".join(candidates)
    return f"index.html not found. Looked in:<br>{looked}", 404


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    source = str(payload.get("source", "Direct Input")).strip()[:60] or "Direct Input"

    if not text:
        return jsonify({"error": "Paste a manager message before running the analysis."}), 400
    if len(text) > MAX_INPUT_CHARS:
        return (
            jsonify({"error": f"Message is too long. Limit is {MAX_INPUT_CHARS} characters."}),
            400,
        )
    if GEMINI_KEY.startswith("AIzaSyYourActualKey"):
        return (
            jsonify({"error": "Gemini API key not set. Add a GEMINI_KEY environment variable (Vercel) or edit app.py locally."}),
            500,
        )

    signals = extract_signals(text)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=build_user_prompt(text, source, signals),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.4,
            ),
        )
        raw = response.text
        result = normalize_result(parse_model_json(raw))
        result["signals"] = signals
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001 - surface any SDK/network error to the UI
        traceback.print_exc()
        message = str(exc).strip() or exc.__class__.__name__
        return jsonify({"error": f"Analysis failed: {message[:300]}"}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True, use_reloader=False)
