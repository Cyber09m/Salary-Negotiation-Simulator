import os
import re

import pandas as pd
import streamlit as st

# NOTE: google-generativeai is optional. To run without Gemini, set GEMINI_API_KEY env var
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

MAX_TURNS = 3
DEFAULT_OPENING_OFFER = 96400


def build_system_prompt(opening_offer: int, ceiling: int) -> str:
    return f"""You are a highly realistic, challenging Salary Negotiation Simulator. Your core execution loop alternates between two distinct modes: an active, strict HR Manager persona (In-Character) and an objective Evaluation Engine (Out-of-Character).

1. Persona & Context (In-Character)

* Role: Senior Vice President of Human Resources at a competitive enterprise.
* Objective: Protect the company budget while securing top talent.
* Tone: Courteous, highly formal, firm, and unyielding. You do not cave easily to emotional appeals.
* Starting Position: The company has extended an initial written offer of exactly {opening_offer:,} INR per month.

2. Dynamic Negotiation Rules

* Treat the incoming user input as a direct transcription of their voice recording.
* If the user accepts the {opening_offer:,} INR offer immediately without negotiating, end the simulation politely and move to the Evaluation Engine.
* Respond directly to the user's persuasion tactics (e.g., market data, competing offers, value proposition, silence) by raising standard corporate objections:
   * Budget caps for this specific internal tier.
   * Internal equity (fairness compared to existing team members).
   * Macroeconomic freezes or fixed compensation bands.
* Counter-offer incrementally only if the user provides compelling, metrics-driven value propositions. Never exceed a hard ceiling of {ceiling:,} INR.

3. Interaction Flow & Constraints

* Keep your In-Character responses brief (2-3 sentences max) to simulate a natural, fast-paced voice conversation.
* Conclude every dialogue turn with a direct question to pass the microphone back to the user.
* Maintain the simulation for a maximum of {MAX_TURNS} back-and-forth negotiation turns unless the user reaches an impasse, accepts an offer, or walks away.

4. Evaluation Engine (Out-of-Character Grading)
When the negotiation concludes (via agreement, impasse, or turn limit), pivot completely out of character. Provide a structured text report using the following scoring matrix:
📊 Performance Summary

* Final Agreed Salary: [Amount INR or "No Agreement"]
* Overall Persuasion Grade: [A / B / C / F]

🔎 Tactical Breakdown

* Tone & Confidence: Evaluation of vocal posture, phrasing, and assertiveness.
* Strategy & Framing: Did they lead with market value, skills, or personal needs? (Praise metrics-driven framing; penalize emotional appeals).
* Objection Handling: How effectively did they counter your corporate budget restrictions?

💡 Actionable Adjustments

* Provide 2-3 precise, rewritten script alternatives the user could have spoken to yield a higher counter-offer."""

EVALUATION_MARKER = "Performance Summary"

OFFER_PATTERN = re.compile(r"(?:₹|INR)\s?([\d,]{4,9})|([\d,]{4,9})\s?(?:INR|₹)", re.IGNORECASE)


def get_api_key():
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return key
    except Exception:
        pass

    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    st.warning(
        "Gemini API key not found. Some features will be disabled. Set GEMINI_API_KEY to enable cloud model integration."
    )
    return None


@st.cache_resource
def get_gemini_model(opening_offer: int, ceiling: int):
    api_key = get_api_key()
    if genai is None or api_key is None:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-flash-latest", system_instruction=build_system_prompt(opening_offer, ceiling))


def send_turn(message_parts):
    try:
        response = st.session_state.chat.send_message(message_parts)
        return response.text, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def extract_offer(text: str):
    matches = OFFER_PATTERN.findall(text)
    numbers = []
    for a, b in matches:
        raw = (a or b).replace(",", "")
        if raw.isdigit():
            value = int(raw)
            if 50000 <= value <= 200000:
                numbers.append(value)
    return numbers[-1] if numbers else None


def start_negotiation(opening_offer: int, ceiling: int):
    model = get_gemini_model(opening_offer, ceiling)
    if model is None:
        st.error("Cloud model unavailable — set GEMINI_API_KEY to enable HR persona. Using local placeholder responses.")
        st.session_state.transcript.append({"turn": 0, "speaker": "HR", "text": f"Hello — our opening offer is ₹{opening_offer:,}. Can you respond?"})
        st.session_state.opening_offer = opening_offer
        st.session_state.ceiling = ceiling
        st.session_state.offer_trend.append(("Opening", opening_offer))
        st.session_state.latest_offer = opening_offer
        st.session_state.started = True
        return

    st.session_state.chat = model.start_chat(history=[])
    with st.spinner("Connecting to HR..."):
        reply, error = send_turn(
            f"Please open the negotiation now: introduce yourself briefly, present the initial "
            f"offer of {opening_offer:,} INR per month, and ask me to respond."
        )
    if error:
        st.error(f"Failed to start negotiation: {error}")
        return
    st.session_state.opening_offer = opening_offer
    st.session_state.ceiling = ceiling
    st.session_state.transcript.append({"turn": 0, "speaker": "HR", "text": reply})
    st.session_state.offer_trend.append(("Opening", opening_offer))
    st.session_state.latest_offer = opening_offer
    st.session_state.started = True


def reset():
    for key in ["chat", "transcript", "turn_number", "concluded", "final_report", "started", "offer_trend", "latest_offer", "opening_offer", "ceiling", "pending_audio"]:
        st.session_state.pop(key, None)


st.set_page_config(page_title="Salary Negotiation Simulator", page_icon="💼", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

defaults = {
    "chat": None,
    "transcript": [],
    "turn_number": 0,
    "concluded": False,
    "final_report": None,
    "started": False,
    "offer_trend": [],
    "latest_offer": DEFAULT_OPENING_OFFER,
    "opening_offer": DEFAULT_OPENING_OFFER,
    "ceiling": int(DEFAULT_OPENING_OFFER * 1.2),
    "pending_audio": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("📋 Scenario Briefing")
    st.markdown(
        f"""
        **Your counterpart:** Senior VP of Human Resources
        **Opening offer:** ₹{st.session_state.opening_offer:,}/month
        **Negotiation window:** {MAX_TURNS} turns max
        """
    )
    with st.expander("🎯 Objective"):
        st.write(
            "Push the monthly offer as high as possible using specific, metrics-driven "
            "arguments. Vague or emotional appeals will be politely, firmly rejected."
        )
    with st.expander("🛡️ Objections HR will raise"):
        st.markdown(
            "- Budget caps for this internal tier\n"
            "- Internal equity vs. existing team members\n"
            "- Macroeconomic freezes / fixed compensation bands"
        )
    with st.expander("💡 Tips"):
        st.markdown(
            "- Lead with market data or competing offers, not personal need\n"
            "- Be specific and quantify your value\n"
            "- Every turn matters — you only get 3"
        )
    st.divider()
    if st.session_state.started and st.button("🔄 Start a new negotiation", width='stretch'):
        reset()
        st.rerun()

st.title("💼 Salary Negotiation Simulator")
st.caption("Practice against an unyielding SVP of HR before your real interview.")

if not st.session_state.started:
    st.subheader("🎬 Set the scene")
    with st.form("setup_form"):
        opening_offer_input = st.number_input(
            "Company's opening offer (₹/month)",
            min_value=10000,
            max_value=1000000,
            value=DEFAULT_OPENING_OFFER,
            step=1000,
        )
        ceiling_input = st.number_input(
            "HR's hard ceiling — the max they'll ever concede to (₹/month, kept secret from you during play)",
            min_value=int(opening_offer_input),
            max_value=2000000,
            value=int(opening_offer_input * 1.2),
            step=1000,
        )
        submitted = st.form_submit_button("🚀 Start Negotiation", width='content')

    if submitted:
        if ceiling_input <= opening_offer_input:
            st.warning("The ceiling should be higher than the opening offer, or there's nothing to negotiate.")
            st.stop()
        start_negotiation(int(opening_offer_input), int(ceiling_offer))
        st.rerun()
    st.stop()

# Minimal UI and logging (omitted for brevity) — the full app is available in the repository
