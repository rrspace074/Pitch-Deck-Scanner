# app.py — Streamlit x Dify "Pitch Audit AI" (single-shot app)
# Run:  streamlit run app.py
# Env vars expected (or via .streamlit/secrets.toml):
#   NEXT_PUBLIC_API_URL   (default: https://api.dify.ai/v1)
#   NEXT_PUBLIC_APP_KEY   (required)
#   NEXT_PUBLIC_APP_ID    (optional)
#   OPENAI_API_KEY        (optional; if set, used to reformat Dify output)
#   DIFY_INPUTS_JSON      (optional; JSON dict merged into inputs for the app)

import os
import json
import base64
import random
import mimetypes
import uuid
import hashlib
from typing import Generator, List, Optional

import requests
import streamlit as st

# For OpenAI post-processing (use requests; no extra deps)
import time

# -----------------------------
# Config & helpers
# -----------------------------

def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    # Prefer Streamlit secrets first, then environment
    try:
        val = st.secrets.get(name)  # type: ignore[attr-defined]
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(name, default)

API_BASE = _get_secret("NEXT_PUBLIC_API_URL", "https://api.dify.ai/v1").rstrip("/")
API_KEY = _get_secret("NEXT_PUBLIC_APP_KEY", "")
APP_ID = _get_secret("NEXT_PUBLIC_APP_ID", "")
OPENAI_KEY = _get_secret("OPENAI_API_KEY", "")

# Stable per-session user id (Dify requires a user identifier)
if "dify_user" not in st.session_state:
    st.session_state.dify_user = f"user-{uuid.uuid4()}"

# Track Dify conversation id
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# Track uploads across reruns to avoid re-uploading same file
st.session_state.setdefault("uploaded_fps", {})   # fp -> {id, type, name}
st.session_state.setdefault("uploaded_payload", [])
st.session_state.setdefault("uploader_key", 0)

# -----------------------------
# Page config + styles
# -----------------------------

st.set_page_config(page_title="Pitch Audit AI • Dify", page_icon="🤖", layout="wide")

# --- Dark theme with gold accents (inspired by reference) ---
DARK_THEME_CSS = """
<style>
  :root{
    --gold: #F5C542;
    --gold-2: #D9A521;
    --bg: #0c0f14;
    --panel: #11151d;
    --text: #EAEFF5;
    --muted: #B7BEC9;
    --muted-2: #96A0AF;
  }
  /* App background */
  div[data-testid="stAppViewContainer"]{
    background:
      radial-gradient(80% 120% at 10% 10%, rgba(245,197,66,0.09) 0%, rgba(0,0,0,0) 55%),
      radial-gradient(60% 100% at 90% 10%, rgba(245,197,66,0.07) 0%, rgba(0,0,0,0) 60%),
      repeating-linear-gradient(135deg, rgba(245,197,66,0.06) 0 2px, rgba(0,0,0,0) 2px 22px),
      var(--bg);
    color: var(--text);
  }
  /* Remove Streamlit default white top header */
  header[data-testid="stHeader"],
  div[data-testid="stHeader"],
  div[data-testid="stToolbar"]{
    display: none !important;
  }
  .block-container{ padding-top: 0 !important; }
  section[data-testid="stSidebar"]{ background:#0b0e12; border-right:1px solid rgba(255,255,255,.06); }

  /* Hero */
  .hero{ margin: 24px 0 22px 0; padding: 64px 24px 48px; text-align: center; position: relative; background:
      radial-gradient(120% 160% at 50% 0%, rgba(245,197,66,0.08) 0%, rgba(20,22,30,0.0) 60%);
      border-radius: 20px;
  }
  .hero h1{ font-size:56px; line-height:1.1; margin:0 0 12px 0; color:var(--text); font-weight:800; }
  .hero h1 .powered { font-size:18px; font-weight:700; color: var(--muted); margin-left:16px; vertical-align: middle; }
  .hero h1 .powered img{ height:56px; width:auto; border-radius:10px; vertical-align:middle; margin-left:8px; }
  .hero h2{ font-size:28px; font-weight:700; margin:0 0 16px 0; opacity:1; display:inline-block; padding:12px 20px; border-radius:14px; background:linear-gradient(180deg, var(--gold) 0%, var(--gold-2) 100%); color:#14161e; box-shadow:0 8px 24px rgba(245,197,66,0.25); }
  .hero p{ color:var(--muted); margin:0 0 26px 0; font-size:16px; }
  .hero .pill{ display:inline-block; padding:14px 22px; border-radius:14px; background:linear-gradient(180deg, var(--gold) 0%, var(--gold-2) 100%); color:#14161e; font-weight:800; box-shadow:0 8px 24px rgba(245,197,66,0.25); margin:10px 0 18px 0; }
  .primary-cta{ background:linear-gradient(180deg,var(--gold) 0%, var(--gold-2) 100%); color:#14161e; font-weight:800; padding:12px 22px; border-radius:12px; border:1px solid rgba(255,255,255,.08); display:inline-block; box-shadow:0 8px 24px rgba(245,197,66,0.25); text-decoration:none; }
  .primary-cta:hover{ filter:brightness(1.03); }
  .small-note{ color:var(--muted-2); font-size:14px; margin-top:10px; }

  /* Buttons */
  .stButton > button[kind="primary"]{
    background: linear-gradient(180deg, var(--gold) 0%, var(--gold-2) 100%) !important;
    color: #14161e !important; border: 0 !important; border-radius: 12px !important; font-weight: 800 !important;
    box-shadow: 0 8px 24px rgba(245,197,66,0.25) !important; padding: 0.6rem 1.5rem !important; max-width: 280px; margin: 0 auto; display:block;
  }
  .stButton > button[kind="secondary"]{
    background:#161a22 !important; color: var(--text) !important; border:1px solid rgba(255,255,255,.08) !important; border-radius:12px !important;
  }

  /* Uploader */
  div[data-testid="stFileUploaderDropzone"],
  div[data-testid="stFileUploaderDropzone"] > div,
  div[data-testid="stFileUploader"] section {
    background:linear-gradient(180deg, var(--gold) 0%, var(--gold-2) 100%) !important;
    color:#14161e !important;
  }
  div[data-testid="stFileUploaderDropzone"] *{ color:#14161e !important; }Limit 15MB
  div[data-testid="stFileUploaderDropzone"] p:nth-of-type(2){ color:transparent !important; position:relative; }
  div[data-testid="stFileUploaderDropzone"] p:nth-of-type(2)::after{ content: "Limit 15MB per file • PDF, PPT, PPTX, DOC, DOCX, TXT"; position:absolute; left:0; right:0; top:0; color:#14161e !important; }
  /* Fallback targeting for versions where the helper is the last <p> */
  div[data-testid="stFileUploaderDropzone"] p:last-of-type{ color:transparent !important; position:relative; }
  div[data-testid="stFileUploaderDropzone"] p:last-of-type::after{ content: "Limit 15MB per file • PDF, PPT, PPTX, DOC, DOCX, TXT"; position:absolute; left:0; right:0; top:0; color:#14161e !important; }
  /* Browse button inside uploader */
  div[data-testid="stFileUploader"] button{ background:#14161e !important; color:var(--text) !important; border:1px solid rgba(20,22,30,.45) !important; border-radius:10px !important; }
  div[data-testid="stFileUploader"] small { display: none !important; }

  /* Result + info */
  .result-box { border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 16px; background: #0f131b; color: var(--text); }
  .tiny { text-align:center; font-size:.85rem; color: var(--muted-2); margin-top: 10px; }
  h1, h2, h3, h4, h5, h6, label, p, .stMarkdown { color: var(--text); }
  .spacer24 { height:24px; }
  .spacer32 { height:32px; }
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

# Load local logo (your_logo.jpeg) as base64 so we can embed in HTML

def _load_logo_b64(path: str = "your_logo.jpeg") -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None

logo_b64 = _load_logo_b64()
logo_img_tag = f'<img style="height:56px;width:auto;border-radius:10px;vertical-align:middle;margin-left:10px;" src="data:image/jpeg;base64,{logo_b64}">' if logo_b64 else ""

# Hero section to match dark/gold theme
st.markdown(
    f"""
    <div class=\"hero\">
        <h1>Pitch Audit AI <span class=\"powered\">powered by {logo_img_tag if logo_img_tag else 'AI'}</span></h1>
        <h2>Create investor‑ready decks</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar — only "Clear conversation"
# -----------------------------
with st.sidebar:
    st.write("")
    if st.button("Clear conversation", type="secondary", use_container_width=True):
        st.session_state.conversation_id = None
        st.session_state.uploaded_fps = {}
        st.session_state.uploaded_payload = []
        st.session_state.pop("last_output", None)
        st.session_state["uploader_key"] += 1  # reset the file_uploader selection
        st.rerun()

    # Info line under the button
    st.markdown("<div style='display:flex;align-items:center;gap:8px;color:#B7BEC9;font-size:0.9rem;'>ℹ️ <span>Reload if it doesn't respond in 60 sec.</span></div>", unsafe_allow_html=True)

# Safety check for keys (required)
if not API_KEY:
    st.error("NEXT_PUBLIC_APP_KEY is required. Set it in your environment or .streamlit/secrets.toml.")
    st.stop()
def _build_inputs() -> dict:
    """Build inputs for Dify app. Merge optional JSON from secrets/env and set fallbacks
    for common variables seen in workflows (structured_output, website, llm)."""
    base: dict = {}
    raw = _get_secret("DIFY_INPUTS_JSON", "")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                base.update(parsed)
        except Exception:
            pass
    base.setdefault("structured_output", False)
    base.setdefault("website", "")
    base.setdefault("llm", "")
    return base

# -----------------------------
# Dify HTTP helpers
# -----------------------------

def _headers(json_mode: bool = True) -> dict:
    h = {"Authorization": f"Bearer {API_KEY}"}
    if json_mode:
        h["Content-Type"] = "application/json"
    if APP_ID:
        h["X-Dify-App-Id"] = APP_ID
    return h



def _infer_file_type(mime: str) -> str:
    if not mime:
        return "document"
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    return "document"




def dify_upload_file(file_name: str, file_bytes: bytes, mime: Optional[str], user: str) -> str:
    """Upload a file to Dify for this conversation; returns upload_file_id."""
    url = f"{API_BASE}/files/upload"
    files = {"file": (file_name, file_bytes, mime or "application/octet-stream")} 
    data = {"purpose": "conversation", "user": user}
    resp = requests.post(url, headers={"Authorization": f"Bearer {API_KEY}"}, files=files, data=data, timeout=600)
    if resp.status_code >= 400:
        raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text}")
    j = resp.json()
    upload_id = j.get("id") or j.get("data", {}).get("id")
    if not upload_id:
        raise RuntimeError(f"Unexpected upload response: {j}")
    return upload_id


def dify_stream_chat(query: str, files_payload: Optional[List[dict]], user: str) -> Generator[str, None, None]:
    payload = {"inputs": _build_inputs(), "query": query, "response_mode": "streaming", "user": user}
    if st.session_state.conversation_id:
        payload["conversation_id"] = st.session_state.conversation_id
    if files_payload:
        payload["files"] = files_payload

    url = f"{API_BASE}/chat-messages"
    with requests.post(url, headers=_headers(json_mode=True), json=payload, stream=True, timeout=1200) as r:
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = r.text
            raise RuntimeError(f"{r.status_code}: {err}")
        final_buffer = []
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or raw.startswith(":"):
                continue
            if not raw.startswith("data:"):
                continue
            data_str = raw[5:].strip()
            if not data_str:
                continue
            try:
                evt = json.loads(data_str)
            except Exception:
                continue
            conv_id = evt.get("conversation_id")
            if conv_id and not st.session_state.conversation_id:
                st.session_state.conversation_id = conv_id
            event_type = evt.get("event")
            # Stream incremental tokens when available
            if event_type in ("message", "agent_message", "tool_message", "message_delta"):
                delta = (evt.get("answer") or evt.get("output_text") or evt.get("data") or "")
                if isinstance(delta, dict):
                    delta = delta.get("text") or delta.get("content") or ""
                if delta:
                    final_buffer.append(str(delta))
                    yield str(delta)
            elif event_type in ("message_end", "agent_message_end", "completed", "workflow_finished"):
                # Some Dify apps send only the final text on *_end events; make sure we surface it.
                tail = (evt.get("answer") or evt.get("output_text") or "")
                if isinstance(tail, dict):
                    tail = tail.get("text") or tail.get("content") or ""
                if tail:
                    final_buffer.append(str(tail))
                    yield str(tail)
                break
            elif event_type == "error":
                err_msg = evt.get("message") or evt.get("error") or "Model error"
                raise RuntimeError(err_msg)


def dify_blocking_chat(query: str, files_payload: Optional[List[dict]], user: str, timeout_sec: int = 1200) -> str:
    """Call Dify with response_mode=blocking and return the final answer text.
    Ensures we wait for long-running workflows and still capture the final output.
    """
    payload = {"inputs": _build_inputs(), "query": query, "response_mode": "blocking", "user": user}
    if st.session_state.conversation_id:
        payload["conversation_id"] = st.session_state.conversation_id
    if files_payload:
        payload["files"] = files_payload

    url = f"{API_BASE}/chat-messages"
    r = requests.post(url, headers=_headers(json_mode=True), json=payload, timeout=timeout_sec)
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"{r.status_code}: {err}")
    j = r.json()
    # Capture conversation id if present
    conv_id = j.get("conversation_id") or j.get("result", {}).get("conversation_id")
    if conv_id and not st.session_state.conversation_id:
        st.session_state.conversation_id = conv_id

    # Dify variants: answer, output_text, result.answer, data.text
    def pick(obj: dict) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        for key in ("answer", "output_text", "text", "content"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return None

    # Try several shapes
    ans = pick(j) or pick(j.get("data", {})) or pick(j.get("result", {}))
    if not ans:
        # Some apps return a list of messages under data
        data = j.get("data") or j.get("result") or {}
        msgs = data.get("messages") if isinstance(data, dict) else None
        if isinstance(msgs, list) and msgs:
            for m in reversed(msgs):
                ans = pick(m) or pick(m.get("data", {}))
                if ans:
                    break
    if not ans:
        raise RuntimeError("Dify returned no answer in blocking mode.")
    return str(ans)


# -----------------------------
# OpenAI-only Dify-output formatter
# -----------------------------

def format_with_openai(text: str) -> str:
    """
    Use OpenAI to strictly format the Dify output into the required Markdown layout.
    If OPENAI_KEY is missing or the API fails, fall back to the original text.
    """
    if not OPENAI_KEY:
        return text
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "temperature": 0,
                "max_tokens": min(2048, max(300, len(text) // 4)),
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a strict Markdown formatter.
Your ONLY job: take whatever text the user provides (messy, unstructured, or chatty) and rewrite it into the exact schema below. Do not add greetings, explanations, or extra lines. No code blocks.

Look into the Output Format Example and Strictly Follow that Format.

Global formatting rules
- Titles are **bold lines** (no # headings, no trailing colons).
- Bullets must use '- ' only (dash + space). No other bullet glyphs.
- Numbered lists are allowed (1., 2., …) where specified.
- Keep meaning and wording where possible; fix obvious grammar/spelling lightly.
- One blank line between sections. No extra blank lines inside items.

Required section order & shapes

Thanks for uploading Pitch Deck, Model is cooking Please wait.

You can also do your Token audit using our Tokenomics Audit Tool: https://www.tokenomics.checker.tde.fi/

Here's What our Model Thinks For Pitch Deck:
Need Multiple Changes

1) **Strengths**
- A simple bullet list. Each item starts with '- '.

2) **Red Flags (High Priority Concerns)**
- A simple bullet list of the top issues. Each item starts with '- '.

3) **Improvement Tips**
- Use a numbered list.
- For each item, nest the following lines exactly:
  - **Slide Name**: <Title>
  - Why: <1–2 sentences>
  - Bullets:
    - <point>
    - <point>
  - Sources: <short reference or "Not specified">

4) **Add**
- Numbered list. Same 4 sub-lines as above. Use for NEW slides/content. If nothing to add, write '- None.'.

5) **Remove / Merge**
- Numbered list. Same 4 sub-lines as above. In 'Why', state what to remove/merge and why.

6) **Change**
- Numbered list. Same 4 sub-lines **plus** a line for a new title if present:
  - **Slide Name**: <Current Title>
  - New Title: <New Title>  (omit if not applicable)
  - Why: <...>
  - Bullets:
    - <point>
  - Sources: <...>

7) **Consistency Check**
- A simple bullet list of cross-checks/contradictions. If none, '- None noted.'.

8) **The Action Plan**
- A simple bullet list of next actions (5–8 concise bullets).

9) **Data Points (Can Include)**
- A simple bullet list of concrete numbers/tables to collect.

- Schedule a Demo Call
https://calendly.com/tdefi_project_calls/45min

Normalization & cleanup
- Convert any '•' or odd bullets → '- '.
- Remove quotes around slide names (example: - **Slide Name**: Competition & Differentiation).
- If a required field is missing, write a concise placeholder (e.g., - Sources: Not specified).
- If the input mentions 'Add:' items inside other sections, move them under **Add** with the proper structure.

Self-check before finalizing (do NOT print this checklist)
- All 10 sections appear in the exact order and with bold titles (no '#').
- Bullets use only '- '; sub-bullets are indented by two spaces then '- '.
- 'Improvement Tips', 'Add', 'Remove / Merge', and 'Change' use numbered items, each containing the required sub-lines.
- The final section shows the link on its own line (no bullets/text).

Output Formate Example (Strictly Follow This Format)

"Thanks for uploading Pitch Deck, Model is cooking Please wait.

You can also do your Token audit using our Tokenomics Audit Tool: https://www.tokenomics.checker.tde.fi/

Here's What our Model Thinks For Pitch Deck:
Need Multiple Changes

Reason

Lacks business model, competition, ask, and use of funds; missing market positioning essentials.

Strengths
	•	Shows real traction with 60K+ active users and notable partnerships.
	•	Clearly states the problem of centralized AI agent risks and why privacy-first matters now.
	•	Presents a technically differentiated solution (open-source protocols, cross-chain interoperability).

Red Flags (High Priority Concerns)
	•	No clear competitive landscape comparing SINT to leading AI/Web3 agent platforms.
	•	ICPs and use cases are broad; target segments and personas need sharpening.
	•	Funding ask lacks breakdown by category, runway, and milestones.
	•	Unit economics/pricing are unclear; need early monetization proof points.
	•	Traction by segment and conversion/retention metrics are missing.

Improvement Tips
    1. Slide Name: Competition & Differentiation
	•	Why: Investors must see direct comparisons and SINT’s edge over decentralized and centralized alternatives.
	•	Bullets:
	•	List top competitors (e.g., Fetch.ai, SingularityNET, Phala Network) and feature/funding/user comparisons.
	•	Highlight unique moat (e.g., MCP protocol, confidential compute stack).
	•	Sources: Not specified.

	2.	Slide Name: Go-To-Market Playbook
	•	Why: Needs concrete tactics per target segment from pilot → paid → scale.
	•	Bullets:
	•	Define ICPs, channels, and onboarding funnel with metrics.
	•	Include a pilot case study and milestone timeline.
	•	Sources: GTM broad in current deck.

	3.	Slide Name: Tokenomics Deep Dive
	•	Why: Investors expect clear token utility, allocation, and emissions logic.
	•	Bullets:
	•	Allocation pie chart and vesting schedule.
	•	Emissions/burn logic over time.
	•	Sources: Only surface-level token info present.

Add
    1. Slide Name: Competition & Differentiation
	•	Why: Deck lacks an explicit competitor view; investors need apples-to-apples comparison.
	•	Bullets:
	•	Competitor matrix (features/users/funding).
	•	Why partners choose SINT; evidence of switching.
	•	Sources: Not specified.

	2.	Slide Name: Go-To-Market Execution Plan
	•	Why: Must show how you’ll acquire, convert, and retain initial ICPs.
	•	Bullets:
	•	First ICPs, channels, and growth loop.
	•	Pilot → paid conversion metrics and targets.
	•	Sources: GTM sections are generic.

Remove / Merge
    1. Slide Name: Revenue from Partnerships
	•	Why: Repetitive with Business Model; merge for clarity.
	•	Bullets:
	•	Consolidate revenue details into the Business Model slide.
	•	Sources: Duplicates existing content.

Change 
    1. Slide Name: Business Model
	•	New Title: How SINT Makes Money & Scales Revenue
	•	Why: Current slide lists channels but lacks pricing/examples and scalability proof.
	•	Bullets:
	•	Show revenue streams with example pricing per user/partner type.
	•	Add early results or projections; explain percentage split logic vs. norms.
	•	Sources: Existing “Commissions/Subscriptions/Marketplace” references.

	2.	Slide Name: Market Opportunity
	•	New Title: Market Size & Segmentation
	•	Why: Needs TAM split by segments and priority focus with sources.
	•	Bullets:
	•	Break out $TAM by segment and cite sources for each figure.
	•	Clarify priority segments and near-term addressable market.
	•	Sources: TAM noted without detail.

Consistency Check
	•	$600k seed closed vs. $3.6M ask later—clarify whether this is a new round or cumulative.
	•	Revenue projections appear aggressive without conversion math.
	•	Initial circulating supply percentage has inconsistencies (if token is included).

The Action Plan
	•	Build a competitor matrix versus Fetch.ai, SingularityNET, Phala Network.
	•	Define ICPs and priority use cases; tie to channel tactics and milestones.
	•	Break down the $3.6M raise into categories mapped to runway and milestones.
	•	Add unit economics (pricing, margins) with worked examples.
	•	Collect activation, retention, and pilot → paid conversion metrics.
	•	Create a tokenomics deep-dive slide with allocation/vesting/emissions.

Data Points (Can Include)
	•	Recent MAU/PAU and cohort retention (D7/D30).
	•	Pricing tiers/examples per revenue stream.
	•	Competition research table (features/adoption/funding).
	•	Fundraising ask, valuation context, runway, and milestone plan.
	•	CAC/LTV/payback/churn where available.

Schedule a Demo Call
https://calendly.com/tdefi_project_calls/45min"

"""
                    },
                    {"role": "user", "content": text},
                ],
            },
            timeout=60,
        )
        if resp.status_code < 400:
            j = resp.json()
            content = ((j.get("choices") or [{}])[0].get("message") or {}).get("content")
            if content and content.strip():
                return content.strip()
    except Exception:
        pass
    return text

# -----------------------------
# Minimal single-shot UI with pre-upload + progress
# -----------------------------

accepted_mimes = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
]

files = st.file_uploader(
    "Attach your pitch deck (PDF / PPT / DOCX)",
    type=["pdf", "ppt", "pptx", "doc", "docx", "txt"],
    accept_multiple_files=True,
    help="Uploads are sent to Dify for this conversation.",
    key=f"uploader_{st.session_state['uploader_key']}"
)

# Removed per request: external max-size line under the dropzone

# Extra vertical breathing room under the banner
st.markdown('<div class="spacer32"></div>', unsafe_allow_html=True)

st.markdown('<div class="spacer24"></div>', unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color: #B7BEC9; font-size: 1.05rem; margin-bottom: 24px;'>PitchAudit AI analyzes your deck against proven investor templates, flags gaps, and rewrites slides using your content. Get a readiness score, must‑fix checklist, and clean exports.</div>",
    unsafe_allow_html=True
)

# When user selects files, upload them immediately and show a progress bar.
ready_to_start = False
current_fps = []
if files:
    # Compute fingerprints and upload any new ones
    total = len(files)
    uploaded_count = 0
    progress_bar_slot = st.empty()
    progress_text_slot = st.empty()
    bar = progress_bar_slot.progress(0)
    progress_text_slot.caption(f"Uploading files… (0/{total})")

    new_payload: List[dict] = []

    for idx, uf in enumerate(files, start=1):
        data = uf.getvalue()
        # Enforce 15MB per file limit
        if len(data) > 15 * 1024 * 1024:
            size_mb = len(data) / (1024 * 1024)
            progress_bar_slot.empty()
            progress_text_slot.empty()
            st.error(f"{uf.name} is {size_mb:.2f} MB. Maximum allowed is 15 MB.")
            st.stop()
        mime = uf.type or mimetypes.guess_type(uf.name)[0] or "application/octet-stream"
        fp = hashlib.sha256(data).hexdigest()
        current_fps.append(fp)

        if fp not in st.session_state.uploaded_fps:
            try:
                upload_id = dify_upload_file(uf.name, data, mime, st.session_state.dify_user)
                st.session_state.uploaded_fps[fp] = {
                    "id": upload_id,
                    "type": _infer_file_type(mime),
                    "name": uf.name,
                }
            except Exception as e:
                progress_bar_slot.empty()
                progress_text_slot.empty()
                st.error(f"Attachment upload failed — model is sleeping 😴. Try reloading.\n\nDetails: {e}")
                st.stop()
        # Build payload in current order
        rec = st.session_state.uploaded_fps[fp]
        new_payload.append({
            "type": rec["type"],
            "transfer_method": "local_file",
            "upload_file_id": rec["id"],
        })

        uploaded_count = idx
        pct = int((uploaded_count / total) * 100)
        # Move smoothly
        bar.progress(min(pct, 100))
        progress_text_slot.caption(f"Uploading files… {uf.name} ({idx}/{total})")

    # Remove entries for files no longer selected
    to_keep = set(current_fps)
    st.session_state.uploaded_fps = {k: v for k, v in st.session_state.uploaded_fps.items() if k in to_keep}

    st.session_state.uploaded_payload = new_payload
    progress_bar_slot.empty()
    progress_text_slot.empty()
    st.success(f"Files ready ✅  ({uploaded_count}/{total})")
    ready_to_start = uploaded_count == total and total > 0
else:
    # If no files selected, reset uploaded payload
    st.session_state.uploaded_payload = []

# Start button enabled only after uploads finished
start_placeholder = st.empty()
start = start_placeholder.button("Start", type="primary", use_container_width=True, disabled=(not ready_to_start))

if start:
    # Remove the Start button and show a "Model is cooking…" label with a live progress bar
    start_placeholder.empty()
    cooking_text = st.empty()
    cooking_text.markdown("<div style='text-align:center; color:#B7BEC9; margin: 6px 0;'>Model is cooking…</div>", unsafe_allow_html=True)

    files_payload: List[dict] = st.session_state.uploaded_payload or []

    RANDOM_QUERIES = [
        "Scan this pitch deck and give a crisp summary (company, problem, solution, traction, ask).",
        "Extract key metrics, GTM, business model, competitors and risks from this deck.",
        "Summarize the opportunity, product, moat, and top 5 red flags in this deck.",
        "Create a bullet summary of team, roadmap, market size, business model and funding ask.",
    ]
    query = random.choice(RANDOM_QUERIES)

    try:
        # Visual progress bar while the model is generating; fills to 90% gradually and 100% at completion.
        prog_slot = st.empty()
        prog = prog_slot.progress(0)

        # Consume Dify stream silently, only track progress; do not display raw output
        final_chunks: List[str] = []
        pct = 0
        for chunk in dify_stream_chat(query, files_payload or None, st.session_state.dify_user):
            pct = min(90, pct + 2)
            prog.progress(pct)
            final_chunks.append(str(chunk))
        prog.progress(95)

        final_text = "".join(final_chunks).strip()
        if not final_text:
            # Fallback: wait for final result with blocking call
            final_text = dify_blocking_chat(query, files_payload or None, st.session_state.dify_user)
        prog.progress(100)

        # Format the raw Dify output using OpenAI to match the required Markdown style
        final_output = format_with_openai(final_text)

        # clear the progress UI once complete
        prog_slot.empty()
        cooking_text.empty()
        st.session_state.last_output = final_output
        st.markdown(f"<div class='result-box'>{final_output}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error("Model is sleeping 😴. Try to reload the app.")
        with st.expander("Show technical details", expanded=False):
            st.code(str(e))

# Show last output if present (after rerun etc.)
if st.session_state.get("last_output") and not start:
    st.markdown(f"<div class='result-box'>{st.session_state.last_output}</div>", unsafe_allow_html=True)
