# app.py — Streamlit x Dify "Pitch Audit AI" (single-shot app)
# Run:  streamlit run app.py
# Env vars expected (or via .streamlit/secrets.toml):
#   NEXT_PUBLIC_API_URL   (default: https://api.dify.ai/v1)
#   NEXT_PUBLIC_APP_KEY   (required)
#   NEXT_PUBLIC_APP_ID    (optional)

import os
import json
import base64
import random
import mimetypes
import uuid
import hashlib
import re
from typing import Generator, List, Optional

import requests
import streamlit as st

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

# Safety check for key (no sidebar inputs anymore)
if not API_KEY:
    st.warning("Set NEXT_PUBLIC_APP_KEY as an environment variable or in .streamlit/secrets.toml to use the app.")

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


# Structured query builder and renderer helpers
def build_structured_query() -> str:
    """
    Ask Dify to return STRICT JSON that we can render into a clean, consistent layout.
    This leaves your server-side workflow untouched and enforces a schema client-side.
    """
    schema = {
        "one_line_verdict": {"status": "", "why": ""},
        "does_well": [],
        "must_fix_now": [],
        "slides_to_add": [
            {
                "title": "",
                "why": "",
                "bullets": [],
                "sources": []
            }
        ],
        "slides_to_remove_or_merge": [],
        "slides_to_change": [],
        "numbers_consistency_check": [],
        "action_plan": [],
        "data_to_collect": []
    }

    # We instruct the model to output ONLY minified JSON for easy parsing.
    return (
        "You are Pitch Audit AI. Read the uploaded deck files and produce a rigorous audit. "
        "Output ONLY MINIFIED JSON (no markdown, no prose) that exactly matches this schema keys: "
        + json.dumps(schema, separators=(',', ':'))
        + ". Populate all fields thoughtfully; use [] when unknown. DO NOT include backticks."
    )


def try_render_structured(raw_text: str) -> Optional[str]:
    """
    Attempt to parse the model output as JSON (even if it includes stray text/code fences),
    and render a nicely structured Markdown block. Returns None if parsing fails.
    """
    # Strip any code fences
    cleaned = raw_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Extract the widest plausible JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None

    json_str = cleaned[start:end+1]
    try:
        data = json.loads(json_str)
    except Exception:
        return None

    # Helper to bullet lists
    def bullets(items):
        if not items:
            return "- _Not present_\n"
        return "".join([f"- {str(x).strip()}\n" for x in items if str(x).strip()])

    # Slides to add (numbered with inner bullets)
    slides_md = ""
    for i, s in enumerate(data.get("slides_to_add") or [], start=1):
        title = (s or {}).get("title") or "Slide"
        why = (s or {}).get("why") or ""
        pts = (s or {}).get("bullets") or []
        srcs = (s or {}).get("sources") or []
        slides_md += f"\n{i}.\n\nSlide Name \"{title}\" Why\n\n"
        if why:
            slides_md += f"- {why}\n"
        if pts:
            for p in pts:
                slides_md += f"- {p}\n"
        if srcs:
            slides_md += f"Sources: {', '.join([str(x) for x in srcs])}\n"

    verdict = data.get("one_line_verdict") or {}
    status = verdict.get("status") or "_Not stated_"
    why_verdict = verdict.get("why") or "_Not stated_"

    md = []
    md.append("Thanks for uploading Pitch Deck, Model is cooking  \nPlease wait.")
    md.append("")
    md.append("You can also do your token audit using our Tokenomics Audit Tool: https://www.tokenomics.checker.tde.fi/")
    md.append("")
    md.append("Here's What our Model Thinks For  Pitch Deck:")
    md.append(f"**{status}**")
    md.append("")
    md.append("**Reason**")
    md.append(why_verdict)
    md.append("")
    md.append("**Strengths:**")
    md.append(bullets(data.get("does_well")))
    md.append("")
    md.append("**Red Flags (High Priority Concerns):**")
    md.append(bullets(data.get("must_fix_now")))
    md.append("")
    md.append("**Improvement Tips:**")
    md.append("")
    md.append("- **Add**")
    md.append(bullets([(s.get("title") or "Slide") for s in (data.get("slides_to_add") or [])]))
    md.append(slides_md if slides_md else "")
    md.append("")
    md.append("- **Remove / Merge**")
    md.append(bullets(data.get("slides_to_remove_or_merge")))
    md.append("")
    md.append("- **Change**")
    md.append(bullets(data.get("slides_to_change")))
    md.append("")
    md.append("**Consistency Check**")
    md.append(bullets(data.get("numbers_consistency_check")))
    md.append("")
    md.append("**The Action Plan**")
    md.append(bullets(data.get("action_plan")))
    md.append("")
    md.append("**Data Points (Can Include):**")
    md.append(bullets(data.get("data_to_collect")))
    md.append("")
    md.append("Scheduled a demo call with us for more insights:\n\nhttps://calendly.com/tdefi_project_calls/45min")

    # Wrap in your result box
    return "<div class='result-box'>" + "\n\n".join(md) + "</div>"


# -----------------------------
# Heuristic normalization of free-form Dify output to template
# -----------------------------
def normalize_to_template(raw_text: str) -> str:
    """
    Heuristically convert a free-form Dify answer into the standard, neat template.
    We DO NOT change your workflow—this only post-processes whatever text Dify returns.
    """
    if not raw_text:
        return ""

    t = raw_text.replace("\r\n", "\n")
    # Remove code fences if any
    t = t.replace("```json", "").replace("```", "")

    # Normalize bullets
    bullet_chars = ["•", "◦", "–", "—", "·", "*", "↳", "»"]
    for ch in bullet_chars:
        t = t.replace(f"\n{ch} ", "\n- ")
        t = t.replace(f"\n {ch} ", "\n- ")
    # Ensure any remaining lines that begin with Unicode dashes are treated as '-'
    t = re.sub(r"^\s*[–—]\s+", "- ", t, flags=re.MULTILINE)

    # Section headers mapping
    header_map = {
        "verdict": re.compile(r"(?im)^\s*(one[-\s]?line\s+verdict|verdict|status)\s*[:\-]?\s*$"),
        "reason": re.compile(r"(?im)^\s*(reason|why)\s*[:\-]?\s*$"),
        "strengths": re.compile(r"(?im)^\s*(strengths|does\s*well|pros|advantages)\s*[:\-]?\s*$"),
        "red_flags": re.compile(r"(?im)^\s*(red\s*flags?|high\s*priority\s*concerns|cons|risks|must\s*fix(\s*now)?)\s*[:\-]?\s*$"),
        "add": re.compile(r"(?im)^\s*(slides?\s*to\s*add|add)\s*[:\-]?\s*$"),
        "remove_merge": re.compile(r"(?im)^\s*(slides?\s*to\s*remove\s*/?\s*merge|remove\s*/?\s*merge|remove|merge)\s*[:\-]?\s*$"),
        "change": re.compile(r"(?im)^\s*(slides?\s*to\s*change|change|tweaks?)\s*[:\-]?\s*$"),
        "consistency": re.compile(r"(?im)^\s*(consistency\s*check|numbers\s*consistency|sanity\s*check)\s*[:\-]?\s*$"),
        "action": re.compile(r"(?im)^\s*(the\s*action\s*plan|action\s*plan|next\s*steps|todo|to\-do)\s*[:\-]?\s*$"),
        "data": re.compile(r"(?im)^\s*(data\s*points?(?:\s*\(can\s*include\))?|metrics\s*to\s*collect|evidence\s*to\s*collect)\s*[:\-]?\s*$"),
        # Improvement subsections (nested)
        "improvement": re.compile(r"(?im)^\s*(improvement\s*tips?)\s*[:\-]?\s*$"),
        "impr_add": re.compile(r"(?im)^\s*(\-?\s*)?(add)\s*[:\-]?\s*$"),
        "impr_remove": re.compile(r"(?im)^\s*(\-?\s*)?(remove\s*/?\s*merge|remove|merge)\s*[:\-]?\s*$"),
        "impr_change": re.compile(r"(?im)^\s*(\-?\s*)?(change|tweaks?)\s*[:\-]?\s*$"),
    }

    # State accumulators
    sections = {
        "verdict": [],
        "reason": [],
        "strengths": [],
        "red_flags": [],
        "add": [],
        "remove_merge": [],
        "change": [],
        "consistency": [],
        "action": [],
        "data": [],
    }

    current = None
    impr_mode = None  # which of add/remove/change inside Improvement Tips

    for line in t.split("\n"):
        ln = line.strip()
        if not ln:
            continue

        # Check for top-level headers
        matched_header = False
        for key, rx in header_map.items():
            if key.startswith("impr_") or key == "improvement":
                continue
            if rx.match(ln):
                current = key
                impr_mode = None
                matched_header = True
                break
        if matched_header:
            continue

        # Improvement tips group
        if header_map["improvement"].match(ln):
            current = "add"  # default to add until we hit a subheader
            impr_mode = "add"
            continue
        if header_map["impr_add"].match(ln):
            current = "add"
            impr_mode = "add"
            continue
        if header_map["impr_remove"].match(ln):
            current = "remove_merge"
            impr_mode = "remove_merge"
            continue
        if header_map["impr_change"].match(ln):
            current = "change"
            impr_mode = "change"
            continue

        # Bucket bullets vs paragraphs
        if ln.startswith("- "):
            bucket = current or ("strengths" if not sections["strengths"] else "red_flags")
            sections[bucket].append(ln[2:].strip())
        else:
            # If this looks like "Status: XYZ" capture as verdict
            m = re.match(r"(?i)^\s*(status|verdict)\s*:\s*(.+)$", ln)
            if m:
                sections["verdict"].append(m.group(2).strip())
            else:
                # Non-bulleted text — prefer Reason if empty, else Action
                target = "reason" if not sections["reason"] else "action"
                sections[target].append(ln)

    # Build the final markdown using the template the user provided
    def bullets_md(items):
        if not items:
            return "- _Not present_\n"
        return "".join([f"- {x}\n" for x in items])

    verdict_line = sections["verdict"][0] if sections["verdict"] else "_Not stated_"
    reason_text = "\n".join(sections["reason"]) if sections["reason"] else "_Not stated_"

    md = []
    md.append("Thanks for uploading Pitch Deck, Model is cooking  \nPlease wait.\n")
    md.append("You can also do your token audit using our Tokenomics Audit Tool: https://www.tokenomics.checker.tde.fi/\n")
    md.append("Here's What our Model Thinks For  Pitch Deck:\n")
    md.append(f"**{verdict_line}**\n")
    md.append("**Reason**\n")
    md.append(f"{reason_text}\n")
    md.append("**Strengths:**\n")
    md.append(bullets_md(sections["strengths"]))
    md.append("\n**Red Flags(High Priority Concerns) :**\n")
    md.append(bullets_md(sections["red_flags"]))
    md.append("\n**Improvement Tips:**\n\n- **Add**\n")
    md.append(bullets_md(sections["add"]))
    md.append("\n- **Remove / Merge**\n")
    md.append(bullets_md(sections["remove_merge"]))
    md.append("\n- **Change**\n")
    md.append(bullets_md(sections["change"]))
    md.append("\n**Consistency Check**\n")
    md.append(bullets_md(sections["consistency"]))
    md.append("\n**The Action Plan**\n")
    md.append(bullets_md(sections["action"]))
    md.append("\n**Data Points (Can Include):**\n")
    md.append(bullets_md(sections["data"]))
    md.append("\nScheduled a demo call with us for more insights:\n\nhttps://calendly.com/tdefi_project_calls/45min")
    return "<div class='result-box'>" + "\n".join(md) + "</div>"


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
    payload = {"inputs": {}, "query": query, "response_mode": "streaming", "user": user}
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
            # Common incremental tokens
            if event_type in ("message", "agent_message", "tool_message", "message_delta"):
                delta = (
                    evt.get("answer")
                    or evt.get("output_text")
                    or evt.get("data")  # sometimes dify nests text in data
                    or ""
                )
                if isinstance(delta, dict):
                    # best effort: try common keys
                    delta = delta.get("text") or delta.get("content") or ""
                if delta:
                    yield str(delta)

            # Some Dify deployments only send the final text on *_end or completed events.
            elif event_type in ("message_end", "agent_message_end", "completed", "workflow_finished"):
                final_ans = evt.get("answer") or evt.get("output_text") or ""
                if isinstance(final_ans, dict):
                    final_ans = final_ans.get("text") or final_ans.get("content") or ""
                if final_ans:
                    yield str(final_ans)
                break

            elif event_type == "error":
                err_msg = evt.get("message") or evt.get("error") or "Model error"
                raise RuntimeError(err_msg)

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

    # Ask for structured JSON so we can render a consistent layout
    RANDOM_QUERIES = [
        "Scan this pitch deck and give a crisp summary (company, problem, solution, traction, ask).",
        "Extract key metrics, GTM, business model, competitors and risks from this deck.",
        "Summarize the opportunity, product, moat, and top 5 red flags in this deck.",
        "Create a bullet summary of team, roadmap, market size, business model and funding ask.",
    ]
    query = RANDOM_QUERIES[0]

    try:
        # Visual progress bar while the model is generating; fills to 90% gradually and 100% at completion.
        prog_slot = st.empty()
        prog = prog_slot.progress(0)

        raw_chunks = []
        pct = 0
        for chunk in dify_stream_chat(query, files_payload or None, st.session_state.dify_user):
            raw_chunks.append(chunk)
            pct = min(90, pct + 2)
            prog.progress(pct)

        prog.progress(100)
        prog_slot.empty()
        cooking_text.empty()

        raw_text = "".join(raw_chunks)

        # Try to render the structured JSON; if it fails, heuristically structure the free-form text;
        # if that also fails, fall back to raw.
        structured_md = try_render_structured(raw_text)
        if not structured_md:
            structured_md = normalize_to_template(raw_text)
        if structured_md:
            st.markdown(structured_md, unsafe_allow_html=True)
            st.session_state.last_output = structured_md
        else:
            st.markdown(f"<div class='result-box'>{raw_text}</div>", unsafe_allow_html=True)
            st.session_state.last_output = raw_text
    except Exception as e:
        st.error("Model is sleeping 😴. Try to reload the app.")
        with st.expander("Show technical details", expanded=False):
            st.code(str(e))

# Show last output if present (after rerun etc.)
if st.session_state.get("last_output") and not start:
    st.markdown(f"{st.session_state.last_output}", unsafe_allow_html=True)
