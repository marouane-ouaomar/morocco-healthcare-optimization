"""
webapp/pages/triage_bot.py
===========================
Phase 5 — Symptom Triage Chatbot (DEMO ONLY)
Chat-style UI using st.chat_message / st.chat_input.
See docs/SAFETY.md — research demo only, not a medical device.
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.facility_locator import LEVEL_FACILITY_TYPES, find_nearest
from src.triage_engine import EMERGENCY_NUMBERS, AdviceLevel, TriageResponse, triage

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Triage Bot · Morocco Healthcare",
    page_icon="🩺",
    layout="centered",
)

C = {
    "primary": "#1F4E79", "teal": "#2F8F9D",
    "danger":  "#922B21", "warning": "#CA6F1E",
    "success": "#1E8449", "muted":   "#5D6D7E",
    "border":  "#D5E8F0", "light_bg":"#EAF2F8",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&family=Source+Serif+4:wght@600;700&display=swap');
html,body,[class*="css"]{{font-family:'Source Sans 3',sans-serif;}}
#MainMenu,footer{{visibility:hidden;}}
.block-container{{max-width:760px;padding-top:2rem;padding-bottom:6rem;}}
[data-testid="stChatMessage"]{{border-radius:12px;margin-bottom:4px;}}
.emerg-card{{background:#922B21;color:white;border-radius:8px;
  padding:14px 18px;font-size:.88rem;line-height:1.6;margin:4px 0;}}
.emerg-card h4{{margin:0 0 8px 0;font-size:1rem;}}
.num-grid{{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:8px;}}
.num-item{{background:rgba(255,255,255,.15);border-radius:4px;
  padding:5px 9px;font-size:.8rem;}}
.num-item b{{display:block;font-size:.95rem;}}
.advice-card{{border-radius:8px;padding:14px 18px;border-left:4px solid;
  font-size:.86rem;line-height:1.65;margin:4px 0;}}
.advice-card h4{{margin:0 0 8px 0;font-size:.93rem;}}
.badge{{display:inline-block;font-size:.6rem;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;padding:2px 7px;border-radius:3px;
  background:rgba(0,0,0,.09);margin-top:8px;}}
.footer-note{{font-size:.7rem;color:#5D6D7E;margin-top:10px;
  padding-top:8px;border-top:1px solid #D5E8F0;line-height:1.5;}}
.fac-card{{display:flex;align-items:flex-start;gap:11px;background:white;
  border:1px solid #D5E8F0;border-radius:8px;padding:11px 14px;
  border-left:4px solid;margin-bottom:6px;}}
.fac-icon{{font-size:1.5rem;flex-shrink:0;margin-top:2px;line-height:1;}}
.fac-info{{flex:1;}}
.fac-name{{font-size:.86rem;font-weight:700;color:#1B2631;}}
.fac-type{{font-size:.7rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.8px;margin-top:2px;}}
.fac-region{{font-size:.73rem;color:#5D6D7E;margin-top:2px;}}
.fac-link{{color:#2F8F9D;font-size:.71rem;text-decoration:none;
  display:inline-block;margin-top:4px;}}
.fac-dist{{font-size:.95rem;font-weight:700;text-align:right;
  white-space:nowrap;flex-shrink:0;}}
.fac-km{{font-size:.63rem;font-weight:400;display:block;color:#5D6D7E;}}
.pub-badge{{background:#1E8449;color:white;font-size:.58rem;
  padding:1px 5px;border-radius:3px;margin-left:5px;}}
.priv-badge{{background:#CA6F1E;color:white;font-size:.58rem;
  padding:1px 5px;border-radius:3px;margin-left:5px;}}
</style>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
  background: linear-gradient(135deg, {C["primary"]} 0%, #163d61 100%);
  padding: 20px 26px;
  border-radius: 10px;
  border-left: 6px solid {C["teal"]};
  margin-bottom: 18px;
  box-shadow: 0 4px 16px rgba(31,78,121,0.18);
">
  <div style="display:flex;gap:4px;margin-bottom:10px;">
    <span style="display:inline-block;width:22px;height:4px;background:#c1272d;border-radius:2px;"></span>
    <span style="display:inline-block;width:22px;height:4px;background:#006233;border-radius:2px;"></span>
  </div>
  <div style="font-family:'Source Serif 4',serif;font-size:1.4rem;font-weight:700;
              color:white;line-height:1.2;">
    🩺 Triage Assistant
  </div>
  <div style="font-size:0.78rem;color:rgba(255,255,255,0.72);
    text-transform:uppercase;letter-spacing:1.2px;margin-top:6px;">
    Morocco Healthcare · Phase 5 Demo
  </div>
</div>""", unsafe_allow_html=True)

api_mode = bool(
    os.getenv("ANTHROPIC_API_KEY","").strip()
    and os.getenv("ANTHROPIC_API_KEY") != "your_anthropic_api_key_here"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "triage_done" not in st.session_state:
    st.session_state.triage_done = False
if "last_level" not in st.session_state:
    st.session_state.last_level = None

# ── Load facilities ───────────────────────────────────────────────────────────
FACILITIES_PATH = ROOT / "data/processed/facilities.geojson"

@st.cache_data(show_spinner=False)
def load_facilities_cached():
    if not FACILITIES_PATH.exists():
        return None
    return gpd.read_file(FACILITIES_PATH)

facilities_gdf = load_facilities_cached()

# ── Read location from query params ──────────────────────────────────────────
params = st.query_params
user_lat = user_lon = None
try:
    if "user_lat" in params and "user_lon" in params:
        user_lat = float(params["user_lat"])
        user_lon = float(params["user_lon"])
        if not (21.0 <= user_lat <= 36.0 and -18.0 <= user_lon <= -0.5):
            user_lat = user_lon = None
except (ValueError, TypeError):
    pass

# ── Geolocation component ─────────────────────────────────────────────────────
def geolocation_html() -> str:
    return """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  *{margin:0;padding:0;box-sizing:border-box;font-family:"Segoe UI",sans-serif;}
  body{background:transparent;padding:2px 0;}
  #btn{background:#1F4E79;color:white;border:none;border-radius:6px;
    padding:8px 16px;font-size:12.5px;font-weight:600;cursor:pointer;
    letter-spacing:.4px;transition:background .2s;}
  #btn:hover{background:#2F8F9D;}
  #btn:disabled{background:#aaa;cursor:default;}
  #st{font-size:11.5px;color:#5D6D7E;margin-top:6px;}
  .ok{color:#1E8449!important;font-weight:600;}
  .err{color:#922B21!important;}
</style></head><body>
<button id="btn" onclick="go()">📍 Share My Location</button>
<div id="st">Grant location access to find nearby facilities.</div>
<script>
function go(){
  var btn=document.getElementById("btn"),st=document.getElementById("st");

  // Walk up to the true top-level window (works on localhost AND Streamlit Cloud)
  var topWin = window;
  try { while (topWin !== topWin.parent) { topWin = topWin.parent; } } catch(e) {}

  var geo = topWin.navigator.geolocation || navigator.geolocation;
  if(!geo){st.textContent="❌ Geolocation not supported.";st.className="err";return;}

  btn.disabled=true;btn.textContent="⏳ Locating...";
  st.textContent="Check your browser address bar for the permission prompt...";

  geo.getCurrentPosition(
    function(p){
      var lat=p.coords.latitude.toFixed(6),lon=p.coords.longitude.toFixed(6);
      st.textContent="✅ Got it — updating...";st.className="ok";

      try {
        // Build new URL from the top-level window location
        var url=new URL(topWin.location.href);
        url.searchParams.set("user_lat",lat);
        url.searchParams.set("user_lon",lon);
        topWin.location.href=url.toString();
      } catch(e) {
        // Fallback: try window.top directly
        try {
          var url2=new URL(window.top.location.href);
          url2.searchParams.set("user_lat",lat);
          url2.searchParams.set("user_lon",lon);
          window.top.location.href=url2.toString();
        } catch(e2) {
          st.textContent="❌ Could not update location. Try the manual entry below.";
          st.className="err";
        }
      }
    },
    function(e){
      btn.disabled=false;btn.textContent="📍 Share My Location";
      var m={1:"❌ Permission denied — click the 🔒 icon in the address bar.",
             2:"❌ Position unavailable.",3:"❌ Timed out."};
      st.textContent=m[e.code]||"❌ Error.";st.className="err";
    },
    {enableHighAccuracy:true,timeout:10000,maximumAge:60000}
  );
}
</script></body></html>"""

# ── HTML helpers ──────────────────────────────────────────────────────────────
def emergency_html(advice_text: str) -> str:
    nums = "".join(
        f"<div class='num-item'>{n}<b>{v}</b></div>"
        for n,v in EMERGENCY_NUMBERS.items()
    )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>
      *{{margin:0;padding:0;box-sizing:border-box;font-family:"Segoe UI",sans-serif;}}
      .emerg-card{{background:#922B21;color:white;border-radius:8px;
        padding:14px 18px;font-size:.88rem;line-height:1.6;}}
      .emerg-card h4{{margin:0 0 8px 0;font-size:1rem;}}
      .num-grid{{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:8px;}}
      .num-item{{background:rgba(255,255,255,.15);border-radius:4px;padding:5px 9px;font-size:.8rem;}}
      .num-item b{{display:block;font-size:.95rem;}}
    </style></head><body>
    <div class="emerg-card">
      <h4>🚨 Emergency — Call Immediately</h4>
      <div>{advice_text}</div>
      <div class="num-grid">{nums}</div>
    </div></body></html>"""


def advice_html(title: str, advice_text: str, disclaimer: str,
                bg: str, border: str, fg: str, source: str) -> str:
    src_lbl = {"rule":"Rule-based","llm":"AI-assisted","fallback":"Local fallback"}.get(source, source)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>
      *{{margin:0;padding:0;box-sizing:border-box;font-family:"Segoe UI",sans-serif;}}
      .card{{border-radius:8px;padding:14px 18px;border-left:4px solid {border};
        background:{bg};font-size:.86rem;line-height:1.65;color:{fg};}}
      h4{{margin:0 0 8px 0;font-size:.93rem;}}
      .badge{{display:inline-block;font-size:.6rem;font-weight:700;letter-spacing:1px;
        text-transform:uppercase;padding:2px 7px;border-radius:3px;
        background:rgba(0,0,0,.09);margin-top:8px;}}
      .footer-note{{font-size:.7rem;color:#5D6D7E;margin-top:10px;
        padding-top:8px;border-top:1px solid #D5E8F0;line-height:1.5;}}
    </style></head><body>
    <div class="card"><h4>{title}</h4><div>{advice_text}</div>
    <div class="badge">{src_lbl}</div>
    <div class="footer-note">{disclaimer}</div></div>
    </body></html>"""


def facility_card_html(fac) -> str:
    dist_color = "#1E8449" if fac.distance_km < 5 else "#CA6F1E" if fac.distance_km < 20 else "#922B21"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>
      *{{margin:0;padding:0;box-sizing:border-box;font-family:"Segoe UI",sans-serif;}}
      .card{{display:flex;align-items:flex-start;gap:11px;background:white;
        border:1px solid #D5E8F0;border-radius:8px;padding:11px 14px;
        border-left:4px solid {fac.color};}}
      .icon{{font-size:1.5rem;flex-shrink:0;margin-top:2px;line-height:1;}}
      .info{{flex:1;}}
      .name{{font-size:.86rem;font-weight:700;color:#1B2631;}}
      .type{{font-size:.7rem;font-weight:600;text-transform:uppercase;
        letter-spacing:.8px;margin-top:2px;color:{fac.color};}}
      .region{{font-size:.73rem;color:#5D6D7E;margin-top:2px;}}
      a{{color:#2F8F9D;font-size:.71rem;text-decoration:none;
        display:inline-block;margin-top:4px;}}
      a:hover{{text-decoration:underline;}}
      .dist{{font-size:.95rem;font-weight:700;text-align:right;
        white-space:nowrap;flex-shrink:0;color:{dist_color};}}
      .km{{font-size:.63rem;font-weight:400;display:block;color:#5D6D7E;}}
      .pub{{background:#1E8449;color:white;font-size:.58rem;
        padding:1px 5px;border-radius:3px;margin-left:5px;}}
      .priv{{background:#CA6F1E;color:white;font-size:.58rem;
        padding:1px 5px;border-radius:3px;margin-left:5px;}}
    </style></head><body>
    <div class="card">
      <div class="icon">{fac.icon}</div>
      <div class="info">
        <div class="name">{fac.name}
          {"<span class='pub'>Public</span>" if fac.sector=="public" else "<span class='priv'>Private</span>" if fac.sector=="private" else ""}
        </div>
        <div class="type">{fac.label}</div>
        <div class="region">📌 {fac.region}</div>
        <a href="{fac.google_maps_url}" target="_blank">🗺 Open in Google Maps</a>
      </div>
      <div class="dist">{fac.distance_km:.1f}<span class="km">km away</span></div>
    </div></body></html>"""


# ── Render a stored message ───────────────────────────────────────────────────
def render_message(msg: dict) -> None:
    role = msg["role"]
    kind = msg.get("kind", "text")

    with st.chat_message(role, avatar="🩺" if role == "assistant" else "🧑"):
        if kind == "text":
            st.markdown(msg["content"])

        elif kind == "disclaimer":
            st.markdown("""
            <div style='background:#FEF9E7;border:2px solid #CA6F1E;border-radius:8px;
              padding:14px 18px;font-size:.82rem;line-height:1.6;color:#5D4E00;'>
              <b>⚠️ Demo only — not a medical device.</b> In an emergency call
              <b>SAMU 15</b> or <b>112</b> immediately. This tool does not diagnose.
            </div>""", unsafe_allow_html=True)

        elif kind == "emergency":
            components.html(emergency_html(msg["advice_text"]), height=220, scrolling=False)

        elif kind == "advice":
            meta = {
                "see_doctor": ("🩺 Consult a Doctor", "#EBF5FB", "#2F8F9D", "#154360"),
                "monitor":    ("👁 Monitor Symptoms",  "#EAFAF1", "#1E8449", "#1D6A39"),
            }
            title, bg, border, fg = meta.get(msg["level"], ("ℹ️ Guidance","#EAF2F8","#5D6D7E","#1B2631"))
            components.html(
                advice_html(title, msg["advice_text"], msg["disclaimer"], bg, border, fg, msg["source"]),
                height=210, scrolling=False,
            )

        elif kind == "geolocation":
            components.html(geolocation_html(), height=72, scrolling=False)

        elif kind == "location_confirmed":
            st.markdown(
                f"<div style='font-size:.8rem;color:#1E8449;font-weight:600;'>"
                f"✅ Location captured: {msg['lat']:.4f}°N, {msg['lon']:.4f}°E</div>",
                unsafe_allow_html=True,
            )

        elif kind == "facilities":
            rec = {
                "monitor":    "💊 For mild symptoms — nearest **pharmacies**:",
                "see_doctor": "🩺 For professional evaluation — nearest **clinics/doctors**:",
                "emergency":  "🏥 Nearest **hospitals** — go immediately:",
            }
            st.markdown(rec.get(msg["level"], "📍 Nearest facilities:"))
            for fac in msg["facilities"]:
                components.html(facility_card_html(fac), height=125, scrolling=False)

        elif kind == "no_facilities":
            st.info("No facilities found within 150 km. Please contact emergency services directly.")

        elif kind == "facility_prompt":
            st.markdown(msg["content"])


# ── Replay all stored messages ────────────────────────────────────────────────
for msg in st.session_state.messages:
    render_message(msg)

# ── Boot greeting (first load only) ──────────────────────────────────────────
if not st.session_state.messages:
    greeting_msgs = [
        {"role": "assistant", "kind": "disclaimer", "content": ""},
        {"role": "assistant", "kind": "text",
         "content": (
             f"Hello! I'm your **symptom triage assistant** 👋\n\n"
             f"Tell me what you're experiencing and I'll suggest whether you need "
             f"a **pharmacy**, a **clinic**, or **emergency care** — and show you the "
             f"nearest one to you.\n\n"
             f"*Mode: {'🤖 AI-assisted' if api_mode else '📋 Rule-based fallback'}*"
         )},
    ]
    for m in greeting_msgs:
        st.session_state.messages.append(m)
        render_message(m)

# ── If location just arrived, show facilities without waiting for new message ─
if user_lat is not None and st.session_state.triage_done and st.session_state.last_level:
    already_shown = any(m.get("kind") == "facilities" for m in st.session_state.messages)
    if not already_shown and facilities_gdf is not None:
        level = st.session_state.last_level
        nearby = find_nearest(
            user_lat=user_lat, user_lon=user_lon,
            facilities_gdf=facilities_gdf,
            advice_level=level, n=3, max_distance_km=150.0,
        )
        loc_msg = {"role": "assistant", "kind": "location_confirmed",
                   "lat": user_lat, "lon": user_lon}
        st.session_state.messages.append(loc_msg)
        render_message(loc_msg)

        if nearby:
            fac_msg = {"role":"assistant","kind":"facilities","level":level,"facilities":nearby}
        else:
            fac_msg = {"role":"assistant","kind":"no_facilities","content":""}
        st.session_state.messages.append(fac_msg)
        render_message(fac_msg)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Describe your symptoms…"):

    user_msg = {"role": "user", "kind": "text", "content": prompt}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Analysing…"):
            result: TriageResponse = triage(prompt)

        level = result.advice_level
        st.session_state.triage_done = True
        st.session_state.last_level = level

        if level == AdviceLevel.EMERGENCY or result.escalate:
            msg = {"role":"assistant","kind":"emergency","advice_text":result.advice_text}
            st.session_state.messages.append(msg)
            components.html(emergency_html(result.advice_text), height=220, scrolling=False)

        else:
            msg = {
                "role":"assistant","kind":"advice","level":level,
                "advice_text":result.advice_text,
                "disclaimer":result.disclaimer,"source":result.source,
            }
            st.session_state.messages.append(msg)
            meta = {
                AdviceLevel.SEE_DOCTOR: ("🩺 Consult a Doctor","#EBF5FB","#2F8F9D","#154360"),
                AdviceLevel.MONITOR:    ("👁 Monitor Symptoms", "#EAFAF1","#1E8449","#1D6A39"),
            }
            title,bg,border,fg = meta.get(level,("ℹ️","#EAF2F8","#5D6D7E","#1B2631"))
            components.html(
                advice_html(title,result.advice_text,result.disclaimer,bg,border,fg,result.source),
                height=210, scrolling=False,
            )

        if user_lat is not None and facilities_gdf is not None:
            nearby = find_nearest(
                user_lat=user_lat, user_lon=user_lon,
                facilities_gdf=facilities_gdf,
                advice_level=level, n=3, max_distance_km=150.0,
            )
            if nearby:
                rec = {
                    "monitor":    "💊 For mild symptoms — nearest **pharmacies**:",
                    "see_doctor": "🩺 Nearest **clinics/doctors** for evaluation:",
                    "emergency":  "🏥 Nearest **hospitals** — go immediately:",
                }
                st.markdown(rec.get(level, "📍 Nearest facilities:"))
                fac_msg = {"role":"assistant","kind":"facilities","level":level,"facilities":nearby}
                st.session_state.messages.append(fac_msg)
                for fac in nearby:
                    components.html(facility_card_html(fac), height=125, scrolling=False)
            else:
                no_fac = {"role":"assistant","kind":"no_facilities","content":""}
                st.session_state.messages.append(no_fac)
                st.info("No facilities found within 150 km.")
        else:
            geo_prompt = {
                "role": "assistant", "kind": "facility_prompt",
                "content": (
                    "📍 **Share your location** below to see the nearest "
                    + ("pharmacy." if level == AdviceLevel.MONITOR
                       else "clinic or doctor." if level == AdviceLevel.SEE_DOCTOR
                       else "hospital.")
                ),
            }
            st.session_state.messages.append(geo_prompt)
            st.markdown(geo_prompt["content"])
            geo_msg = {"role":"assistant","kind":"geolocation","content":""}
            st.session_state.messages.append(geo_msg)
            components.html(geolocation_html(), height=72, scrolling=False)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='font-size:.65rem;font-weight:700;letter-spacing:2px;
      text-transform:uppercase;color:{C["primary"]};margin-bottom:8px;'>
      Triage Bot Controls
    </div>""", unsafe_allow_html=True)

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.triage_done = False
        st.session_state.last_level = None
        st.query_params.clear()
        st.rerun()

    if user_lat is not None:
        st.markdown(
            f"<div style='font-size:.75rem;color:{C['success']};margin-top:8px;'>"
            f"📍 {user_lat:.4f}°N, {user_lon:.4f}°E</div>",
            unsafe_allow_html=True,
        )
        if st.button("📍 Reset location", use_container_width=True):
            st.query_params.clear()
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:.72rem;color:{C["muted"]};line-height:1.6;'>
      <b> Demo only</b><br>
      Not a medical device.<br>
      Emergency: <b>Call 15</b> / <b>112</b>
    </div>""", unsafe_allow_html=True)