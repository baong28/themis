# ===========================================
# ⚖️ Themis – Legal Discovery AI Assistant
# Author: baong28 @ Kirkendall Dwyer Law
# ===========================================

import streamlit as st
from datetime import datetime
import time
import logging
from index import *
from ask import *
from prompts.builder import *

st.set_page_config(
    page_title="Themis/Athena - Legal Discovery AI Assistant from Kirkendall Dwyer Law",
    layout="centered",
    page_icon="⚖️"
)

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

# ============================================================
# 💅 STYLE SECTION
# ============================================================
# --- Custom CSS ---         background-color: #4c516d;
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f5f6fa; 
        color: #333;
    }

    .chat-container {
        width: 100%;
        max-width: 800px;
        margin: auto;
        padding: 1rem;
    }

    .chat-bubble {
        border-radius: 18px;
        padding: 14px 20px;
        margin: 10px 0;
        font-size: 16px;
        max-width: 85%;
        line-height: 1.6;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    }

    .user-bubble {
        background: linear-gradient(to right, #d1f7e0, #a8edd3);
        color: #000;
        align-self: flex-end;
        text-align: right;
        margin-left: auto;
    }

    .puppy-bubble {
        background: linear-gradient(to right, #e0f0fa, #ffffff); /* gradient nhạt */
        color: #1e3d59; /* cùng màu header */
        align-self: flex-start;
        text-align: left;
        margin-right: auto;
    }

    .timestamp {
        font-size: 11px;
        color: #888;
        margin-bottom: 4px;
    }

    .avatar {
        font-size: 22px;
        vertical-align: middle;
        margin-right: 5px;
    }

    .title-header {
        font-size: 30px;
        text-align: center;
        font-weight: bold;
        color: #1e3d59;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        font-size: 14px;
        color: #666;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 🏛️ HEADER SECTION
# ============================================================
st.markdown('<div class="title-header">⚖️ Themis – Legal Discovery AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Trained with data. Driven by curiosity. Here to help your journey at Kirkendall Dwyer Law.</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Made by <b>baong28</b> from KD.</div>', unsafe_allow_html=True)

# ============================================================
# 🧭 SIDEBAR CONFIGURATION
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration Panel")

    # Mode selection
    st.subheader("🧠 Assistant Mode")
    mode = st.radio(
        "Choose model specialization:",
        options=["Themis (Legal Discovery)", "Athena (Research Insight)"],
        index=0,
        help="Select Themis for legal context or Athena for general research."
    )

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("⚖️ Themis v1.0 | © 2025 Kirkendall Dwyer Law | Built by baong28")

# ============================================================
# 💬 CHAT SESSION HANDLING
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

def display_message(role: str, content: str, timestamp: str):
    """Render a single chat bubble."""
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    avatar = "👤" if role == "user" else "⚖️"
    align = "right" if role == "user" else "left"
    st.markdown(f"""
        <div style="text-align: {align};">
            <div class="timestamp">{avatar} {timestamp}</div>
            <div class="chat-bubble {bubble_class}">{content}</div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# 🧠 MAIN CHAT UI
# ============================================================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Show previous messages
for msg in st.session_state.messages:
    display_message(msg["role"], msg["content"], msg["time"])

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# ✍️ USER INPUT
# ============================================================
user_input = st.chat_input(f"What would you like to ask {mode}?")

if user_input:
    now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_input, "time": now})
    display_message("user", user_input, now)

    try:
        with st.spinner(f"⚖️ {mode} is thinking..."):
            logging.info(f"[{mode}] User: {user_input}")

            # ✅ ask() now only takes one argument
            full_reply = ask(user_input)
            reply_time = datetime.now().strftime("%H:%M")

            # Typing animation
            placeholder = st.empty()
            simulated = ""
            for char in full_reply:
                simulated += char
                placeholder.markdown(f"""
                    <div style="text-align: left;">
                        <div class="timestamp">⚖️ {reply_time}</div>
                        <div class="chat-bubble assistant-bubble">{simulated}▌</div>
                    </div>
                """, unsafe_allow_html=True)
                time.sleep(0.001)

            # Final output
            placeholder.markdown(f"""
                <div style="text-align: left;">
                    <div class="timestamp">⚖️ {reply_time}</div>
                    <div class="chat-bubble assistant-bubble">{simulated}</div>
                </div>
            """, unsafe_allow_html=True)

            # Save message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_reply,
                "time": reply_time
            })
            logging.info("Response generated successfully.")

    except Exception as e:
        logging.exception("Error during response generation.")
        st.error(f"⚠️ An error occurred: {str(e)}")
