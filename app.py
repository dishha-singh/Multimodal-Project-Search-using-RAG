import os

import requests
import streamlit as st
from PIL import Image

# The frontend only draws the page. All the work (CLIP, Qdrant, Gemini, logging)
# happens in the FastAPI backend: start it first with   uvicorn backend:app --port 8000
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Multi-Modal Product Search", layout="wide")
st.title("Multi-Modal Product Search")

# On screens narrower than 1000 px the two halves stack (search first, results below).
st.markdown(
    """
    <style>
    @media (max-width: 1000px) {
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        div[data-testid="stColumn"], div[data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_colours():
    r = requests.get(f"{API_URL}/colours", timeout=10)
    r.raise_for_status()
    return r.json()["colours"]


try:
    COLOURS = fetch_colours()
except requests.RequestException:
    st.error("Cannot reach the backend. Start it in another terminal with:\n\n"
             "uvicorn backend:app --port 8000")
    st.stop()

left, right = st.columns(2, gap="large")

# ---------------------------------------------------------------- LEFT: the search part
with left:
    text_query = st.text_input("Describe what you want (optional)", placeholder="e.g. black sports shoes")
    uploaded = st.file_uploader("Or upload a photo (optional)", type=["jpg", "jpeg", "png"])
    colour = st.selectbox("Colour (optional)", ["Any"] + COLOURS)
    search_clicked = st.button("Search")

    if uploaded and search_clicked:
        st.image(Image.open(uploaded).convert("RGB"), caption="Your photo", width=180)
        uploaded.seek(0)

# ---------------------------------------------------------------- RIGHT: summary + products
with right:
    if not search_clicked:
        st.caption("Your results will appear here.")
    else:
        files = None
        if uploaded:
            files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}
        data = {"text": text_query, "colour": "" if colour == "Any" else colour}
        try:
            resp = requests.post(f"{API_URL}/search", data=data, files=files, timeout=180)
        except requests.RequestException as e:
            st.error(f"Could not reach the backend: {e}")
            st.stop()

        if resp.status_code != 200:
            try:
                st.warning(resp.json().get("detail", resp.text))
            except ValueError:
                st.warning(resp.text)
        else:
            body = resp.json()
            if body.get("summary"):
                st.info(body["summary"])
            else:
                st.caption("(AI summary is off: no GEMINI_API_KEY was set for the backend. "
                           "The results below still work.)")
            st.caption(f"Mode: {body['mode']}  |  {body['latency_ms']:.0f} ms")

            columns = st.columns(5)
            for col, item in zip(columns, body["results"]):
                col.image(f"{API_URL}{item['image_url']}", width=130)
                col.caption(f"{item['title']}\n\nMatch score: {item['score']:.2f}")