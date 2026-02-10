import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Legal AI – Speech Enabled",
    page_icon="⚖️",
    layout="wide"
)

# --------------------------------------------------
# UI STYLE
# --------------------------------------------------
st.markdown("""
<style>
html, body, .stApp {
    background-color: #0b0f19;
    color: #e5e7eb;
}
h1, h2 { color: #e5e7eb; }
input, textarea {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border: 1px solid #6366f1 !important;
}
.answer {
    background: #020617;
    border-left: 4px solid #6366f1;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 12px;
}
.source {
    font-size: 14px;
    color: #94a3b8;
}
.btn {
    background: #6366f1;
    color: white;
    padding: 8px 14px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1 style='text-align:center;'>⚖️ Legal AI – Speech Enabled Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Streamlit Cloud Safe • Browser Voice</p>", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📂 Upload Legal PDFs")
files = st.sidebar.file_uploader(
    "Upload PDF files (max 5)",
    type=["pdf"],
    accept_multiple_files=True
)

# --------------------------------------------------
# TEXT SPLITTER
# --------------------------------------------------
def split_text(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# --------------------------------------------------
# VECTOR INDEX (SAFE)
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_index(files):
    texts, sources = [], []

    for f in files:
        reader = PdfReader(f)
        full_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "

        if not full_text.strip():
            continue

        chunks = split_text(full_text)
        texts.extend(chunks)
        sources.extend([f.name] * len(chunks))

    if not texts:
        return None, None, None

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        batch_size=16
    )

    nn = NearestNeighbors(n_neighbors=4, metric="cosine")
    nn.fit(embeddings)

    return nn, texts, sources

# --------------------------------------------------
# BROWSER SPEECH + TTS (CLOUD SAFE)
# --------------------------------------------------
st.markdown("""
<script>
function startDictation() {
    if (!('webkitSpeechRecognition' in window)) {
        alert("Speech recognition not supported in this browser");
        return;
    }
    const recognition = new webkitSpeechRecognition();
    recognition.lang = "en-IN";
    recognition.onresult = function(e) {
        document.getElementById("speech_input").value =
            e.results[0][0].transcript;
    };
    recognition.start();
}

function speakText(text) {
    const msg = new SpeechSynthesisUtterance(text);
    msg.lang = "en-IN";
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(msg);
}
</script>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
if files:
    nn, texts, sources = build_index(files)

    if nn is None:
        st.error("❌ No readable text found in PDFs")
        st.stop()

    col1, col2 = st.columns([6, 1])

    with col1:
        question = st.text_input(
            "Ask a legal question",
            key="speech_input"
        )

    with col2:
        st.markdown(
            '<button class="btn" onclick="startDictation()">🎤 Speak</button>',
            unsafe_allow_html=True
        )

    if question:
        with st.spinner("Searching documents..."):
            q_emb = model.encode([question], show_progress_bar=False)
            _, indices = nn.kneighbors(q_emb)

        st.markdown("## 📌 Relevant Answers")

        combined_answer = ""

        for idx in indices[0]:
            combined_answer += texts[idx] + " "
            st.markdown(
                f"<div class='answer'>{texts[idx]}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='source'>Source: {sources[idx]}</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            f'<button class="btn" onclick="speakText(`{combined_answer[:1500]}`)">🔊 Read Answer</button>',
            unsafe_allow_html=True
        )
else:
    st.info("⬅️ Upload legal PDF documents to begin.")
