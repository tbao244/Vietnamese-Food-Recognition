import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="Nhận diện món ăn Việt Nam",
    page_icon="🍜",
    layout="centered")

#----Custom CSS----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;}

/* Page background */
.stApp {
    background: #faf8f5;}

/* Hide default Streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 1rem;}
            
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 0.25rem;
    line-height: 1.2;}
            
.accent-bar {
    width: 48px;
    height: 3px;
    background: #c0392b;
    margin: 0.75rem auto;
    border-radius: 2px;}
            
.hero p {
    color: #666;
    font-size: 0.95rem;
    font-weight: 300;}

/* ── Food tag pills ── */
.food-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 0.5rem 0;}
            
.food-tag {
    background: rgba(192, 57, 43, 0.07);
    color: #a02315;
    border: 0.5px solid rgba(192, 57, 43, 0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 500;}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1.5px dashed #d1c9c0;
    border-radius: 16px;
    padding: 1.5rem;
    transition: border-color 0.2s;}
            
[data-testid="stFileUploader"]:hover {
    border-color: #c0392b;}
            
[data-testid="stFileUploader"] label {
    font-size: 0.9rem;
    color: #555;}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 0.5px solid #e0d8d0;}

/* ── Result card: prediction ── */
.result-card {
    background: #fff;
    border-radius: 16px;
    border: 0.5px solid #e0d8d0;
    overflow: hidden;
    margin-top: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    backdrop-filter: blur(10px);}
            
.result-header {
    background: #c0392b;
    padding: 1.1rem 1.4rem;}
            
.result-header h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #fff;
    font-weight: 600;
    margin: 0 0 2px;}
            
.result-header p {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.75);
    margin: 0;}
            
.result-body {
    padding: 1.2rem 1.4rem;}
            
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;}
            
.conf-label {
    font-size: 0.82rem;
    color: #888;}
            
.conf-value {
    font-size: 0.9rem;
    font-weight: 500;
    color: #1a1a1a;}
            
.conf-track {
    height: 6px;
    background: #f0ece8;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 1.2rem;}
            
.conf-fill {
    height: 100%;
    background: #c0392b;
    border-radius: 3px;
    transition: width 0.8s ease;}

/* ── Top predictions mini-bars ── */
.top-preds-label {
    font-size: 0.72rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.6rem;}
            
.pred-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;}
            
.pred-name {
    font-size: 0.8rem;
    color: #555;
    width: 110px;
    flex-shrink: 0;}
            
.pred-track {
    flex: 1;
    height: 4px;
    background: #f0ece8;
    border-radius: 2px;
    overflow: hidden;}
            
.pred-fill {
    height: 100%;
    background: rgba(192, 57, 43, 0.35);
    border-radius: 2px;}
            
.pred-pct {
    font-size: 0.75rem;
    color: #aaa;
    width: 36px;
    text-align: right;}

/* ── Expander styling ── */
[data-testid="stExpander"] {
    border: 0.5px solid #e0d8d0 !important;
    border-radius: 12px !important;
    background: #fff !important;}
            
[data-testid="stExpander"] summary {
    font-size: 0.85rem;
    color: #666;}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #c0392b !important;}

/* ── Disclaimer ── */
.disclaimer {
    text-align: center;
    font-size: 0.75rem;
    color: #bbb;
    margin-top: 2rem;
    padding: 0.75rem;
    border-top: 0.5px solid #e8e0d8;}
            
</style>
""", unsafe_allow_html=True)


# ---- Constants ----
CLASS_NAMES = [
    'Banh mi', 'Banh xeo', 'Bun bo Hue', 'Com tam', 'Goi cuon', 
    'Hu tieu', 'Mi quang', 'Nem chua', 'Pho', 'Xoi xeo'
]

# ---- Hero header ----
st.markdown("""
<div class="hero">
    <h1>🍜 Nhận Diện Món Ăn Việt Nam</h1>
    <div class="accent-bar"></div>
    <p>Upload một bức ảnh đồ ăn và để AI đoán xem đó là món gì nhé!</p>
</div>
""", unsafe_allow_html=True)

# ---- Supported foods accordion ----
with st.expander("📌 Xem danh sách các món AI có thể nhận diện"):
    tags_html = "".join(f'<span class="food-tag">{name}</span>' for name in CLASS_NAMES)
    st.markdown(f"""
    <div class="food-tags">{tags_html}</div>
    <p style="font-size:0.78rem;color:#aaa;margin-top:8px">
        Hiện tại hỗ trợ <strong>10 món ăn</strong> truyền thống Việt Nam.
    </p>
    """, unsafe_allow_html=True)

# ---- Model loader ----
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'models', 'food_model.h5')
    return tf.keras.models.load_model(model_path)
model = load_model()

# ---- File uploader ----
uploaded_file = st.file_uploader(
    "Chọn một bức ảnh đồ ăn...",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    st.image(image, caption="Ảnh bạn vừa tải lên", use_column_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("⏳ AI đang phân tích..."):
        predictions = model.predict(img_array, verbose=0)

    probs = predictions[0]
    top_idx = np.argsort(probs)[::-1]
    predicted_class = CLASS_NAMES[top_idx[0]]
    confidence = float(probs[top_idx[0]])

    top3_rows = ""
    for idx in top_idx[:3]:
        idx = int(idx)
        pct = float(probs[idx]) * 100
        top3_rows += f"""
        <div class="pred-row">
            <span class="pred-name">{CLASS_NAMES[idx]}</span>
            <div class="pred-track">
                <div class="pred-fill" style="width:{pct:.1f}%"></div>
            </div>
            <span class="pred-pct">{pct:.1f}%</span>
        </div>
        """

    conf_pct = round(confidence * 100, 2)
    conf_bar_w = int(round(confidence * 100))

    st.markdown(f"""
    <div class="result-card">
        <div class="result-header">
            <h2>{predicted_class}</h2>
            <p>🍲 Dự đoán món ăn</p>
        </div>
        <div class="result-body">
            <div class="conf-row">
                <span class="conf-label">📊 Độ tin cậy</span>
                <span class="conf-value">{conf_pct}%</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{conf_bar_w}%"></div>
            </div>
            <div class="top-preds-label">Top dự đoán</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(top3_rows, unsafe_allow_html=True)

# ---- Footer ----
st.markdown("""
<div class="disclaimer">
    10 món ăn Việt Nam truyền thống
</div>
""", unsafe_allow_html=True)
