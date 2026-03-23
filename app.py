# VisionAI — Object Detection (YOLOv8 + Streamlit)
# =========================================================

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="VisionAI · Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* ===== BASE ===== */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #070d1a !important;
    color: #d8dce8 !important;
    font-family: Arial, sans-serif !important;
    font-size: 16px !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 65% 40% at 10% 0%, rgba(0,160,255,0.09) 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 90% 100%, rgba(100,50,255,0.10) 0%, transparent 55%),
        #070d1a !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"],
#MainMenu, footer { display:none !important; visibility:hidden !important; }
[data-testid="stMainBlockContainer"] { padding-top: 0.5rem !important; }

/* ===== HERO ===== */
.hero { text-align:center; padding:2.5rem 1rem 1.5rem; }
.hero-badge {
    display:inline-block;
    background:rgba(0,160,255,0.10);
    border:1px solid rgba(0,160,255,0.35);
    color:#00a8ff;
    font-size:13px; letter-spacing:3px; text-transform:uppercase;
    padding:5px 18px; border-radius:100px; margin-bottom:18px;
    font-weight:600;
}
.hero-title {
    font-family: Arial Black, Arial, sans-serif !important;
    font-size: clamp(48px, 7vw, 88px) !important;
    font-weight: 900 !important;
    color: #ffffff !important;
    letter-spacing: -2px !important;
    line-height: 1 !important;
    margin-bottom: 14px !important;
}
.hero-title em { color:#00a8ff; font-style:normal; }
.hero-sub {
    color:rgba(216,220,232,0.45);
    font-size:15px; max-width:460px;
    margin:0 auto 18px; line-height:1.9;
}

/* ===== SECTION LABEL ===== */
.sec-label {
    font-size:12px; letter-spacing:4px; text-transform:uppercase;
    color:#00a8ff; font-weight:700;
    margin-bottom:10px; display:flex; align-items:center; gap:8px;
}
.sec-label::after { content:''; flex:1; height:1px; background:rgba(0,160,255,0.18); }

/* ===== FILE UPLOADER — dark theme ===== */
/* Target every wrapper layer Streamlit generates */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] > div > div,
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] > div,
[data-testid="stFileUploadDropzone"] > div > div {
    background: #0d1a2e !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed rgba(0,160,255,0.35) !important;
    padding: 28px 20px !important;
}
/* All text inside uploader */
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] label {
    color: rgba(216,220,232,0.70) !important;
    font-size: 15px !important;
    font-family: Arial, sans-serif !important;
}
/* Browse files button */
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploadDropzone"] button span {
    background: #00a8ff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 8px 20px !important;
    font-family: Arial, sans-serif !important;
    letter-spacing: 1px !important;
}
[data-testid="stFileUploadDropzone"] button:hover {
    background: #0090e0 !important;
}
/* Uploaded file row */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] > div {
    background: rgba(0,160,255,0.08) !important;
    border: 1px solid rgba(0,160,255,0.20) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] p,
[data-testid="stFileUploaderFile"] div {
    color: rgba(216,220,232,0.75) !important;
    font-size: 14px !important;
}
/* Delete button (×) */
[data-testid="stFileUploaderDeleteBtn"] button {
    background: rgba(255,80,80,0.15) !important;
    border: 1px solid rgba(255,80,80,0.30) !important;
    border-radius: 6px !important;
    color: #ff6060 !important;
}

/* ===== STAT CARDS ===== */
.stat-row { display:flex; gap:14px; margin:20px 0 14px; }
.stat-card {
    flex:1; background:#0d1a2e;
    border:1px solid rgba(255,255,255,0.07);
    border-top:3px solid #00a8ff;
    border-radius:14px; padding:20px 10px; text-align:center;
}
.stat-n {
    font-family: Arial Black, Arial, sans-serif !important;
    font-size: 48px !important;
    font-weight: 900 !important;
    color: #00a8ff !important;
    line-height: 1 !important;
    display: block !important;
    margin-bottom: 8px !important;
}
.stat-l {
    font-size: 11px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: rgba(216,220,232,0.40) !important;
    font-family: Arial, sans-serif !important;
}

/* ===== TAGS ===== */
.tags { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.tag {
    background:rgba(0,160,255,0.08);
    border:1px solid rgba(0,160,255,0.25);
    color:#00a8ff; font-size:13px; letter-spacing:1px;
    padding:5px 14px; border-radius:100px;
    font-family:Arial,sans-serif;
}

/* ===== SUCCESS ALERT ===== */
[data-testid="stAlert"] {
    background:rgba(0,160,255,0.08) !important;
    border:1px solid rgba(0,160,255,0.25) !important;
    border-radius:10px !important;
    color:#d8dce8 !important; font-size:15px !important;
}

/* ===== IMAGES ===== */
[data-testid="stImage"] img {
    border-radius:14px !important;
    border:1px solid rgba(255,255,255,0.07) !important;
}

/* ===== SPINNER ===== */
[data-testid="stSpinner"] p {
    color:#00a8ff !important; font-size:15px !important;
}

/* ===== PLACEHOLDER ===== */
.placeholder {
    height:300px; border-radius:14px;
    border:2px dashed rgba(255,255,255,0.08);
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    color:rgba(216,220,232,0.20); gap:12px;
    font-size:13px; letter-spacing:3px; text-align:center;
}

/* ===== DIVIDER & FOOTER ===== */
hr { border:none !important; border-top:1px solid rgba(255,255,255,0.06) !important; margin:24px 0 !important; }
.footer {
    text-align:center; color:rgba(216,220,232,0.22);
    font-size:13px; letter-spacing:1px; padding-bottom:20px;
}
.footer b { color:#00a8ff; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# Model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


# Custom PIL annotation renderer
def draw_annotations(image_pil, results):
    img     = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    result = results[0]
    boxes  = result.boxes
    names  = result.names

    palette = [
        (0,   168, 255),
        (120,  80, 255),
        (0,   210, 140),
        (255, 170,   0),
        (255,  60, 110),
        (0,   200, 200),
        (255, 120,  40),
    ]

    img_w, _ = img.size
    font_size = max(20, int(img_w * 0.030))

    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except Exception:
            font = ImageFont.load_default()

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = f"  {names[cls_id]}  {conf:.0%}  "
        color  = palette[cls_id % len(palette)]

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color + (230,), width=3)

        # Label dimensions
        bbox = draw.textbbox((0, 0), label, font=font)
        lw   = bbox[2] - bbox[0]
        lh   = bbox[3] - bbox[1]
        pad  = 5

        lx1 = x1
        ly1 = y1 - lh - pad * 2
        lx2 = x1 + lw
        ly2 = y1

        if ly1 < 0:          # flip below box if too close to top
            ly1 = y2
            ly2 = y2 + lh + pad * 2

        draw.rectangle([lx1, ly1, lx2, ly2], fill=color + (215,))
        draw.text((lx1 + 4, ly1 + pad // 2), label,
                  fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")


# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ YOLOv8 &nbsp;·&nbsp; COCO &nbsp;·&nbsp; 80 Classes</div>
    <div class="hero-title">Vision <em>AI</em></div>
    <div class="hero-sub">
        Real-time object detection powered by neural networks.<br>
        Upload any image and watch the model identify what's inside.
    </div>
</div>
""", unsafe_allow_html=True)

# Layout
col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.markdown('<div class="sec-label">Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop image here or click Browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="sec-label" style="margin-top:16px">Original</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

with col_r:
    if uploaded_file:
        image_np = np.array(image)

        st.markdown('<div class="sec-label">Detection Result</div>', unsafe_allow_html=True)

        with st.spinner("Running inference..."):
            results = model(image_np, verbose=False)

        annotated = draw_annotations(image, results)
        st.image(annotated, use_container_width=True)

        result   = results[0]
        boxes    = result.boxes
        names    = result.names
        num      = len(boxes)
        detected = []
        confs    = []

        for box in boxes:
            detected.append(names[int(box.cls[0])])
            confs.append(float(box.conf[0]))

        unique   = sorted(set(detected))
        avg_conf = (sum(confs) / len(confs) * 100) if confs else 0

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <span class="stat-n">{num}</span>
                <div class="stat-l">Objects Found</div>
            </div>
            <div class="stat-card">
                <span class="stat-n">{len(unique)}</span>
                <div class="stat-l">Unique Classes</div>
            </div>
            <div class="stat-card">
                <span class="stat-n">{avg_conf:.0f}%</span>
                <div class="stat-l">Avg Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if unique:
            st.markdown('<div class="sec-label" style="margin-top:8px">Detected Classes</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="tags">' +
                ''.join(f'<span class="tag">● {c}</span>' for c in unique) +
                '</div>',
                unsafe_allow_html=True
            )

        st.success(f"✅  Detection complete — {num} object(s) identified.")

    else:
        st.markdown("""
        <div class="placeholder">
            <div style="font-size:36px">🎯</div>
            <div>AWAITING INPUT<br>
            <span style="font-size:11px;opacity:0.45;letter-spacing:2px">
                Upload an image to begin
            </span></div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer">'
    'Built with <b>YOLOv8</b> · Ultralytics · Streamlit &nbsp;·&nbsp; Model: COCO (80 classes)'
    '</div>',
    unsafe_allow_html=True
)