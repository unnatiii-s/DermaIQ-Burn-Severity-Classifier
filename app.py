import streamlit as st

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="DermaIQ Burn Classifier", page_icon="🔥")

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ---------------------------- CSS Styling ----------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@500&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@600&display=swap');

    /* ✅ White background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        color: #000000;
    }

    .stProgress > div > div > div > div {
        background-color: #FF6B6B;
    }

    footer {visibility: hidden;}

    .sidebar-title {
        font-family: "Raleway", cursive;
        margin-top : 1px;
        margin-bottom : 25px;
        font-size: 26px;
        font-weight: bold;
        background: #F9F9F9;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.3em;
        text-align: center;
    }

    .sidebar-section {
        font-size: 15px;
        color: #212529;
        padding: 5px 0 15px 0;
    }

 .sidebar-box {
    background-color: #ffffff !important;  /* White */
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

    /* Container for buttons at top right corner */
    .top-right-buttons {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 100;
        display: flex;
        gap: 10px;
        color: #000000;
    }

    /* Buttons with white background and dark text */
    .top-right-buttons button,
    .top-right-buttons button[kind="primary"] {
        color: #121212 !important;
        background-color: #ffffff !important;
        border: 2px solid #121212 !important;
        border-radius: 5px !important;
        padding: 6px 12px !important;
        font-weight: 600 !important;
        cursor: pointer;
        transition: background-color 0.3s, color 0.3s;
    }

    .top-right-buttons button:hover,
    .top-right-buttons button[kind="primary"]:hover {
        color: #ffffff !important;
        background-color: #000000 !important;
        border-color: #ffffff !important;
    }

    .top-right-buttons button:focus,
    .top-right-buttons button[kind="primary"]:focus {
        outline: none !important;
        box-shadow: 0 0 5px #121212 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- SIDEBAR ----------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'> DermaIQ Assistant</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-box sidebar-section'>"
                "<b>🩺 What it does:</b><br>"
                "• Classifies burn severity<br>"
                "• Suggests treatment<br>"
                "• Highlights burn region with Grad-CAM<br>"
                "• Displays model performance"
                "</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-box sidebar-section'>"
                "<b>🔥 Burn Severity Classes:</b><br>"
                "🔴 <b>0:</b> First-degree<br>"
                "🟠 <b>1:</b> Second-degree<br>"
                "⚫ <b>2:</b> Third-degree"
                "</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-box sidebar-section'>"
                "<b>📘 Tip:</b> Upload a clear image of the burn on clean skin."
                "</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.link_button("🌐 GitHub", "https://github.com/unnatiii-s/DermaIQ-Burn-Severity-Classifier")
    with col2:
        st.link_button("💬 Feedback", "#")

# ---------------------------- Top-Right Functional Buttons ----------------------------

# Initialize session state if not present
if "show_accuracy" not in st.session_state:
    st.session_state["show_accuracy"] = False
if "show_confusion" not in st.session_state:
    st.session_state["show_confusion"] = False

# Inject CSS to float horizontal buttons to the top-right
st.markdown("""
<style>
.button-container {
    position: fixed;
    top: 80px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: row;
    gap: 12px;
}

.button-container button {
    background-color: white;
    color: #000000;
    border: 2px solid #000000;
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: bold;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.button-container button:hover {
    background-color: #000000;
    color: #ffffff;
}
</style>

<div class="button-container">
    <form action="/" method="get">
        <button type="submit" name="btn" value="accuracy">📊 Accuracy & Loss</button>
        <button type="submit" name="btn" value="confusion">🔁 Confusion Matrix</button>
    </form>
</div>
""", unsafe_allow_html=True)

# ✅ Use updated method to get query parameters
btn = st.query_params.get("btn", None)

# Update session state based on query param
if btn == "accuracy":
    st.session_state["show_accuracy"] = True
    st.session_state["show_confusion"] = False
elif btn == "confusion":
    st.session_state["show_confusion"] = True
    st.session_state["show_accuracy"] = False

# ---------------------------- Display evaluation visuals ----------------------------
if st.session_state["show_accuracy"]:
    st.markdown("### 📈 Model Accuracy & Loss")
    st.image("models/accuracy_loss_plot.png", caption="Model Accuracy vs Loss", use_container_width=True)

if st.session_state["show_confusion"]:
    st.markdown("### 🔄 Confusion Matrix")
    st.image("models/confusion_matrix.png", caption="Model Confusion Matrix", use_container_width=True)


# ---------------------------- Load Model ----------------------------
model = load_model("models/dermaiq_trained_model.h5")
labels = ["First-degree", "Second-degree", "Third-degree"]

# ---------------------------- Treatment Info ----------------------------
treatments = {
    0: """
**🩹 First-Degree Burn Treatment:**
- Soak the burn in cool (not cold) water for 5–10 minutes  
- Apply aloe vera or moisturizing lotion  
- Cover with a sterile non-adhesive bandage  
- Take over-the-counter pain relief like ibuprofen
""",
    1: """
**🧴 Second-Degree Burn Treatment:**
- Cool the burn with water (no ice)  
- Apply antibiotic ointment  
- Use clean, non-stick bandage (change daily)  
- Seek medical help for large/sensitive areas
""",
    2: """
**🚨 Third-Degree Burn Treatment:**
- ⚠️ **Emergency! Call emergency services immediately**  
- Do **not** apply water or ointments  
- Cover with a clean, dry cloth  
- Watch for shock (keep the person warm)  
- Hospital care and surgery may be required
"""
}

# app header
st.markdown("""
<h1 style='text-align: center;
            font-family:"Raleway", cursive;
            font-size: 3rem;
            margin-top: 60px;
            background: -webkit-linear-gradient(45deg, #D72638, #3F88C5, #F49D37);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            margin-bottom: 0.5em;'>
   DermaIQ: Burn Severity Classifier
</h1>
""", unsafe_allow_html=True)


# ---------------------------- Show evaluation images based on button ----------------------------

st.markdown("#### 📤 Upload a burn image to get analysis and treatment suggestion")

# ---------------------------- File Upload ----------------------------
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((224, 224))
    img_array = img_to_array(image_resized)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    confidence = np.max(preds[0]) * 100

    # Show prediction
    st.markdown(f"""
    <div style='color: black; font-weight: bold; font-size: 28px; margin-bottom: 8px;'>
        🎯 Predicted Burn Severity: <span style="font-weight: bold;">{labels[class_index]}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='color: black; font-weight: bold; font-size: 25px; margin-bottom: 12px;'>
        🤖 Confidence: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence / 100)

    # Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer("Conv_1_bn").output, *model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap + tf.keras.backend.epsilon())
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    original_np = np.array(image_resized)
    superimposed = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

        # Suggested Treatment FIRST
    st.markdown("### 🩺 Suggested Treatment")
    st.markdown(f"""
    <div style='background-color:#fff9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd; color: black; font-size: 16px;'>
        {treatments[class_index]}
    </div>
    """, unsafe_allow_html=True)
    st.caption("⚠️ AI-generated guidance. Please consult a medical professional.")

    # THEN show uploaded image & Grad-CAM
    st.markdown("### 📷 Uploaded Image & Grad-CAM Heatmap")

    img_col, cam_col = st.columns(2)

    with img_col:
        st.markdown("**Uploaded Image**")
        st.image(image, width=250)

    with cam_col:
        st.markdown(f"**Grad-CAM Heatmap ({labels[class_index]})**")
        st.image(superimposed_rgb, width=250)