import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pickle
from pathlib import Path
from utils import extract_ripeness, extract_texture, detect_defects
import time

# --- PATH CONFIGURATION (WITH ERROR HANDLING) ---
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "fruit_classifier.h5"
LABEL_MAP_PATH = BASE_DIR / "model" / "label_map.pkl"

# --- MODEL VERIFICATION (ENHANCED) ---
def verify_model_files():
    """Check if model files exist with detailed error messages"""
    errors = []
    
    if not MODEL_PATH.exists():
        errors.append(f"""
        🚨 **Critical Error**: Model file not found at:  
        `{MODEL_PATH}`  
        → Solution: Run `python training/train_model.py`
        """)
    
    elif MODEL_PATH.stat().st_size < 1024000:  # 1MB
        errors.append(f"""
        🚑 **Corrupted Model**: File too small ({MODEL_PATH.stat().st_size/1024:.1f}KB)  
        → Solution: Delete and retrain with `python training/train_model.py`
        """)
    
    if not LABEL_MAP_PATH.exists():
        errors.append(f"""
        📛 **Label Map Missing**:  
        `{LABEL_MAP_PATH}`  
        → Solution: Recreate during model training
        """)
    
    if errors:
        st.error("\n\n".join(errors))
        st.stop()

verify_model_files()

# --- UI CONFIG (PROFESSIONAL THEME) ---
st.set_page_config(
    page_title="Fruit Quality AI", 
    layout="centered", 
    page_icon="🥭",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "🍍 AI-powered Fruit Quality Grading System v2.0"
    }
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
<style>
    .stApp { background: #f9fff6; }
    .upload-box { 
        border: 2px dashed #4CAF50 !important; 
        border-radius: 10px; 
        padding: 2rem; 
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card { 
        background: white; 
        border-radius: 10px; 
        padding: 1rem; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        margin-bottom: 1rem;
    }
    .grade-A { color: #2e7d32; font-weight: 800; }
    .grade-B { color: #f9a825; font-weight: 800; }
    .grade-C { color: #c62828; font-weight: 800; }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("🍓 AI Fruit Quality Inspector")
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="color: #666; font-size: 1.1rem;">
    Upload high-quality fruit images for instant quality analysis
    </p>
</div>
""", unsafe_allow_html=True)

# --- IMAGE UPLOAD (COMPATIBLE DESIGN) ---
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "**Drag & Drop Fruit Image Here**",
    type=["jpg", "png", "jpeg"],
    help="Supported formats: JPG, PNG, JPEG (Max 5MB)",
    key="uploader"
)
st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN PROCESSING (WITH LOADING STATES) ---
if uploaded_file:
    with st.spinner("🔍 Analyzing your fruit..."):
        try:
            # --- IMAGE PROCESSING ---
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unsupported or corrupted image format")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display uploaded image with border
            st.image(img, caption="Uploaded Image", use_column_width=True, output_format="PNG")
            
            # --- MODEL PREDICTION ---
            model = load_model(str(MODEL_PATH))
            with open(LABEL_MAP_PATH, "rb") as f:
                label_map = pickle.load(f)
            
            # Preprocess and predict
            img_processed = cv2.resize(img, (128, 128)) / 255.0
            pred = model.predict(np.expand_dims(img_processed, axis=0), verbose=0)
            fruit = label_map[np.argmax(pred[0])].capitalize()
            confidence = round(np.max(pred) * 100, 2)
            
            # --- QUALITY ANALYSIS ---
            ripeness = extract_ripeness(img)
            contrast, homogeneity = extract_texture(img)
            defects = detect_defects(img)
            
            # --- RESULTS DISPLAY (MODERN CARDS) ---
            st.subheader("📊 Quality Report", divider="green")
            
            # Metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Fruit Type</h3>
                    <p style="font-size: 1.5rem;">🍎 {fruit}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Confidence</h3>
                    <p style="font-size: 1.5rem;">🔮 {confidence}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Ripeness</h3>
                    <p style="font-size: 1.5rem;">🌡️ {ripeness}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Quality grade with visual indicators
            if defects < 5 and ripeness > 70:
                grade_class = "grade-A"
                grade_emoji = "🌟🌟🌟"
                grade = "A (Premium Quality)"
            elif defects < 10:
                grade_class = "grade-B"
                grade_emoji = "🌟🌟"
                grade = "B (Good Quality)"
            else:
                grade_class = "grade-C"
                grade_emoji = "🌟"
                grade = "C (Commercial Grade)"
            
            st.markdown(f"""
            <div class="metric-card" style="margin-top: 1rem;">
                <h2>Final Quality Grade</h2>
                <p class="{grade_class}" style="font-size: 2rem;">
                    {grade} {grade_emoji}
                </p>
                <p>Defect Density: {defects}% | Texture: Contrast {contrast:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visual quality meter
            st.progress(ripeness/100, text=f"Ripeness Meter: {ripeness}%")
            
            # Detailed analysis expander
            with st.expander("🔬 Detailed Analysis Metrics"):
                st.markdown(f"""
                - **Homogeneity**: {homogeneity:.2f}  
                - **Color Variance**: {contrast:.2f}  
                - **Defect Areas**: {defects}%  
                - **Confidence Scores**:  
                    - {fruit}: {confidence}%  
                """)
            
        except Exception as e:
            st.error(f"""
            ## ❌ Analysis Failed
            **Error Details**:  
            `{str(e)}`  
            
            ### Troubleshooting:
            1. Use a clear, well-lit image
            2. Ensure the fruit is centered
            3. Try a different file format
            """)
            st.image("https://i.imgur.com/YJqjYfL.png", width=300)  # Error illustration

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #888;">
    <hr style="border: 0.5px solid #eee;">
    <p>🍍 FruitAI v2.0 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)