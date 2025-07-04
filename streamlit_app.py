import streamlit as st
import requests
from PIL import Image
import io
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import json
# Set page config
st.set_page_config(page_title="DR Detection Pro", page_icon="👁️", layout="wide")

# Custom CSS for enhanced dark mode
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5em 2em;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    
    .severity-indicator {
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        color: #FFFFFF;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .severity-indicator:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    
    .footer {
        text-align: center;
        color: #AAAAAA;
        font-size: 0.8em;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #333333;
    }
    
    .stTextInput>div>div>input {
        background-color: #2E2E2E;
        color: #FFFFFF;
        border-radius: 5px;
    }
    
    .stSelectbox>div>div>select {
        background-color: #2E2E2E;
        color: #FFFFFF;
        border-radius: 5px;
    }
    
    .stTextInput>label {
        color: #FFFFFF;
    }
    
    .stSelectbox>label {
        color: #FFFFFF;
    }
    
    .stExpander {
        background-color: #2E2E2E;
        border-radius: 10px;
    }
    
    .stExpander>div>div>div>div {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
lottie_eye = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_xyadoh9h.json")
lottie_upload = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_urbk83vw.json")

# Sidebar
with st.sidebar:
    st.title("DR Detection Pro")
    st_lottie(lottie_eye, height=200)
    st.info(
        "This app uses advanced AI to detect "
        "the presence and severity of Diabetic Retinopathy "
        "from retinal scan images."
    )

# Navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "About DR", "Upload_and_Predict", "FAQ"],
    icons=["house", "info-circle", "upload", "question-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#1E1E1E"},
        "icon": {"color": "#4CAF50", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#2E2E2E"},
        "nav-link-selected": {"background-color": "#2E2E2E"},
    }
)
print("Selected:", selected)
# Function to send image to Flask API and get prediction
def predict_diabetic_retinopathy(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'image': img_byte_arr}
    response = requests.post('http://localhost:5000/predict', files=files)
    print(response.json())  # Print the response dictionary
    if 'error' in response.json():
        return "Error", None
    elif 'prediction' in response.json():
        prediction = response.json()['prediction']
    else:
        return "Unknown", None

    severity_levels = {
        0: "No Diabetic Retinopathy",
        1: "Mild Diabetic Retinopathy",
        2: "Moderate Diabetic Retinopathy",
        3: "Severe Diabetic Retinopathy",
        4: "Proliferative Diabetic Retinopathy"
    }

    return severity_levels.get(prediction, "Unknown"), prediction

# ...




# Home page
if selected == "Home":
    st.title("Welcome to DR Detection Pro")
    st.write(
        "Our cutting-edge application aids in the early detection of Diabetic Retinopathy "
        "using state-of-the-art machine learning techniques. Simply upload your retinal "
        "scan image, and receive an instant assessment of the DR severity level."
    )
    st.image("https://www.aao.org/image.axd?id=46315aff-9197-4c08-b20f-5853977e5d1f&t=637520579278770000", 
             caption="Example of a retinal scan", use_column_width=True)

# About DR page
elif selected == "About DR":
    st.title("Understanding Diabetic Retinopathy")
    st.write(
        "Diabetic retinopathy is a serious complication of diabetes that affects the eyes. "
        "It occurs when high blood sugar levels damage the blood vessels in the retina, "
        "the light-sensitive tissue at the back of the eye."
    )
    st.subheader("Severity Levels:")
    levels = {
        "No DR (0)": "No visible signs of Diabetic Retinopathy",
        "Mild (1)": "Small areas of balloon-like swelling in the retina's tiny blood vessels",
        "Moderate (2)": "As the disease progresses, some blood vessels that nourish the retina become blocked",
        "Severe (3)": "Many more blood vessels are blocked, depriving blood supply to areas of the retina",
        "Proliferative DR (4)": "The most advanced stage, where new blood vessels grow in the retina"
    }
    for level, description in levels.items():
        st.markdown(f"**{level}:** {description}")

    st.subheader("Risk Factors")
    st.write("The main risk factors for developing diabetic retinopathy include:")
    st.markdown("- Duration of diabetes")
    st.markdown("- Poor control of blood sugar levels")
    st.markdown("- High blood pressure")
    st.markdown("- High cholesterol")
    st.markdown("- Pregnancy")
    st.markdown("- Tobacco use")

    st.subheader("Prevention and Management")
    st.write("To prevent or slow the progression of diabetic retinopathy:")
    st.markdown("- Manage your blood sugar levels")
    st.markdown("- Keep your blood pressure and cholesterol under control")
    st.markdown("- Have regular eye exams")
    st.markdown("- Exercise regularly")
    st.markdown("- Quit smoking if you're a smoker")

# Upload & Predict page
elif selected == "Upload_and_Predict":
    print("Upload & Predict page selected")
    st.title("Diabetic Retinopathy Detection")
    st.write("Upload a retinal scan image to detect the presence and severity of Diabetic Retinopathy.")

    uploaded_file = st.file_uploader("Choose a retinal scan image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                diagnosis, severity = predict_diabetic_retinopathy(image)
                if diagnosis == "Error":
                    st.write("Error:", diagnosis)
                else:
                    st.subheader("Results:")
                    st.write(f"Diagnosis: {diagnosis}")
                    
                    # Animated color-coded severity indicator
                    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]
                    severity = int(severity)  # Convert severity to integer
                    st.markdown(f"""
                        <div class="severity-indicator" style="background-color: {colors[severity]};">
                            <h3>Severity Level: {severity}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.write("Please consult with a healthcare professional for a comprehensive evaluation.")
    else:
        st_lottie(lottie_upload, height=300)

# FAQ page
elif selected == "FAQ":
    st.title("Frequently Asked Questions")

    faq_data = [
        ("What is Diabetic Retinopathy?", 
         "Diabetic Retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the retina, the light-sensitive tissue at the back of the eye."),
        ("How accurate is this detection tool?", 
         "While our tool uses advanced AI algorithms, it's not a substitute for professional medical diagnosis. Always consult with an eye care specialist for accurate diagnosis and treatment."),
        ("How often should I get my eyes checked?", 
         "If you have diabetes, you should have a comprehensive dilated eye exam at least once a year, or more frequently if recommended by your eye care professional."),
        ("Can Diabetic Retinopathy be reversed?", 
         "Early stages of Diabetic Retinopathy may be reversible with proper management of blood sugar levels. However, advanced stages may require medical intervention and may not be fully reversible."),
        ("What are the symptoms of Diabetic Retinopathy?", 
         "Symptoms may include blurred vision, dark or empty areas in your vision, difficulty perceiving colors, vision loss, and floaters. However, early stages often have no symptoms.")
    ]

    for question, answer in faq_data:
        with st.expander(question):
            st.write(answer)

# Footer
st.markdown(
"""
<div class="footer">
    <p> 2023 DR Detection Pro. All rights reserved.</p>
    <p>Disclaimer: This tool is for educational purposes only. 
    Always consult with a healthcare professional for medical advice.</p>
    <p><a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></p>
</div>
""", 
unsafe_allow_html=True
)