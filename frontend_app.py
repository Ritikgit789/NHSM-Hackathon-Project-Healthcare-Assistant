import os
import tempfile
import base64
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image, Audio
import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import sqlite3
import pandas as pd


# Set page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="wide")

# Load environment variables and configure APIs
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Define agents (as in the original code)
web_agent=Agent(
    name="Web Agent",
    description=(
        "You are a specialized digital research assistant tasked with searching the web for the most current and relevant information "
        "In response to user queries. You utilize trusted tools to gather accurate data and ensure that every piece of information is properly cited."
    ),
    role="search the web and provide source-cited, accurate information",
    model=Groq(id="qwen-2.5-32b"),
    tools=[DuckDuckGoTools()],
    instructions=(
        "When processing a query, please perform the following steps:\n"
        "- Utilize the web search tool to retrieve the most up-to-date and accurate information related to the query.\n"
        "- Provide the key findings clearly, ensuring that your response is informative and easy to understand.\n"
        "- Always include inline source citations for every claim or fact mentioned, to maintain transparency and trustworthiness."
    ),
    show_tool_calls=True,
    markdown=True,
)

image_agent = Agent(
    name="Image Analyzer",
    description="Digital healthcare assistant trained to analyze images and give suggestions.",
    role="analyze medical images and provide detailed insights & medicines on visible conditions",
    model=Gemini(id="gemini-2.0-flash", grounding=True),
     instructions=(
        "When analyzing an image, please do the following:\n"
        "- Clearly describe the observable features and any abnormalities in the image.\n"
        "- Provide detailed information on the potential diseases or conditions associated with these features, "
        "- Including possible causes and relevant characteristics.\n"
        "- Focus solely on delivering an informative analysis based on the image content without including any disclaimers.\n"
        "If no user query is provided, focus solely on delivering an informative analysis based on the image content without including any disclaimers."
    ),
    show_tool_calls=True,
    markdown=True,
)

audio_agent=Agent(
    name="Audio Agent",
    description=(
        "You are a specialized digital healthcare assistant trained to analyze audio. "
        "Your task is to identify observable signs of diseases or conditions and provide detailed, factual insights about them."
    ),
    role="analyze medical audio and provide detailed insights on audible conditions",
    model=Gemini(id="gemini-2.0-flash",grounding=True),
    instructions=(
        "When analyzing an audio, please do the following:\n"
        "- Clearly describe the observable features and any abnormalities mentioned or observed in the audio.\n"
        "- Provide detailed information on the potential diseases or conditions associated with these features, "
        "- Including possible causes and relevant characteristics.\n"
        "- Focus solely on delivering an informative analysis based on the audio content without including any disclaimers."
    ),
    show_tool_calls=True,
    markdown=True,
)

# Group agents into a team
agent_team = Agent(
    team=[web_agent, image_agent, audio_agent],
    model=Gemini(id="gemini-2.0-flash"),
    instructions=["Respond only with healthcare-related insights."],
    markdown=True,
)

# Sidebar for mode selection

st.sidebar.title("🩺 Choose an Option")
mode = st.sidebar.radio("Select mode:", ("Text Query", "Image Analysis", "Audio Analysis",))

# Display app title and description
st.title("🔬 HealSense AI: AI Chatbot for Diagnosis")
st.markdown(
    """
    Welcome to AI Healthcare Chatbot! 🔍
Your intelligent assistant for real-time healthcare insights.

✅ Ask health-related queries and get instant answers.
✅ Upload medical images or audio for AI-powered analysis as well as doctors.
✅ Get accurate, data-driven insights to support better decisions.

Smart -> Fast -> Reliable. Your AI-powered healthcare companion. 🏥
    
    """
)

# ✅ Ensure session states
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None

# st.sidebar.title("🩺 Choose an Option")
# app_mode = st.sidebar.radio("Select a feature:", ("Doctor Search"))




@st.cache_data
def load_doctor_data():
    return pd.read_csv("D:/NHSM Hackathon/Doctor list.csv")  # Ensure "doctors.csv" is in the same folder

doctors_df = load_doctor_data()

# ✅ Disease-to-Specialist Mapping (Customize as Needed)
disease_specialty_mapping = {
    "skin rash": "Dermatologist",
    "skin infection":"Dermatologist",
    "allergy": "Dermatologist",
    "Psoriasis": "Dermatologist",
    "acne": "Dermatologist",
    "throat infection": "ENT",
    "Depression": "Psychriatist",
    "Overthinking":"Psychriatist",
    "ear pain": "ENT",
    "heart attack": "Cardiologist",
    "chest pain": "Cardiologist",
    "cancer": "Oncologist",
    "lung cancer": "Oncologist",
    "brain tumor": "Neurologist",
    "stroke": "Neurologist",
    "broken bone": "Orthopedic",
    "joint pain": "Orthopedic",
}

# ✅ Function to find doctors based on the detected specialty
def get_doctors_by_specialty(specialty):
    matching_doctors = doctors_df[doctors_df["Speciality"].str.contains(specialty, case=False, na=False)]
    return matching_doctors if not matching_doctors.empty else None

# ✅ Function to extract the most relevant specialty from AI response
def detect_specialty_from_response(response_content):
    for disease, specialty in disease_specialty_mapping.items():
        if disease in response_content.lower():
            return specialty
    return None  # If no match is found

# ✅ Function to display recommended doctors
def display_doctor_profiles(doctors):
    """ Show doctor details in a white box with blue text. """
    if doctors is None:
        st.warning("⚠️ No doctors found for this condition.")
        return

    st.markdown(
        "<div style='border: 2px solid blue; border-radius: 10px; padding: 15px; background-color: white; color: blue;'>"
        "<h4>👨‍⚕️ Recommended Doctors:</h4>", unsafe_allow_html=True)

    for _, doc in doctors.iterrows():
        name, specialty, hospital, profile_link = doc["Doctors"], doc["Speciality"], doc["Hospital"], doc["Profile"]

        # ✅ Clickable profile link
        profile_display = f"<a href='{profile_link}' target='_blank' style='color: white;'>🌐 View Profile</a>" if pd.notna(profile_link) else ""

        # ✅ Clickable hospital link
        hospital_display = f"<a href='{hospital}' target='_blank' style='color: white;'>🏥 {hospital}</a>" if pd.notna(hospital) and hospital.startswith("http") else f"🏥 {hospital}"

        st.markdown(
            f"<p><strong>{name}</strong> - {specialty} <br>"
            f"{hospital_display} | {profile_display}</p>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# # Initialize paths
image_path = None
audio_path = None

# # image_file = st.file_uploader("Choose an image (X-ray, MRI, CT scan, etc.)", type=["jpg", "png", "jpeg"])

# Query input for text mode
import time  # For simulating "thinking"

# Text Query Mode
# ✅ Process User Query + Fetch Doctors
if mode == "Text Query":
    st.subheader("➕ Ask Your Healthcare Question")
    user_question = st.text_input("What would you like to know about health issues?")

    if st.button("Submit Query"):
        if user_question:
            with st.spinner("🤔 Thinking... Please wait..."):
                time.sleep(2)  # Simulate AI processing

                # ✅ Get AI Response
                response = web_agent.run(user_question)
                response_content = response.content

                # ✅ Detect disease from AI response
                detected_specialty = detect_specialty_from_response(response_content)

                # ✅ Display AI Response
                st.markdown(
                    "<div style='border: 2px solid green; border-radius: 10px; padding: 10px; background-color: #007BFF; color: white;'>"
                    f"<strong>🤖 Medmind AI:</strong><br>{response_content}</div>",
                    unsafe_allow_html=True
                )

                # ✅ Fetch and Display Doctor Recommendations
                if detected_specialty:
                    recommended_doctors = get_doctors_by_specialty(detected_specialty)
                    display_doctor_profiles(recommended_doctors)
                else:
                    st.info("⚠️ No matching doctor found for this condition.")

        else:
            st.error("⚠️ Please enter a query before submitting!")


###################################

elif mode == "Image Analysis":
   def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

# ✅ Function to analyze image using Gemini AI
def analyze_with_gemini(image_path, query):
    encoded_image = encode_image(image_path)
    if not encoded_image:
        raise ValueError("Failed to encode the image. Please check the file.")

    # ✅ Send image to Gemini AI
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([
        {"text": query},  
        {"inline_data": {"mime_type": "image/png", "data": encoded_image}}  
    ])

    return response.text if response and hasattr(response, 'text') else None

# ✅ Function to detect disease from AI response
def detect_specialty_from_response(response_content):
    for disease, specialty in disease_specialty_mapping.items():
        if disease in response_content.lower():
            return specialty
    return None

# ✅ Image Analysis UI in Streamlit
if mode == "Image Analysis":
    st.subheader("📷 Image Analysis")

    # ✅ Upload image
    image_file = st.file_uploader("Upload a medical image (X-ray, MRI, CT scan):", type=["jpg", "png", "jpeg"])

    if image_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.name)[1]) as temp_image:
                temp_image.write(image_file.getbuffer())
                image_path = temp_image.name

            # ✅ Display uploaded image
            st.image(image_file, caption="✅ Image Uploaded Successfully", use_container_width=True)

            # ✅ Get user query
            user_query = st.text_input("Enter your query:")
            if not user_query.strip():
                user_query = "Please analyze this medical image and provide insights, medicines and fetch doctor list and give some doctors prfile according to image disease."

            # ✅ Analyze the image when button is clicked
            if st.button("Analyze Image"):
                with st.spinner("🔎 Analyzing image... Please wait..."):
                    result = analyze_with_gemini(image_path, user_query)

                    if result:
                        # ✅ Detect specialty from AI response
                        detected_specialty = detect_specialty_from_response(result)

                        # ✅ Display AI Response
                        st.markdown(
                            f"<div style='border: 2px solid green; border-radius: 10px; "
                            f"padding: 10px; background-color: #ff1343;'>"
                            f"<strong>🤖 Medmind AI:</strong><br>{result}</div>",
                            unsafe_allow_html=True
                        )

                        # ✅ Fetch and Display Doctor Recommendations
                        if detected_specialty:
                            recommended_doctors = get_doctors_by_specialty(detected_specialty)
                            display_doctor_profiles(recommended_doctors)
                        else:
                            st.info("Doctors not in the list, but ")
                    else:
                        st.warning("⚠️ No valid response from the AI.")

                # ✅ Clean up temporary file
                os.unlink(image_path)

        except Exception as e:
            st.error(f"🚨 File Processing Error: {e}")


##########################################

## ✅ Function to encode audio
def encode_audio(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error encoding audio: {e}")
        return None

# ✅ Function to analyze audio using Gemini AI
def analyze_audio_with_gemini(audio_path, query):
    encoded_audio = encode_audio(audio_path)
    if not encoded_audio:
        raise ValueError("Failed to encode the audio file.")

    # ✅ Send audio to Gemini AI
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([
        {"text": query},  
        {"inline_data": {"mime_type": "audio/wav", "data": encoded_audio}}  
    ])

    return response.text if response and hasattr(response, 'text') else None

# ✅ Function to detect disease from AI response
def detect_specialty_from_response(response_content):
    for disease, specialty in disease_specialty_mapping.items():
        if disease in response_content.lower():
            return specialty
    return None

# ✅ Ensure `audio_path` is stored in session state
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None

# ✅ Audio Analysis UI in Streamlit
elif mode == "Audio Analysis":
    st.sidebar.write("🎤 Record or Upload Audio")

    # ✅ Audio recording controls
    duration = st.sidebar.slider("🎙️ Select Recording Duration (seconds):", 3, 10, 5)
    record_button = st.sidebar.button("🎙️ Start Recording")

    # ✅ Handle recording
    if record_button:
        st.write("🎙️ Recording...")
        recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype="int16")
        sd.wait()
        st.write("✅ Recording complete!")

        # ✅ Save the recording and store path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            wav.write(temp_audio.name, 44100, recording)
            st.session_state["audio_path"] = temp_audio.name  # Store in session state

        st.success("🎤 Audio recorded successfully!")

    # ✅ File uploader for existing audio
    audio_file = st.file_uploader("📂 Upload an audio file (.wav)", type=["wav"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            st.session_state["audio_path"] = temp_audio.name  # Store uploaded file path
        st.success("✅ Audio file uploaded successfully!")

    # ✅ Query input
    query = st.text_area("📝 Enter a related query (optional):")
    if not query.strip():
        query = "Analyze this audio and provide insights, disease name, and recommended doctors."

    # ✅ Analyze Audio Button
    if st.button("🔍 Analyze Audio"):
        if st.session_state["audio_path"]:  # ✅ Use stored audio file
            try:
                response = analyze_audio_with_gemini(st.session_state["audio_path"], query)

                if response:
                    # ✅ Detect specialty from AI response
                    detected_specialty = detect_specialty_from_response(response)

                    # ✅ Display AI Response
                    st.markdown(
                        f"<div style='border: 2px solid green; border-radius: 10px; "
                        f"padding: 10px; background-color: #ff1343;'>"
                        f"<strong>🤖 Medmind AI:</strong><br>{response}</div>",
                        unsafe_allow_html=True
                    )

                    # ✅ Fetch and Display Doctor Recommendations
                    if detected_specialty:
                        recommended_doctors = get_doctors_by_specialty(detected_specialty)
                        display_doctor_profiles(recommended_doctors)
                    else:
                        st.info("⚠️ No matching doctor found for this condition.")
                else:
                    st.warning("⚠️ No valid response from AI.")
            except Exception as e:
                st.error(f"🚨 AI Processing Error: {e}")
        else:
            st.warning("⚠️ Please record or upload an audio file before analyzing!")




#########################################################



# Styling with markdown
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(120deg, #f8f9fa, #e9ecef);
            font-family: "sans-serif", poppins;
        }
        h1, h2, h3 {
            color: #343a40;
            text-align: center;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
            font-size: 16px;
        }
        .stSidebar {
            background-color: #ff1324;
            padding: 15px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to encode image in Base64 format for Gemini API
