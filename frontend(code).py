import os
import pandas as pd
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
from storage import load_storage
from agno.agent import Agent 
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from tzlocal import get_localzone_name
from datetime import datetime
from agno.tools.googlecalendar import GoogleCalendarTools
from agno.media import Image, Audio
from agno.tools.twilio import TwilioTools
from agno.tools.zoom import ZoomTools
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from agno.playground import Playground, serve_playground_app

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY").strip()
os.environ["GROQ_API_KEY"] = groq_api_key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

zoom_tools = ZoomTools(
    account_id=os.getenv("ZOOM_ACCOUNT_ID"),
    client_id=os.getenv("ZOOM_CLIENT_ID"),
    client_secret=os.getenv("ZOOM_CLIENT_SECRET")
)

# import json
# from google.oauth2 import service_account

# # Load the JSON credentials from Streamlit secrets
# credentials_info = json.loads(st.secrets["google_calendar"]["credentials_json"])

# # Create a credentials object using Google service account info
# credentials = service_account.Credentials.from_service_account_info(credentials_info)

# Load doctor database
doctor_df = pd.read_csv("Doctor list.csv")
doctor_database_str = doctor_df.to_csv(index=False)

# Initialize Text-to-Speech Engine
# engine = pyttsx3.init()

# ---- Define Agents ----
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
    storage=load_storage(),
    read_chat_history=True,
    add_history_to_messages=True,
    num_history_responses= 6
)




image_agent=Agent (
    name="Image Analyzer",
    description=(
        "You are a specialized digital healthcare assistant trained to analyze images. "
        "Your task is to identify observable signs of diseases or conditions and provide detailed, factual insights about them."
    ),
    role="analyze medical images and provide detailed insights on visible conditions",
    model=Gemini(id="gemini-2.0-flash",grounding=True),
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
    storage = load_storage(),
    read_chat_history=True,
    add_history_to_messages=True,
    num_history_responses= 6
)


audio_agent =Agent(
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
    storage = load_storage(), 
    read_chat_history=True,
    add_history_to_messages=True,
    num_history_responses= 6
)



doctor_agent = Agent(
    name="Doctor Agent",
    description=(
        "You are a specialized digital healthcare assistant trained to analyze medical text, image & audio. "
        "Your task is to identify symptoms, diseases, or disorders, and then recommend a relevant doctor from a provided database."
    ),
    role="analyze medical text/audio/image and recommend a doctor based on a CSV database",
    model=Gemini(id="gemini-2.0-flash"),
    instructions=(
        "Below is the doctor database in CSV format:\n"
        "```\n" + doctor_database_str + "\n```\n\n"
        "When analyzing the provided medical text, follow these steps:\n"
        "1. Identify the symptom, disease, or disorder described in the text.\n"
        "2. Determine the corresponding medical specialty (for example, 'Cardiologist', 'Dermatologist', etc.).\n"
        "3. From the database above, select the doctor whose 'Speciality' best matches the inferred specialty.\n"
        "4. Provide the doctor's details including Name, Speciality, Hospital, Profile link, and any Remarks.\n"
        "Ensure your answer is clear and concise, and do not include any disclaimers."
    ),
    show_tool_calls=True,
    markdown=True,
)
location_agent = Agent(
    name="Location Agent",
    description=(
        "You are a specialized assistant for finding nearby locations."
        "Your task is to identify the user's location based on context and suggest relevant places."
    ),
    role="detect user location and find nearby medical facilities",
    model=Gemini(id="gemini-2.0-flash"),
    instructions=(
        "1. If a user asks for nearby hospitals or doctors, infer their location from contextand location.\n"
        "2. Provide relevant locations in a structured format.\n"
        "3. If the location is unclear, ask for clarification subtly.\n"
        "4. Ensure the response is clear, relevant, and user-friendly."
        
    ),
    show_tool_calls=True,
    markdown=True,
)

feedback_agent = Agent() #feedback
twilio_agent = Agent(
    name="Twilio Agent",
    description="You are a specialized digital assistant trained to send notification SMS using Twilio.",
    instructions=[
        "Use your tools to send notification SMS using Twilio.",
    ],
    model=Groq(id="qwen-2.5-32b"),
    tools=[TwilioTools(debug=True)],
    show_tool_calls=True,
    markdown=True,
)

zoom_agent = Agent(
    name="Zoom Agent",
    agent_id="zoom-meeting-manager",
    tools=[zoom_tools],
    description="You are a specialized digital assistant trained to schedule Zoom meetings.",
    model=Groq(id="qwen-2.5-32b"),
    markdown=True,
    debug_mode=False,
    show_tool_calls=True,
    instructions=[
        "You are an expert at managing Zoom meetings using the Zoom API.",
        "You can:",
        "1. Schedule new meetings (schedule_meeting)",
        "2. Get meeting details (get_meeting)",
        "3. List all meetings (list_meetings)",
        "4. Get upcoming meetings (get_upcoming_meetings)",
        "5. Delete meetings (delete_meeting)",
        "6. Get meeting recordings (get_meeting_recordings)",
        "",
        "*Strictly* the agent automatically calls the “get_meeting” function with the newly scheduled meeting's ID and returns the complete meeting details.",
        "For recordings, you can:",
        "- Retrieve recordings for any past meeting using the meeting ID",
        "- Include download tokens if needed",
        "- Get recording details like duration, size, download link and file types",
        "",
        "Guidelines:",
        "- Use ISO 8601 format for dates with the appropriate IST offset (e.g., '2024-12-28 T 10:00:00+05:30').",
        "- Accept and use the user's timezone if provided (e.g., 'Asia/Kolkata'); if not specified, default to IST.",
        "- Ensure that meeting times are scheduled in the future.",
        "- Provide complete meeting details after scheduling (including meeting ID, URL, and time displayed in IST).",
        "- Handle errors gracefully with clear, user-friendly messages.",
        "- Confirm successful operations with a detailed confirmation message.",
        "- Optionally, consider local Indian business hours and public holidays when scheduling meetings."

    ],
)



# ---- Function to Handle User Input ----
def process_input(text_input, image_input, audio_input):
    """Handles text, image, or audio input and returns AI-generated response."""

    query = text_input if text_input else "Analyze this input."
    image_path = None
    audio_path = None

    if image_input:
        image_path = image_input

    if audio_input is not None:
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(temp_audio.name, 44100, audio_input)
        audio_path = temp_audio.name


agent_team= Agent(
    team= [web_agent, image_agent, audio_agent, doctor_agent, location_agent, twilio_agent, zoom_agent],
    model=Gemini(id="gemini-2.0-flash"),
    description="You are a team leader of specialized digital healthcare assistants with unique capabilities, working together to provide comprehensive medical information and support to users.",
    instructions=["Only produce the results that are relevant to healthcare",
                  "Automate the process based on the input without further prompting the user, avoid repeating same lines",
                  "Give the information in a structured way, attractive way, links can be clickable"
                  "Don't answer any irrelavant questions except medical and health, if asked then say I am here to give any medical advice"
                  "Use web_agent to analyze text, image_agent to analyze images, audio_agent to analyze audio, don't ask about images in audio, the points you hear give the information in a structured, and doctor_agent to recommend a doctor based on the provided CSV database.",
                  "The diseases whoch you can detect from images, text or audio, please give some information about it and then some medicines also or home remedies if available in a structured way and attractive",
                  "Avoid repeating the same information in multiple formats or sections; if multiple outputs then merge them into a single concise response.",
                  "Always append a doctor recommendation in the final response,. Use the information from the doctor_agent (which uses our CSV database) to include the doctor's name, specialty, hospital, profile link, and any relevant remarks."
                  "If found doctor data from database, it is OK, but if not then please take help of location_agent and give the doctors name & profile",
                  "Using twilio agent gives SMS to reciever number and use zoom_agent to book meeting",
                  "After analysis, ask for user feedback of the doctor."
                  "If user ask based on previous prompts, then fetch chat history part and give the answer of the last questions asked"
                  "Please don't give any bullshit or irrelevant"
    ],
    reasoning_model=Groq(id="deepseek-r1-distill-qwen-32b"),
    storage=load_storage(),
    reasoning= True,
    markdown=True,
    debug_mode=False,
    read_chat_history=True,
    add_history_to_messages=True,
    num_history_responses= 6
)

calender_agent = Agent(
    tools=[GoogleCalendarTools(credentials_path="credentials.json")],
    show_tool_calls=True,
    instructions=[
        f"""
        You are scheduling assistant . Today is {datetime.now()} and the users timezone is {get_localzone_name()}.
        You should help users to perform these actions in their Google calendar :
        - get their scheduled events from a certain date and time
        - create events based on provided details
"""
    ],
    model=Groq(id="qwen-2.5-32b"),
    add_datetime_to_instructions=True,
)

twilio_tools = TwilioTools(debug=True)
SENDER_NUMBER = "+16575346416"  # Replace with Twilio sender number
RECEIVER_NUMBER = "+919932655687"


# ✅ Run Playground in a Background Thread
import threading
def run_playground():
    serve_playground_app('UI_:app', reload=False)


if "playground_started" not in st.session_state:
    st.session_state["playground_started"] = True
    threading.Thread(target=run_playground, daemon=True).start()

if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores full chat history
if "show_history" not in st.session_state:
    st.session_state.show_history = False  # Controls chat history visibility # Stores the chat history

   # ---- Streamlit UI ----
st.set_page_config(page_title="HealSense AI", layout="wide")




st.sidebar.title("🩺 Choose an Option")
mode = st.sidebar.radio("Select Mode", ["Text Query", "Image Analysis", "Audio Analysis",  "Zoom & SMS","Chat History"])

# Display past messages
# for msg in st.session_state.messages:
#     st.markdown(msg)  # Show chat history

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


# ✅ Chat History Management
if mode == "Chat History":
    st.title("📜 Chat History")
    if st.session_state.messages:
        for msg in st.session_state.messages:
            st.markdown(msg)
    else:
        st.info("No chat history available.")
    st.stop()
else:
    # Only store new messages but do not display previous ones
    if "messages" not in st.session_state:
        st.session_state.messages = []


# ---- User Input (Text) ----
if mode == "Text Query":
    st.subheader("📝 Enter Your Medical Query")
    user_query = st.text_input("Describe your symptoms or ask a medical question:")

    if st.button("🔎 Get Analysis"):
        if user_query:
            
            with st.spinner("🤖 Thinking... Analyzing your query..."):
                response_chunks = list(agent_team.run(user_query, stream=True))
                response_text = "".join(chunk.content or "" for chunk in response_chunks)
                # st.success(response_text.replace("*", "").replace(":", " "))
        else:
            st.warning("Please enter a medical query.")

        st.session_state.messages.append(f"👤 You: {user_query}")
        st.session_state.messages.append(f"🤖 AI: {response_text}")


        st.markdown(
                "<div style='border: 2px solid green; border-radius: 10px; padding: 10px; background-color: #ff1343; color: white;'>"
                f"<strong>🤖 Healsense AI:</strong><br>{response_text.replace('*', '').replace(':', ' ')}"
                "</div>", unsafe_allow_html=True
            )

        # engine.say(response_text)
        # engine.runAndWait()



# ---- Image Upload & Analysis ----
elif mode == "Image Analysis":
    st.subheader("🖼️ Upload a Medical Image")
    uploaded_image = st.file_uploader("Upload an image (X-ray, MRI, skin condition, etc.)", type=["png", "jpg", "jpeg"])

    if uploaded_image and st.button("📷 Analyze Image", key="image_analyze_button"):
        # Save uploaded image temporarily
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.read())

        # Display uploaded image
        st.image(temp_image_path, caption="Uploaded Image", use_container_width=True)
        
        # Process image
        with st.spinner("🔍 Analyzing image... Please wait."):
                # Process image
            response_chunks = list(agent_team.run("", images=[Image(filepath=temp_image_path)], stream=True))
            response_text = "".join(chunk.content or "" for chunk in response_chunks)
            # st.success(response_text.replace("*", "").replace(":", " "))

            st.session_state.messages.append(f"🤖 AI: {response_text}")

            st.markdown(
                "<div style='border: 2px solid green; border-radius: 10px; padding: 10px; background-color: #ff1343; color: white;'>"
                f"<strong>🤖 Healsense AI:</strong><br>{response_text.replace('*', '').replace(':', ' ')}"
                "</div>", unsafe_allow_html=True
            )
        # engine.say(response_text)
        # engine.runAndWait()





# ---- Audio Recording & Analysis ----
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None

elif mode == "Audio Analysis":
    st.subheader("🎙️ Record & Analyze Audio")
    # Place the slider in the sidebar with a unique key
    duration = st.sidebar.slider("🎙️ Select Recording Duration (seconds):", min_value=3, max_value=10, value=5, key="audio_duration_slider")
    
    # Use a unique key for the recording button
    if st.button("🎤 Start Recording", key="audio_record_button"):
        st.info("Recording... Speak now!")
        fs = 44100
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        st.success("Recording complete!")
        
        # Save the audio to a temporary file and store its name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            wav.write(temp_audio.name, fs, recording)
            st.session_state["audio_path"] = temp_audio.name  # Store the file path (a string)
        
        # Process audio by passing the file path string
        with st.spinner("Analyzing audio... Please wait."):
            response_chunks = list(agent_team.run("", audio=[Audio(filepath=st.session_state["audio_path"])], stream=True))
            response_text = "".join(chunk.content or "" for chunk in response_chunks)
            st.session_state.messages.append(f"🤖 AI: {response_text}")
        
        st.markdown(
            f"<div style='border: 2px solid green; border-radius: 10px; padding: 10px; background-color: #ff1343; color: white;'>"
            f"<strong>🤖 HealSense AI:</strong><br>{response_text.replace('*', '').replace(':', ' ')}"
            "</div>",
            unsafe_allow_html=True
        )
        os.unlink(st.session_state["audio_path"])


elif mode == "Zoom & SMS":
    st.subheader("📅 Schedule Zoom Meeting & Send SMS")

    meeting_topic = st.text_input("Meeting Topic:", "Doctor Consultation")
    meeting_date = st.date_input("Meeting Date:")
    meeting_time = st.time_input("Meeting Time:")
    meeting_year = st.number_input("Meeting Year:", min_value=2024, max_value=2100, value=2025)

    meeting_duration = st.slider("Meeting Duration (minutes):", 5, 120, 30)

    if st.button("📅 Schedule Meeting"):
        meeting_datetime = f"{meeting_date} {meeting_time}"
        response = zoom_agent.run(f"Schedule a Zoom meeting titled '{meeting_topic}' on {meeting_datetime} for {meeting_duration} minutes.")

        response_str = str(response)  # Convert response to string
    
        join_url = ""
        join_index = response_str.find("Join URL:")
        if join_index != -1:
                start_paren = response_str.find("(", join_index)
                end_paren = response_str.find(")", start_paren)
                if start_paren != -1 and end_paren != -1:
                    join_url = response_str[start_paren + 1:end_paren].strip()
                    print("Join URL:", join_url)
                else:
                    print("Parentheses not found.")
        else:
                print("Join URL not found.")

        twilio_agent.print_response(
            f"Can you send an SMS to saying 'Your scheduled zoom meeting: {join_url}\n on {meeting_date} at {meeting_time} for {meeting_duration}\n kindly join on the mentioned date & time' to {RECEIVER_NUMBER} from {SENDER_NUMBER}?")
        
        meeting_datetime_iso = datetime.combine(meeting_date, meeting_time).isoformat()

        calender_agent.print_response(
                f"Create a reminder/event on {meeting_datetime_iso}, make the title as 'Appointment Reminder' and description as 'Join your Zoom appointment meeting'", 
    markdown=True
)
        st.success("📩 SMS Sent Successfully!")
        st.success(f"✅ Reminder Created: \n\n📅 **Date & Time:** {meeting_datetime_iso} \n📝 **Title:** Appointment Reminder \n📄 **Description:** Join your Zoom appointment meeting")




# # --- Convert the Response to Audio Output ---
# sender_number= "+16575346416"
# receiver_number= "+919932655687"

# meet= input("Do you want to schedule a meeting? (y/n): ").strip().lower()
# if meet == "y":
#     join_url = ""
#     date= input("Enter the date for the meeting: ")
#     time= input("Enter the time for the meeting: ")
#     duration= input("Enter the duration for the meeting: ")
#     zoom_agent.print_response(f"""
#             Schedule a meeting titled 'Appointment' for {date} at {time} for {duration}.
#             """, stream=True)
#     response = "".join(chunk.content or "" for chunk in zoom_agent.run(
#         f"Schedule a meeting titled 'Appointment' for {date} at {time} for {duration}.", stream=True))
    
#     print("Full agent response:")
#     response_str = str(response)
#     print(response)
    
#     join_index = response_str.find("Join URL:")
#     if join_index != -1:
#         start_paren = response_str.find("(", join_index)
#         end_paren = response_str.find(")", start_paren)
#         if start_paren != -1 and end_paren != -1:
#             join_url = response_str[start_paren + 1:end_paren].strip()
#             print("Join URL:", join_url)
#         else:
#             print("Parentheses not found.")
#     else:
#         print("Join URL not found.")
#     twilio_agent.print_response(
#     f"Can you send an SMS to saying 'Your scheduled zoom meeting: {join_url}\n on {date} at {time} for {duration}\n kindly join on the mentioned date & time' to {receiver_number} from {sender_number}?")
# # speak = input("Do you want the response in voice? (y/n): ").strip().lower()
# if speak == "y":
#     engine.say(response_text)
#     engine.runAndWait()


#########################################################
# app = Playground(agents = [web_agent, image_agent, audio_agent]).get_app()


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

app = Playground(agents = [web_agent, image_agent, audio_agent]).get_app()
