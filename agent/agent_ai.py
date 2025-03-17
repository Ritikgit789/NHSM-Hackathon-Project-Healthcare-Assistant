import os
from dotenv import load_dotenv
import google.generativeai as genai
from agno.agent import Agent 
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image, Audio
import pyttsx3
import os

# For real-time audio recording.
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile

# Load environment variables and configure APIs.
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Mode Selection ---
mode = input("Select mode (text/image/audio): ").strip().lower()

# Initialize media path variables.
image_path = None
audio_path = None

if mode == "image":
    image_path = input("Enter the path for the image file: ").strip()
elif mode == "audio":
    duration = 8  # seconds to record
    fs = 44100    # sample rate (Hz)
    print(f"Recording real-time audio for {duration} seconds...")
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished.
    print("Recording complete!")
    # Save the recorded audio to a temporary WAV file.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        wav.write(temp_audio_file.name, fs, recording)
        audio_path = temp_audio_file.name

query = input("Enter your question: ")

# --- Define Agents ---

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

image_agent=Agent(
    name="Image Analyzer",
    description=(
        "You are a specialized digital healthcare assistant trained to analyze images and gives suggestions. "
        "Your task is to identify observable signs of diseases or conditions and provide detailed medicines, cure, factual insights about them."
    ),
    role="analyze medical images and provide detailed insights on visible conditions",
    model=Gemini(id="gemini-2.0-flash",grounding=True),
    instructions=(
        "When analyzing an image, please do the following:\n"
        "- Clearly describe the observable features and any abnormalities in the image.\n"
        "- Provide detailed information on the potential diseases or conditions associated with these features, "
        "- Including possible causes, medicines, and relevant characteristics.\n"
        "- Focus solely on delivering an informative analysis based on the image content without including any hashtags, stars & disclaimers.\n"
        "If no user query is provided, focus solely on delivering an informative analysis based on the image content without including any disclaimers."
    ),
    show_tool_calls=True,
    markdown=True,
)

audio_aget=Agent(
    name="Audio Agent",
    description=(
        "You are a specialized digital healthcare assistant trained to analyze audio. "
        "Your task is to identify observable signs of diseases or conditions and provide detailed medicines, cure, factual insights about them."
    ),
    role="analyze medical audio and provide detailed insights on audible conditions",
    model=Gemini(id="gemini-2.0-flash",grounding=True),
    instructions=(
        "When analyzing an audio, please do the following:\n"
        "- Clearly describe the observable features and any abnormalities mentioned or observed in the audio.\n"
        "- Provide detailed information on the potential diseases or conditions associated with these features, "
        "- Including possible causes, medicines and relevant characteristics.\n"
        "- Focus solely on delivering an informative analysis based on the audio content without including any hashtags, stars & disclaimers."
    ),
    show_tool_calls=True,
    markdown=True,
)

agent_team= Agent(
    team= [web_agent, image_agent, audio_aget],
    model=Gemini(id="gemini-2.0-flash"),
    instructions=["Only produce the results that are relevant to healthcare, skip saying asterisk and hashtags"],
    markdown=True,
    debug_mode=False,
)


if image_path:
    response_chunks = list(agent_team.run(query, images=[Image(filepath=image_path)], stream=True))
elif audio_path:
    response_chunks = list(agent_team.run(query, audio=[Audio(filepath=audio_path)], stream=True))
else:
    response_chunks = list(agent_team.run(query, stream=True))
    
response_text = "".join(chunk.content or "" for chunk in response_chunks)
print("\nFinal Response:\n", response_text)
# --- Convert the Response to Audio Output ---
engine = pyttsx3.init()
engine.say(response_text)
engine.runAndWait()