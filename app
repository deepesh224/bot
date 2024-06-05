import streamlit as st
import random
import time
import cv2
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import subprocess
from datetime import datetime, timedelta
import spacy

# Configure the API key for Google Generative AI
genai.configure(api_key="AIzaSyBul1X0XZl5ymAPSDtcaqtdNdJFOg17fmI")

# Initialize the VILT model and processor for image question answering
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Configure Google Generative AI (Gemini) model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

chat_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

chat_session = chat_model.start_chat(history=[])

# Initialize the recognizer, TTS engine, and spaCy model for reminders
recognizer = sr.Recognizer()
nlp = spacy.load('en_core_web_sm')

# Function to recognize speech and convert it to text
def recognize_speech():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            st.write("Could not request results; check your network connection.")
            return None

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to parse the reminder text and extract time using spaCy
def parse_reminder(text):
    doc = nlp(text)
    time_value = None
    time_unit = None
    reminder_message = None

    for ent in doc.ents:
        if ent.label_ == "TIME":
            time_value, time_unit = extract_time_value(ent.text)
        if ent.label_ == "DATE":
            time_value, time_unit = extract_time_value(ent.text)

    reminder_message = " ".join([token.text for token in doc if token.ent_type_ not in ["TIME", "DATE"]])
    if time_value and time_unit:
        reminder_time = calculate_reminder_time(time_value, time_unit)
        return reminder_time, reminder_message
    else:
        return None, None

def extract_time_value(time_text):
    time_text = time_text.lower()
    time_value = None
    time_unit = None

    if "second" in time_text:
        time_unit = "second"
        time_value = int(''.join(filter(str.isdigit, time_text)))
    elif "minute" in time_text:
        time_unit = "minute"
        time_value = int(''.join(filter(str.isdigit, time_text)))
    elif "hour" in time_text:
        time_unit = "hour"
        time_value = int(''.join(filter(str.isdigit, time_text)))

    return time_value, time_unit

def calculate_reminder_time(time_value, time_unit):
    if time_unit == "second":
        reminder_time = datetime.now() + timedelta(seconds=time_value)
    elif time_unit == "minute":
        reminder_time = datetime.now() + timedelta(minutes=time_value)
    elif time_unit == "hour":
        reminder_time = datetime.now() + timedelta(hours=time_value)
    else:
        reminder_time = None
    return reminder_time

# Function to capture an image
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.write("Error: Could not open webcam.")
        return None
    ret, frame = cap.read()
    if not ret:
        st.write("Error: Could not read frame.")
        return None
    image_path = 'captured_image.jpg'
    cv2.imwrite(image_path, frame)
    cap.release()
    return image_path

# Function to answer a question about an image
def answer_question(image_path, question):
    image = Image.open(image_path)
    encoding = processor(image, question, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predicted_index = logits.argmax(-1).item()
    answer = model.config.id2label[predicted_index]
    return answer

# Function to open an application
def open_application(app_name):
    applications = {
        'notepad': 'notepad.exe',
        'calculator': 'calc.exe',
        'paint': 'mspaint.exe',
        'command prompt': 'cmd.exe',
        'brave': 'brave.exe',
        'firefox': 'firefox.exe'
    }
    if app_name in applications:
        subprocess.Popen(applications[app_name])
        speak(f"Opening {app_name}")
    else:
        speak("Sorry, I can't open that application.")

# Function to get chatbot response
def chatbot_response(user_input):
    user_input = user_input.lower()
    responses = {
        "i love you": "I love you too guru ji!",
        "i hate you": "Why do you hate me guru ji?",
        "i'm happy": "That's great to hear guru ji!",
        "i'm sad": "I'm sorry to hear that guru ji. How can I help?",
        "i'm angry": "Take a deep breath guru ji. What's bothering you?",
        "thank you": "You're welcome guru ji!",
        "hello": "Hi guru ji! How can I help you today?",
        "goodbye": "Goodbye guru ji! Have a great day!",
        "how are you": "I'm just a bunch of code guru ji, but thanks for asking!"
    }
    return responses.get(user_input, None)

# Streamlit interface
st.title('Interactive Assistant App')

menu = ['Home', 'Capture Image', 'Ask Question', 'Open Application', 'Set Reminder', 'Chat']
choice = st.sidebar.selectbox('Select Action', menu)

if choice == 'Home':
    st.write("Welcome to the Interactive Assistant App. Choose an action from the sidebar.")

elif choice == 'Capture Image':
    st.write("Capturing an image from the webcam.")
    image_path = capture_image()
    if image_path:
        st.image(image_path, caption='Captured Image')
        question = st.text_input('Ask a question about the image:')
        if st.button('Get Answer'):
            answer = answer_question(image_path, question)
            st.write(f"The answer is: {answer}")

elif choice == 'Ask Question':
    st.write("You can ask me anything.")
    question = st.text_input('Ask a question:')
    if st.button('Get Response'):
        response = chatbot_response(question)
        if response:
            st.write(response)
            speak(response)
        else:
            response = chat_session.send_message(question)
            st.write(response.text)
            speak(response.text)

elif choice == 'Open Application':
    st.write("Open an application.")
    app_name = st.text_input('Enter the name of the application:')
    if st.button('Open'):
        open_application(app_name)

elif choice == 'Set Reminder':
    st.write("Set a reminder.")
    reminder_text = st.text_input('Enter the reminder text:')
    if st.button('Set Reminder'):
        reminder_time, reminder_message = parse_reminder(reminder_text)
        if reminder_time:
            st.write(f"Reminder set for {reminder_message} at {reminder_time.strftime('%H:%M:%S')}")
            while datetime.now() < reminder_time:
                time.sleep(1)
            st.write(f"Reminder: {reminder_message}")
            speak(f"guru ji Reminder: {reminder_message}")
        else:
            st.write("Sorry, I could not understand the time for the reminder.")

elif choice == 'Chat':
    st.write("Chat with me.")
    user_input = st.text_input('You:')
    if st.button('Send'):
        response = chatbot_response(user_input)
        if response:
            st.write(response)
            speak(response)
        else:
            response = chat_session.send_message(user_input)
            st.write(response.text)
            speak(response.text)
