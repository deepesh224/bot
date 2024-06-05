import os
from dotenv import load_dotenv
import streamlit as st
import random
import time
import cv2
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch
import google.generativeai as genai
from gtts import gTTS
import base64
import speech_recognition as sr
from googletrans import Translator
import pyttsx3
import subprocess
from datetime import datetime, timedelta
import spacy

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set the GOOGLE_API_KEY environment variable.")
else:
    genai.configure(api_key=api_key)

# Initialize the Gemini Pro models
text_model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')
chat_session = text_model.start_chat(history=[])

# Initialize the VILT model and processor for image question answering
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize Translator
translator = Translator()

# Initialize the recognizer and spaCy model for reminders
recognizer = sr.Recognizer()
nlp = spacy.load('en_core_web_sm')

# Function to translate text
def translate_text(text, src_language, target_language='en'):
    try:
        translation = translator.translate(text, src=src_language, dest=target_language)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Function to detect the language of the text
def detect_language(text):
    try:
        detected_lang = translator.detect(text).lang
        return detected_lang
    except Exception as e:
        st.error(f"Language detection error: {e}")
        return 'en'

# Function to get response for text queries
def get_text_response(question, target_language='en'):
    try:
        detected_lang = detect_language(question)
        translated_question = translate_text(question, src_language=detected_lang, target_language='en')
        response = chat_session.send_message(translated_question, stream=True)
        response_text = ''.join(chunk.text for chunk in response if chunk.text)
        translated_response = translate_text(response_text, src_language='en', target_language=target_language)
        return translated_response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to get response for text + image queries
def get_vision_response(input_text, image, target_language='en'):
    try:
        detected_lang = detect_language(input_text)
        translated_text = translate_text(input_text, src_language=detected_lang, target_language='en')
        if translated_text:
            response = vision_model.generate_content([translated_text, image])
        else:
            response = vision_model.generate_content(image)
        if response.text.strip():  # Check if response text is not empty
            translated_response = translate_text(response.text, src_language='en', target_language=target_language)
            return translated_response
        else:
            st.warning("Response did not contain valid text data.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to generate voice output
def generate_voice(text, language='en'):
    tts = gTTS(text, lang=language_code(language))
    tts.save("response.mp3")
    with open("response.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_base64

# Function to get language code from language name
def language_code(language_name):
    language_codes = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "Hindi": "hi",
        "Telugu": "te",
        "Chinese (Simplified)": "zh-CN",
        "Arabic": "ar",
        "Bengali": "bn",
        "Russian": "ru",
        "Portuguese": "pt",
        "Japanese": "ja",
        # Add more languages as needed
    }
    return language_codes.get(language_name, "en")

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
        "i love you": "I love you too!",
        "i hate you": "Why do you hate me?",
        "i'm happy": "That's great to hear!",
        "i'm sad": "I'm sorry to hear that. How can I help?",
        "i'm angry": "Take a deep breath. What's bothering you?",
        "thank you": "You're welcome!",
        "hello": "Hi! How can I help you today?",
        "goodbye": "Goodbye! Have a great day!",
        "how are you": "I'm just a bunch of code, but thanks for asking!"
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
            response = get_text_response(question)
            st.write(response)
            speak(response)

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
            speak(f"Reminder: {reminder_message}")
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
            response = get_text_response(user_input)
            st.write(response)
            speak(response)


