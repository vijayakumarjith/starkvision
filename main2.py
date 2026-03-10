import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image, ImageTk, ImageSequence
from itertools import count
import tkinter as tk
import string
from keras.models import load_model

# Load the classifier model
classifier = load_model('model.h5', compile=False)

# Set image size for predictions
image_x, image_y = 64, 64

# Recognized phrases for GIFs
isl_gif = [
    'any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
    'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office',
    'do you have money', 'do you want something to drink', 'do you want tea or coffee', 'do you watch TV',
    'dont worry', 'flower is beautiful', 'good afternoon', 'good evening', 'good morning', 'good night',
    'good question', 'had your lunch', 'happy journey', 'hello what is your name', 'how many people are there in your family',
    'i am a clerk', 'i am bore doing nothing', 'i am fine', 'i am sorry', 'i am thinking', 'i am tired',
    'i dont understand anything', 'i go to a theatre', 'i love to shop', 'i had to say something but i forgot',
    'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
    'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
    'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime',
    'shall I help you', 'shall we go together tomorrow', 'sign language interpreter', 'sit down', 'stand up',
    'take care', 'there was traffic jam', 'wait I am thinking', 'what are you doing', 'what is the problem',
    'what is todays date', 'what does your father do', 'what is your job', 'what is your mobile number', 'what is your name',
    'whats up', 'when is your interview', 'when we will go', 'where do you stay', 'where is the bathroom',
    'where is the police station', 'you are wrong'
]
arr = list(string.ascii_lowercase)  # List of letters

# Function to classify characters
def give_char():
    from keras.preprocessing import image
    test_image = image.load_img('tmp1.png', target_size=(image_x, image_y))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    indx = np.argmax(result[0])
    return chars[indx]

# GUI class management
class Tk_Manage(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, VtoS, StoV, LiveVoice):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

# Home page
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Two Way Sign Language Translator", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        tk.Button(self, text="Voice to Sign", command=lambda: controller.show_frame(VtoS)).pack()
        tk.Button(self, text="Sign to Voice", command=lambda: controller.show_frame(StoV)).pack()
        tk.Button(self, text="Live Voice", command=lambda: controller.show_frame(LiveVoice)).pack()

# Class for displaying GIFs
class ImageLabel(tk.Label):
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        self.frames = []
        for i in count(1):
            try:
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
            except EOFError:
                break
        self.delay = im.info.get('duration', 100)
        self.loc = 0
        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.loc += 1
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)

# Live Voice functionality
class LiveVoice(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Live Voice", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage)).pack()
        self.result_label = tk.Label(self)
        self.result_label.pack()

        def listen_voice():
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening for voice input...")
                audio = recognizer.listen(source)
                try:
                    text = recognizer.recognize_google(audio).lower()
                    print(f'You said: {text}')
                    self.display_sign_language(text)
                except sr.UnknownValueError:
                    print("Could not understand the audio")
                except sr.RequestError:
                    print("Could not request results; check your network connection")

        tk.Button(self, text="Listen", command=listen_voice).pack()

    def display_sign_language(self, text):
        for word in isl_gif:
            if word in text:
                gif_path = f'ISL_Gifs/{word}.gif'
                lbl = ImageLabel(self)
                lbl.load(gif_path)
                lbl.pack()
                return
        
        # If no phrase is matched, show each letter as a static image
        for char in text:
            if char in arr:
                image_path = f'letters/{char}.jpg'
                img = Image.open(image_path)
                img = img.resize((200, 200))  # Resize as needed
                photo = ImageTk.PhotoImage(img)
                self.result_label.config(image=photo)
                self.result_label.image = photo
                self.result_label.after(800)  # Display each letter for 0.8 seconds

# Voice to Sign functionality
class VtoS(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Voice to Sign", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        gif_box = tk.Label(self)
        gif_box.pack()

        def gif_stream():
            global cnt
            global gif_frames
            if cnt == len(gif_frames):
                return
            img = gif_frames[cnt]
            cnt += 1
            imgtk = ImageTk.PhotoImage(image=img)
            gif_box.imgtk = imgtk
            gif_box.configure(image=imgtk)
            gif_box.after(1000, gif_stream)

        def Take_input():
            INPUT = inputtxt.get("1.0", "end-1c")
            global gif_frames
            gif_frames = self.func(INPUT)
            global cnt
            cnt = 0
            gif_stream()

        l = tk.Label(self, text="Enter Text or Voice:")
        inputtxt = tk.Text(self, height=4, width=25)
        Display = tk.Button(self, height=2, width=20, text="Convert", command=Take_input)
        l.pack()
        inputtxt.pack()
        Display.pack()

    def func(self, input_text):
        frames = []
        # Check if input_text matches any recognized phrase
        for phrase in isl_gif:
            if phrase in input_text:
                gif_path = f'ISL_Gifs/{phrase}.gif'
                gif = Image.open(gif_path)
                frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
                return frames
        
        # If no matching phrase, return static images for each character
        for char in input_text:
            if char in arr:
                img_path = f'letters/{char}.jpg'
                img = Image.open(img_path)
                img = img.resize((200, 200))  # Resize as needed
                frames.append(img)
        return frames

# Sign to Voice functionality
class StoV(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Sign to Voice", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage)).pack()
        l = tk.Label(self, text="Click the button below and show a letter sign.")
        l.pack()
        result = tk.Label(self, text="")
        result.pack()

        def classify():
            cam = cv2.VideoCapture(0)
            while True:
                ret, test_image = cam.read()
                if not ret:
                    continue
                cv2.imwrite("tmp1.png", test_image)
                char = give_char()
                print(char)
                result.config(text=char)
                break
            cam.release()
            cv2.destroyAllWindows()

        tk.Button(self, text="Capture", command=classify).pack()

# Run the application
app = Tk_Manage()
app.mainloop()
