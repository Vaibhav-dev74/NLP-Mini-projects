# --- Simple Dialogue Classifier GUI (Using Tkinter, No External Classification Libraries) ---
import tkinter as tk
from tkinter import ttk

# --- Core Classification Logic ---

def clean_text(text):
    """
    Performs basic text cleaning: converts to lowercase and removes periods/commas.
    '?' and '!' are kept as they are key features for classification.
    """
    text = text.lower().strip()
    
    # Remove period, comma, colon, semicolon for cleaner keyword matching
    for char in ('.', ',', ':', ';'):
        text = text.replace(char, '')
    
    return text

def classify_dialogue(dialogue_text):
    """
    Classifies the input text into one of five categories:
    GREETING, FAREWELL, QUESTION, REQUEST, EXCLAMATION/EMPHASIS, or STATEMENT.
    """
    if not dialogue_text or dialogue_text.isspace():
        return "EMPTY INPUT"

    cleaned = clean_text(dialogue_text)
    
    # --- 1. Punctuation Feature Check (Highest Priority) ---
    
    # Check for question marks
    if dialogue_text.strip().endswith('?'):
        return "QUESTION"
    
    # Check for exclamation marks
    if dialogue_text.strip().endswith('!'):
        return "EXCLAMATION/EMPHASIS"

    # --- 2. Enhanced Keyword/Structure Check ---
    
    # Define keywords for classification
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "greetings"]
    farewell_keywords = ["bye", "goodbye", "see you", "cya", "farewell", "later", "take care"]
    
    # Keywords for polite requests/commands that don't end in '?'
    request_phrases = ["can you", "could you", "would you", "please", "i need", "i would like to"]
    
    # Implicit question words
    implicit_question_keywords = ["who", "what", "where", "when", "why", "how", "tell me about"]
    
    
    # Check for GREETING
    for word in greeting_keywords:
        # Check if the word is found, using spaces to avoid partial matches (e.g., 'hi' in 'history')
        if f" {word} " in f" {cleaned} ":
            return "GREETING"
            
    # Check for FAREWELL
    for word in farewell_keywords:
        if f" {word} " in f" {cleaned} ":
            return "FAREWELL"

    # Check for REQUEST (Note: This is checked before general questions)
    for word in request_phrases:
        if cleaned.startswith(word): # Check if the phrase starts the sentence
            return "REQUEST"

    # Check for IMPLICIT QUESTION (wh-words)
    for word in implicit_question_keywords:
        if f" {word} " in f" {cleaned} ":
            return "QUESTION"
    
    # --- 3. Default Classification ---
    return "STATEMENT"

# --- GUI Application Class ---

class DialogueClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Simple Dialogue Classifier")
        master.geometry("550x300")
        master.resizable(False, False)
        
        # Set a clean theme
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10, 'bold'), padding=6)
        style.configure('Result.TLabel', font=('Arial', 14, 'bold'), foreground='#333333', padding=10)
        
        # Main Frame
        self.main_frame = ttk.Frame(master, padding="15 15 15 15")
        self.main_frame.pack(fill='both', expand=True)

        # 1. Instruction Label
        self.instruction = ttk.Label(self.main_frame, text="Enter a single sentence to classify its dialogue type:", font=('Arial', 11))
        self.instruction.pack(pady=(0, 10), anchor='w')

        # 2. Input Field (Entry Widget)
        self.input_text = tk.StringVar()
        self.text_entry = ttk.Entry(self.main_frame, textvariable=self.input_text, width=70, font=('Arial', 12))
        self.text_entry.pack(pady=5, padx=5)
        self.text_entry.bind('<Return>', lambda event: self.classify_action()) # Allow Enter key to trigger

        # 3. Classify Button
        self.classify_button = ttk.Button(self.main_frame, text="Classify Dialogue", command=self.classify_action)
        self.classify_button.pack(pady=15)

        # 4. Result Label
        self.result_label_prefix = ttk.Label(self.main_frame, text="Classification:", font=('Arial', 11, 'bold'))
        self.result_label_prefix.pack(pady=(10, 0))
        
        self.result_label = ttk.Label(self.main_frame, text="---", style='Result.TLabel', anchor='center', width=30)
        self.result_label.pack(pady=(5, 10))
        
        # Initial focus
        self.text_entry.focus()


    def classify_action(self):
        """
        Retrieves text from the input, calls the classifier function, and updates the result label.
        """
        input_dialogue = self.input_text.get()
        
        # Call the core classification function
        category = classify_dialogue(input_dialogue)
        
        # Define color based on category for better visual detailing
        color_map = {
            "QUESTION": "#007BFF",        # Blue
            "GREETING": "#28A745",        # Green
            "FAREWELL": "#DC3545",        # Red
            "REQUEST": "#FFC107",         # Yellow/Orange
            "EXCLAMATION/EMPHASIS": "#FD7E14", # Dark Orange
            "STATEMENT": "#6C757D",       # Gray
            "EMPTY INPUT": "#343A40"
        }
        
        # Update the result label with color and text
        self.result_label.config(text=category, foreground=color_map.get(category, '#333333'))


# --- Main Execution ---
if __name__ == "__main__":
    # Create the main window instance
    root = tk.Tk()
    # Instantiate the application
    app = DialogueClassifierApp(root)
    # Start the Tkinter event loop
    root.mainloop()
