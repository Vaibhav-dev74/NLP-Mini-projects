import tkinter as tk
from tkinter import messagebox

# ----------------------------------------------------------------------
# 1. EMOJI DICTIONARY & REVERSE MAP
# ----------------------------------------------------------------------

EMOJI_MAP = {
    "😀": "Grinning Face",
    "😂": "Tears of Joy", # Simplified
    "😊": "Smiling Eyes", # Simplified
    "😍": "Heart-Eyes",
    "🤔": "Thinking Face",
    "🤩": "Star-Struck",
    "😢": "Crying Face",
    "😭": "Loudly Crying",
    "😎": "Sunglasses",
    "🥳": "Partying Face",
    "👋": "Waving Hand",
    "👍": "Thumbs Up",
    "🙏": "Folded Hands / Prayer", # Multi-meaning kept for word mapping
    "🔥": "Fire",
    "❤️": "Red Heart",
    "🍔": "Hamburger / Food",
    "🍕": "Pizza",
    "💻": "Laptop / Computer",
    "💡": "Light Bulb / Idea",
    "🚀": "Rocket / Launch",
    "🐶": "Dog Face / Canine",
    "🐱": "Cat Face / Feline",
    "🐍": "Snake",
    "💰": "Money Bag / Wealth",
    "✅": "Check Mark / Success",
    "❌": "Cross Mark / Failure",
    "🌟": "Glowing Star",
}

# Create the reverse map for Word-to-Emoji conversion.
# This handles multiple words pointing to the same emoji (e.g., 'folded hands' and 'prayer' -> 🙏)
WORD_MAP = {}
for emoji, meaning_set in EMOJI_MAP.items():
    # Clean and standardize the words for lookup
    meanings = [m.strip().lower() for m in meaning_set.split('/')]
    for meaning in meanings:
        # Map both single words (e.g., 'fire') and phrases (e.g., 'thumbs up')
        WORD_MAP[meaning] = emoji


# ----------------------------------------------------------------------
# 2. APPLICATION LOGIC
# ----------------------------------------------------------------------

class EmojiConverterApp:
    def __init__(self, master):
        """Initializes the main application window and components."""
        self.master = master
        master.title("✨ Bi-Directional Emoji Converter")
        master.geometry("550x500") # Increased size slightly
        master.configure(bg="#f4f7f9")

        # Set up a modern, responsive layout using grid
        master.grid_rowconfigure(0, weight=0)
        master.grid_rowconfigure(4, weight=1) # Center the main action area
        master.grid_columnconfigure(0, weight=1)

        self.setup_ui()

    def setup_ui(self):
        """Creates and places all GUI widgets."""

        # 1. Title Label
        title_label = tk.Label(self.master, text="Bi-Directional Converter", font=("Helvetica", 22, "bold"), bg="#f4f7f9", fg="#2c3e50")
        title_label.grid(row=0, column=0, pady=(25, 5), sticky="n")

        # 2. Instructions Label (General)
        instruction_text = "Enter text or emojis below, then select a conversion direction."
        self.instruction_label = tk.Label(self.master, text=instruction_text, font=("Helvetica", 11), bg="#f4f7f9", fg="#7f8c8d")
        self.instruction_label.grid(row=1, column=0, pady=5)

        # 3. Input Text Box (Entry)
        self.input_entry = tk.Entry(self.master, font=("Arial", 16), width=35, bd=2, relief=tk.FLAT, justify='center')
        self.input_entry.grid(row=2, column=0, padx=20, pady=10, ipady=8, sticky="ew")
        self.input_entry.focus_set()

        # 4. Conversion Buttons (Side-by-side)
        button_frame = tk.Frame(self.master, bg="#f4f7f9")
        button_frame.grid(row=3, column=0, pady=15)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        # Button 1: Emoji -> Word
        convert_to_word_button = tk.Button(button_frame, text="Emoji \u2192 Word", command=self.convert_emoji_to_word,
                                   bg="#3498db", fg="white", font=("Helvetica", 12, "bold"),
                                   relief=tk.RAISED, bd=0, activebackground="#2980b9", activeforeground="white",
                                   cursor="hand2", padx=15, pady=8)
        convert_to_word_button.grid(row=0, column=0, padx=10, sticky="ew")
        convert_to_word_button.bind("<Enter>", lambda e: convert_to_word_button.config(bg="#2980b9"))
        convert_to_word_button.bind("<Leave>", lambda e: convert_to_word_button.config(bg="#3498db"))

        # Button 2: Word -> Emoji
        convert_to_emoji_button = tk.Button(button_frame, text="Word \u2192 Emoji", command=self.convert_word_to_emoji,
                                   bg="#2ecc71", fg="white", font=("Helvetica", 12, "bold"),
                                   relief=tk.RAISED, bd=0, activebackground="#27ae60", activeforeground="white",
                                   cursor="hand2", padx=15, pady=8)
        convert_to_emoji_button.grid(row=0, column=1, padx=10, sticky="ew")
        convert_to_emoji_button.bind("<Enter>", lambda e: convert_to_emoji_button.config(bg="#27ae60"))
        convert_to_emoji_button.bind("<Leave>", lambda e: convert_to_emoji_button.config(bg="#2ecc71"))


        # 5. Output Text Area (Displays results)
        output_frame = tk.Frame(self.master, bg="#ecf0f1", bd=1, relief=tk.SOLID)
        output_frame.grid(row=4, column=0, padx=20, pady=(10, 20), sticky="nsew")
        
        output_frame.grid_rowconfigure(0, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)

        self.result_text = tk.Text(output_frame, font=("Arial", 12), wrap=tk.WORD, bd=0, 
                                   bg="#ecf0f1", fg="#34495e", height=10, padx=10, pady=10)
        self.result_text.grid(row=0, column=0, sticky="nsew")

        # Add a scrollbar
        scrollbar = tk.Scrollbar(output_frame, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.result_text['yscrollcommand'] = scrollbar.set

    def convert_emoji_to_word(self):
        """Converts input Emojis to their corresponding Word meanings."""
        input_text = self.input_entry.get().strip()

        if not input_text:
            messagebox.showwarning("Input Error", "Please enter one or more emojis (e.g., '👍🔥') to convert.")
            return

        self.result_text.delete(1.0, tk.END)
        output_lines = []
        found_count = 0

        # Iterate through each character in the input string
        for char in input_text:
            meaning = EMOJI_MAP.get(char)
            if meaning:
                output_lines.append(f"{char}  \u2192  {meaning}")
                found_count += 1
            elif char.strip() != "":
                # Ignore spaces, but report unrecognized characters
                output_lines.append(f"'{char}'  \u2192  [Meaning not found in map]")

        header_color = "#16a085" if found_count > 0 else "#e74c3c"
        result_header = f"--- Emoji \u2192 Word Results ({found_count} Found) ---\n\n"
        
        self.result_text.tag_configure("header", font=("Arial", 13, "bold"), foreground=header_color)
        self.result_text.insert(tk.END, result_header, "header")
        self.result_text.insert(tk.END, "\n".join(output_lines))
        self.result_text.see(1.0)
        
    def convert_word_to_emoji(self):
        """
        Converts input words/phrases to their corresponding Emojis.
        Handles multi-word phrases first (e.g., 'thumbs up') before checking single words.
        """
        # Convert input to lowercase for case-insensitive matching
        input_text = self.input_entry.get().strip().lower()

        if not input_text:
            messagebox.showwarning("Input Error", "Please enter one or more words (e.g., 'thumbs up fire') to convert.")
            return

        self.result_text.delete(1.0, tk.END)
        output_lines = []
        converted_emojis = []
        found_count = 0
        
        # Split input into tokens (words)
        words = [w.strip() for w in input_text.split() if w.strip()]
        
        # Sort keys by length (longest first) to ensure phrases like 'folded hands' are found before 'folded' or 'hands'
        # This makes the conversion more accurate for multi-word meanings.
        sorted_word_keys = sorted(WORD_MAP.keys(), key=len, reverse=True)

        i = 0
        while i < len(words):
            found_match = False
            # Look for matches using multi-word phrases first (up to 3 words long)
            for j in range(min(len(words) - i, 3), 0, -1):
                phrase = " ".join(words[i:i+j])
                
                # Check if the phrase exists in the map
                if phrase in WORD_MAP:
                    emoji = WORD_MAP[phrase]
                    output_lines.append(f"'{phrase.capitalize()}'  \u2192  {emoji}")
                    converted_emojis.append(emoji)
                    found_count += 1
                    i += j # Skip the words that were part of the found phrase
                    found_match = True
                    break
            
            if not found_match:
                # If no phrase match, check the single word
                word = words[i]
                if word in WORD_MAP:
                    emoji = WORD_MAP[word]
                    output_lines.append(f"'{word.capitalize()}'  \u2192  {emoji}")
                    converted_emojis.append(emoji)
                    found_count += 1
                else:
                    output_lines.append(f"'{word.capitalize()}'  \u2192  [Word not found in map]")
                i += 1
                
        # Add a summary line at the bottom
        if converted_emojis:
             output_lines.append(f"\n\n\u27A1 Final Emoji Sequence: {''.join(converted_emojis)}")


        header_color = "#2ecc71" if found_count > 0 else "#e74c3c"
        result_header = f"--- Word \u2192 Emoji Results ({found_count} Found) ---\n\n"
        
        # Insert results into the text widget
        self.result_text.tag_configure("header", font=("Arial", 13, "bold"), foreground=header_color)
        self.result_text.insert(tk.END, result_header, "header")
        self.result_text.insert(tk.END, "\n".join(output_lines))
        self.result_text.see(1.0)


# ----------------------------------------------------------------------
# 3. MAIN EXECUTION BLOCK
# ----------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = EmojiConverterApp(root)
    root.mainloop()
