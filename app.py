import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Data Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load product data
file_path = 'E:/SEMESTER 07/Artificial Intelligence/Project/AI.xlsx'
data = pd.read_excel(file_path)

def analyze_word_sentiment(word):
    vs = analyzer.polarity_scores(word)
    return vs['compound']

def categorize_sentiment(polarity):
    if polarity > 0.1:
        return 'Best'
    elif polarity < -0.1:
        return 'Worst'
    else:
        return 'Average'

def on_submit():
    product_name = product_name_var.get()
    new_review = review_var.get()
    if not product_name or not new_review:
        messagebox.showerror("Error", "Please fill out all fields")
        return
    
    words = preprocess_text(new_review)
    sentiments = []
    for word in words:
        polarity = analyze_word_sentiment(word)
        sentiment_category = categorize_sentiment(polarity)
        sentiments.append(f"{word}: {sentiment_category}")
    
    overall_sentiment = categorize_review_sentiment(words)
    result_var.set(f"Overall Sentiment: {overall_sentiment}")
    word_analysis_var.set("\n".join(sentiments))

def categorize_review_sentiment(words):
    best_count = sum(1 for word in words if categorize_sentiment(analyze_word_sentiment(word)) == 'Best')
    worst_count = sum(1 for word in words if categorize_sentiment(analyze_word_sentiment(word)) == 'Worst')
    average_count = len(words) - best_count - worst_count
    
    if best_count > worst_count and best_count > average_count:
        return 'Best'
    elif worst_count > best_count and worst_count > average_count:
        return 'Worst'
    else:
        return 'Average'

# Create the main window
root = tk.Tk()
root.title("Product Review Sentiment Analysis")

# Create and set the variables
product_name_var = tk.StringVar()
review_var = tk.StringVar()
result_var = tk.StringVar()
word_analysis_var = tk.StringVar()

# Create the UI components
tk.Label(root, text="Select Product:").grid(row=0, column=0, padx=10, pady=10)
product_name_dropdown = ttk.Combobox(root, textvariable=product_name_var)
product_name_dropdown['values'] = data['Product Name'].unique().tolist()
product_name_dropdown.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Enter Review:").grid(row=1, column=0, padx=10, pady=10)
review_entry = tk.Entry(root, textvariable=review_var, width=50)
review_entry.grid(row=1, column=1, padx=10, pady=10)

tk.Button(root, text="Submit", command=on_submit).grid(row=2, column=0, columnspan=2, pady=20)

tk.Label(root, textvariable=result_var).grid(row=3, column=0, columnspan=2, pady=10)

tk.Label(root, text="Word Sentiment Analysis:").grid(row=4, column=0, padx=10, pady=10)
tk.Label(root, textvariable=word_analysis_var, justify='left').grid(row=4, column=1, padx=10, pady=10)

# Start the main loop
root.mainloop()
