import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")
nltk.download("omw-1.4")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# Define the text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# Function to analyze sentiment
def analyze_sentiment(text):
    processed_text = preprocess_text(text)
    sentiment_score = sia.polarity_scores(processed_text)["compound"]
    sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    return processed_text, sentiment_score, sentiment_label

# Streamlit App Title
st.title("NLP-Based Sentiment Analysis App")

# File Upload Section
st.header("Upload a CSV or TXT file for Sentiment Analysis")
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        content = uploaded_file.getvalue().decode("utf-8")
        df = pd.DataFrame({"Text": content.splitlines()})

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    text_column = None
    for col in df.columns:
        if df[col].dtype == "object":
            text_column = col
            break

    if text_column:
        df["Processed_Text"], df["Sentiment_Score"], df["Sentiment_Label"] = zip(*df[text_column].apply(analyze_sentiment))

        st.write("### Sentiment Analysis Results")
        st.dataframe(df[[text_column, "Processed_Text", "Sentiment_Label"]])

        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots()
        df["Sentiment_Label"].value_counts().plot(kind="bar", ax=ax, color=["green", "red", "gray"])
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Word Cloud
        st.write("### Word Cloud of Most Frequent Words")
        all_words = " ".join(df["Processed_Text"].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.warning("No valid text column found in the uploaded file!")

# Real-time Text Sentiment Analysis
st.header("📝 Real-time Text Sentiment Analysis")
user_text = st.text_area("Enter text for sentiment analysis:")

if user_text:
    processed_text, sentiment_score, sentiment_label = analyze_sentiment(user_text)
    st.write(f"Processed Text: {processed_text}")
    st.write(f"Sentiment Score: {sentiment_score}")
    st.write(f"Sentiment Label: {sentiment_label}")

# Real-time Speech Sentiment Analysis
st.header("🎤 Real-time Speech Sentiment Analysis")

if st.button("Start Recording"):
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening... Speak now!")
            with st.spinner("Recording..."):
                audio = recognizer.listen(source, timeout=10)

        speech_text = recognizer.recognize_google(audio)
        st.write(f"Recognized Speech: {speech_text}")

        processed_text, sentiment_score, sentiment_label = analyze_sentiment(speech_text)
        st.write(f"Processed Text: {processed_text}")
        st.write(f"Sentiment Score: {sentiment_score}")
        st.write(f"Sentiment Label: {sentiment_label}")

    except sr.UnknownValueError:
        st.error("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    except OSError:
        st.error("No microphone detected! Make sure your microphone is plugged in and working.")
    except TimeoutError:
        st.error("No speech detected within the time limit.")
