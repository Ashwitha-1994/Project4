import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="ML Sentiment Dashboard", layout="wide")
st.title("💬 ChatGPT ML Sentiment Intelligence Dashboard")

# ==============================
# LOAD PKL FILES
# ==============================
@st.cache_resource
def load_resources():
    with open(r"D:\Project4\dashboard_data.pkl", "rb") as f:
        data = pickle.load(f)

    with open(r"D:\Project4\tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open(r"D:\Project4\sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)

    return data, tfidf, model


# ✅ Correct placement (outside function)
data, tfidf, model = load_resources()

# ==============================
# NLTK SETUP
# ==============================
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ==============================
# TEXT CLEANING FUNCTION
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ==============================
# GENERATE SENTIMENT USING MODEL
# ==============================
if 'sentiment' not in data.columns:
    cleaned = data['review'].apply(clean_text)
    vectors = tfidf.transform(cleaned)
    predictions = model.predict(vectors)
    data['sentiment'] = predictions

# ==============================
# MAP NUMERIC LABELS
# ==============================
if data['sentiment'].dtype != object:
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    data['sentiment'] = data['sentiment'].map(label_map)

data['sentiment'] = data['sentiment'].astype(str)

# ==============================
# DATE PROCESSING
# ==============================
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['month'] = data['date'].dt.to_period('M')

data['review_length'] = data['review'].apply(lambda x: len(str(x).split()))

# ==============================
# ANALYSIS SECTION
# ==============================
st.header("📊 Select Analysis")

question = st.selectbox("Choose Question", [
    "1️⃣ Overall Sentiment Distribution",
    "2️⃣ Sentiment vs Rating",
    "3️⃣ Keywords by Sentiment",
    "4️⃣ Sentiment Over Time",
    "5️⃣ Verified vs Non-Verified",
    "6️⃣ Review Length vs Sentiment",
    "7️⃣ Sentiment by Location",
    "8️⃣ Sentiment by Platform",
    "9️⃣ Sentiment by Version",
    "🔟 Most Common Negative Themes"
])

# ==============================
# RUN BUTTON
# ==============================
if st.button("🚀 Run Analysis"):

    # 1️⃣ Overall Sentiment Distribution
    if question == "1️⃣ Overall Sentiment Distribution":

        counts = data['sentiment'].value_counts()
        percentages = data['sentiment'].value_counts(normalize=True) * 100

        summary = pd.DataFrame({
            "Count": counts,
            "Percentage (%)": percentages.round(2)
        })

        st.subheader("Sentiment Distribution")
        st.write(summary)

        fig, ax = plt.subplots()
        counts.plot(kind='bar', ax=ax)
        ax.set_title("Overall Sentiment Distribution")
        st.pyplot(fig)

    # 2️⃣ Sentiment vs Rating
    elif question == "2️⃣ Sentiment vs Rating":

        cross = pd.crosstab(data['rating'], data['sentiment'])
        st.write(cross)

        fig, ax = plt.subplots()
        sns.heatmap(cross, annot=True, fmt='d', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # 3️⃣ Keywords by Sentiment (FIXED)
    elif question == "3️⃣ Keywords by Sentiment":

        st.subheader("🔎 Combined Sentiment Keywords")

        # Separate text by sentiment
        positive_text = " ".join(
            data[data['sentiment'] == "Positive"]['review'].apply(clean_text)
        )

        neutral_text = " ".join(
            data[data['sentiment'] == "Neutral"]['review'].apply(clean_text)
        )

        negative_text = " ".join(
            data[data['sentiment'] == "Negative"]['review'].apply(clean_text)
        )

        combined_text = positive_text + " " + neutral_text + " " + negative_text

        if combined_text.strip() == "":
            st.warning("No valid words found.")
        else:
            wc = WordCloud(
                width=1000,
                height=500,
                background_color="white",
                collocations=False
            ).generate(combined_text)

            # Color function
            def sentiment_color(word, *args, **kwargs):
                if word in positive_text:
                    return "green"
                elif word in neutral_text:
                    return "blue"
                else:
                    return "red"

            wc.recolor(color_func=sentiment_color)

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(
                "Sentiment Keywords (Green=Positive, Blue=Neutral, Red=Negative)",
                fontsize=16
            )
            st.pyplot(fig)

            top_words = (
                pd.Series(combined_text.split())
                .value_counts()
                .head(20)
                .reset_index()
            )
            top_words.columns = ["Keyword", "Frequency"]

            st.subheader("📌 Top 20 Overall Keywords")
            st.dataframe(top_words, use_container_width=True)

    # 4️⃣ Sentiment Over Time
    elif question == "4️⃣ Sentiment Over Time":
        trend = data.groupby(['month', 'sentiment']).size().unstack().fillna(0)
        st.line_chart(trend)

    # 5️⃣ Verified vs Non-Verified
    elif question == "5️⃣ Verified vs Non-Verified":
        cross = pd.crosstab(data['verified_purchase'], data['sentiment'])
        st.write(cross)

    # 6️⃣ Review Length vs Sentiment
    elif question == "6️⃣ Review Length vs Sentiment":
        avg_length = data.groupby('sentiment')['review_length'].mean()
        st.write(avg_length)

    # 7️⃣ Sentiment by Location
    elif question == "7️⃣ Sentiment by Location":
        location_data = pd.crosstab(data['location'], data['sentiment'])
        st.write(location_data)

    # 8️⃣ Sentiment by Platform
    elif question == "8️⃣ Sentiment by Platform":
        platform_data = pd.crosstab(data['platform'], data['sentiment'])
        st.write(platform_data)

    # 9️⃣ Sentiment by Version
    elif question == "9️⃣ Sentiment by Version":
        version_data = pd.crosstab(data['version'], data['sentiment'])
        st.write(version_data)

    # 🔟 Most Common Negative Themes
    elif question == "🔟 Most Common Negative Themes":
        negative_text = " ".join(
            data[data['sentiment'] == "Negative"]['review'].apply(clean_text)
        )

        words = negative_text.split()
        common_words = pd.Series(words).value_counts().head(20)

        st.subheader("Top 20 Negative Keywords")
        st.write(common_words)

st.markdown("---")
st.markdown("⚡ Powered by TF-IDF + ML Model + Streamlit")