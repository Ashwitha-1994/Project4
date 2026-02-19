import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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

    with open(r"D:\Project4\lemmatizer.pkl", "rb") as f:
        lemmatizer = pickle.load(f)

    with open(r"D:\Project4\stopwords.pkl", "rb") as f:
        stopwords = pickle.load(f)

    with open(r"D:\Project4\sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)

    return data, tfidf, lemmatizer, stopwords, model


data, tfidf, lemmatizer, stopwords, model = load_resources()

# ==============================
# TEXT CLEANING FUNCTION
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords]
    return " ".join(words)


# ==============================
# GENERATE SENTIMENT USING MODEL
# ==============================
if 'sentiment' not in data.columns:
    cleaned = data['review'].apply(clean_text)
    vectors = tfidf.transform(cleaned)
    predictions = model.predict(vectors)
    data['sentiment'] = predictions

# Date processing
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

    # 1️⃣ Overall Sentiment
    if question == "1️⃣ Overall Sentiment Distribution":
        counts = data['sentiment'].value_counts(normalize=True) * 100
        st.write(counts.round(2))

        fig, ax = plt.subplots()
        counts.plot(kind='bar', ax=ax)
        ax.set_ylabel("Percentage")
        st.pyplot(fig)

    # 2️⃣ Sentiment vs Rating
    elif question == "2️⃣ Sentiment vs Rating":
        cross = pd.crosstab(data['rating'], data['sentiment'])
        st.write(cross)

        fig, ax = plt.subplots()
        sns.heatmap(cross, annot=True, fmt='d', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # 3️⃣ Keywords by Sentiment
    elif question == "3️⃣ Keywords by Sentiment":
        sentiment_type = st.selectbox("Select Sentiment",
                                      data['sentiment'].unique())

        text = " ".join(data[data['sentiment']==sentiment_type]['review'])

        wordcloud = WordCloud(width=800,
                              height=400,
                              background_color='white').generate(text)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)

    # 4️⃣ Sentiment Over Time
    elif question == "4️⃣ Sentiment Over Time":
        trend = data.groupby(['month','sentiment']).size().unstack().fillna(0)
        st.line_chart(trend)

    # 5️⃣ Verified vs Non-Verified
    elif question == "5️⃣ Verified vs Non-Verified":
        cross = pd.crosstab(data['verified_purchase'], data['sentiment'])
        st.write(cross)

        fig, ax = plt.subplots()
        cross.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # 6️⃣ Review Length vs Sentiment
    elif question == "6️⃣ Review Length vs Sentiment":
        avg_length = data.groupby('sentiment')['review_length'].mean()
        st.write(avg_length)

        fig, ax = plt.subplots()
        avg_length.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # 7️⃣ Sentiment by Location
    elif question == "7️⃣ Sentiment by Location":
        location_data = pd.crosstab(data['location'], data['sentiment'])
        st.write(location_data)

    # 8️⃣ Sentiment by Platform
    elif question == "8️⃣ Sentiment by Platform":
        platform_data = pd.crosstab(data['platform'], data['sentiment'])
        st.write(platform_data)

        fig, ax = plt.subplots()
        platform_data.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # 9️⃣ Sentiment by Version
    elif question == "9️⃣ Sentiment by Version":
        version_data = pd.crosstab(data['version'], data['sentiment'])
        st.write(version_data)

    # 🔟 Most Common Negative Themes
    elif question == "🔟 Most Common Negative Themes":
        negative_text = " ".join(
            data[data['sentiment']=="Negative"]['review']
        )

        words = re.findall(r'\b\w+\b', negative_text.lower())
        common_words = pd.Series(words).value_counts().head(20)

        st.write("Top 20 Negative Keywords")
        st.write(common_words)

st.markdown("---")
st.markdown("⚡ Powered by TF-IDF + ML Model + Streamlit")
