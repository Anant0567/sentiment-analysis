import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentiment AI",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# TRAINING DATA
# ============================================================

training_data = [
    # ---------------- POSITIVE ----------------
    ("I love this product", "Positive"),
    ("This is amazing", "Positive"),
    ("I am very happy", "Positive"),
    ("Excellent experience", "Positive"),
    ("Fantastic service", "Positive"),
    ("Absolutely wonderful", "Positive"),
    ("This is the best", "Positive"),
    ("I really enjoyed it", "Positive"),
    ("Very good experience", "Positive"),
    ("Great work", "Positive"),
    ("I am satisfied", "Positive"),
    ("Highly recommended", "Positive"),
    ("The service was excellent", "Positive"),
    ("Everything was perfect", "Positive"),
    ("I am impressed", "Positive"),
    ("This made my day", "Positive"),
    ("Very useful and helpful", "Positive"),
    ("The quality is excellent", "Positive"),
    ("I love the design", "Positive"),
    ("The experience was fantastic", "Positive"),
    ("Very happy with the results", "Positive"),
    ("The app works perfectly", "Positive"),
    ("Fast and reliable service", "Positive"),
    ("I would definitely recommend this", "Positive"),
    ("This product is wonderful", "Positive"),

    # ---------------- NEGATIVE ----------------
    ("I hate this product", "Negative"),
    ("This is terrible", "Negative"),
    ("I am very unhappy", "Negative"),
    ("Worst experience ever", "Negative"),
    ("Terrible service", "Negative"),
    ("Absolutely horrible", "Negative"),
    ("This is the worst", "Negative"),
    ("I really disliked it", "Negative"),
    ("Very bad experience", "Negative"),
    ("Poor work", "Negative"),
    ("I am disappointed", "Negative"),
    ("Not recommended", "Negative"),
    ("The service was terrible", "Negative"),
    ("Everything was broken", "Negative"),
    ("I am not satisfied", "Negative"),
    ("This ruined my day", "Negative"),
    ("Very useless and frustrating", "Negative"),
    ("The quality is terrible", "Negative"),
    ("I hate the design", "Negative"),
    ("The experience was horrible", "Negative"),
    ("Very disappointed with the results", "Negative"),
    ("The app does not work", "Negative"),
    ("Slow and unreliable service", "Negative"),
    ("I would not recommend this", "Negative"),
    ("This product is awful", "Negative"),

    # ---------------- NEUTRAL ----------------
    ("The product arrived today", "Neutral"),
    ("I received the package", "Neutral"),
    ("The meeting is at five", "Neutral"),
    ("The weather is cloudy today", "Neutral"),
    ("I went to the store", "Neutral"),
    ("The phone has a large screen", "Neutral"),
    ("The class starts at ten", "Neutral"),
    ("I bought a new notebook", "Neutral"),
    ("The train arrived on time", "Neutral"),
    ("This product costs five hundred rupees", "Neutral"),
    ("The report was submitted today", "Neutral"),
    ("I watched the movie yesterday", "Neutral"),
    ("The computer has eight GB RAM", "Neutral"),
    ("The office is closed today", "Neutral"),
    ("The book has three hundred pages", "Neutral"),
    ("The event starts tomorrow", "Neutral"),
    ("I downloaded the application", "Neutral"),
    ("The store opens at nine", "Neutral"),
    ("The package contains two items", "Neutral"),
    ("The course lasts six months", "Neutral"),
    ("The laptop weighs two kilograms", "Neutral"),
    ("The exam is next week", "Neutral"),
    ("I visited the website", "Neutral"),
    ("The document was uploaded", "Neutral"),
    ("The meeting lasted one hour", "Neutral"),
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Clean text before sentiment analysis."""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags symbol but keep the word
    text = re.sub(r"#", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = pd.DataFrame(
        training_data,
        columns=["text", "sentiment"]
    )

    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X, y)

    return vectorizer, model


vectorizer, model = train_model()


# ============================================================
# SENTIMENT ANALYSIS FUNCTION
# ============================================================

def analyze_sentiment(text):

    cleaned = clean_text(text)

    if not cleaned:
        return "Neutral", 0.0

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Sentiment AI")

st.write(
    "An AI-powered sentiment analysis application "
    "using TF-IDF and Logistic Regression."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Sentiment AI")

st.sidebar.write("Choose an analysis mode:")

mode = st.sidebar.radio(
    "Analysis Mode",
    [
        "✍️ Analyze Text",
        "📁 Analyze CSV"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("📌 About")

st.sidebar.write(
    "This application analyzes text and classifies "
    "it as Positive, Negative, or Neutral."
)

st.sidebar.write(
    "**Machine Learning:** Logistic Regression"
)

st.sidebar.write(
    "**Text Representation:** TF-IDF"
)

st.sidebar.write(
    "**Framework:** Streamlit"
)

st.sidebar.divider()

st.sidebar.caption(
    "Sentiment AI Project"
)


# ============================================================
# TEXT ANALYSIS MODE
# ============================================================

if mode == "✍️ Analyze Text":

    st.header("✍️ Analyze Text")

    st.write(
        "Enter a sentence, review, comment, or message "
        "to analyze its sentiment."
    )

    text_input = st.text_area(
        "Enter your text",
        height=160,
        placeholder="Example: I really enjoyed this product. The quality is excellent!"
    )

    st.subheader("Quick Examples")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "😊 Positive Example",
            use_container_width=True
        ):
            st.session_state.example_text = (
                "I really enjoyed this product. "
                "The quality is excellent!"
            )

    with col2:
        if st.button(
            "😞 Negative Example",
            use_container_width=True
        ):
            st.session_state.example_text = (
                "This product is terrible. "
                "I am very disappointed."
            )

    with col3:
        if st.button(
            "😐 Neutral Example",
            use_container_width=True
        ):
            st.session_state.example_text = (
                "The package arrived today."
            )

    if "example_text" in st.session_state:
        text_input = st.session_state.example_text

    st.write("")

    analyze_button = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not text_input.strip():

            st.warning(
                "Please enter some text before analyzing."
            )

        else:

            sentiment, confidence = analyze_sentiment(
                text_input
            )

            st.divider()

            st.subheader("📊 Analysis Result")

            col1, col2 = st.columns(2)

            with col1:

                if sentiment == "Positive":

                    st.success(
                        "😊 Positive Sentiment"
                    )

                elif sentiment == "Negative":

                    st.error(
                        "😞 Negative Sentiment"
                    )

                else:

                    st.info(
                        "😐 Neutral Sentiment"
                    )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            st.write("")

            st.subheader("📝 Input Text")

            st.write(text_input)

            st.subheader("🔬 Model Information")

            st.write(
                "The text is cleaned and converted into "
                "TF-IDF numerical features. Logistic Regression "
                "then predicts the sentiment."
            )


# ============================================================
# CSV ANALYSIS MODE
# ============================================================

else:

    st.header("📁 Analyze CSV")

    st.write(
        "Upload a CSV file containing a column with text. "
        "The application will automatically analyze every row."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    st.info(
        "Your CSV should contain a text column. "
        "For example: review, comment, text, message."
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("📄 Uploaded Data")

            st.dataframe(
                df,
                use_container_width=True
            )

            # Find a suitable text column
            possible_columns = [
                "text",
                "review",
                "comment",
                "message",
                "content",
                "sentence"
            ]

            selected_column = None

            for column in possible_columns:

                if column in df.columns:

                    selected_column = column
                    break

            # If no standard column exists
            if selected_column is None:

                selected_column = st.selectbox(
                    "Select the column containing text",
                    df.columns
                )

            st.write(
                f"Selected text column: **{selected_column}**"
            )

            analyze_csv_button = st.button(
                "🔍 Analyze CSV",
                type="primary",
                use_container_width=True
            )

            if analyze_csv_button:

                with st.spinner(
                    "Analyzing text..."
                ):

                    results = []
                    confidences = []

                    for text in df[selected_column]:

                        sentiment, confidence = analyze_sentiment(
                            text
                        )

                        results.append(sentiment)
                        confidences.append(
                            round(confidence, 2)
                        )

                    result_df = df.copy()

                    result_df["Sentiment"] = results
                    result_df["Confidence (%)"] = confidences

                st.success(
                    "CSV analysis completed successfully."
                )

                st.subheader("📊 Results")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                # ------------------------------------------------
                # SENTIMENT COUNTS
                # ------------------------------------------------

                sentiment_counts = (
                    result_df["Sentiment"]
                    .value_counts()
                )

                st.subheader(
                    "📈 Sentiment Summary"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "😊 Positive",
                        int(
                            sentiment_counts.get(
                                "Positive",
                                0
                            )
                        )
                    )

                with col2:

                    st.metric(
                        "😞 Negative",
                        int(
                            sentiment_counts.get(
                                "Negative",
                                0
                            )
                        )
                    )

                with col3:

                    st.metric(
                        "😐 Neutral",
                        int(
                            sentiment_counts.get(
                                "Neutral",
                                0
                            )
                        )
                    )

                # ------------------------------------------------
                # CHART
                # ------------------------------------------------

                st.subheader(
                    "📊 Sentiment Distribution"
                )

                chart_data = pd.DataFrame(
                    {
                        "Sentiment": [
                            "Positive",
                            "Negative",
                            "Neutral"
                        ],
                        "Count": [
                            sentiment_counts.get(
                                "Positive",
                                0
                            ),
                            sentiment_counts.get(
                                "Negative",
                                0
                            ),
                            sentiment_counts.get(
                                "Neutral",
                                0
                            )
                        ]
                    }
                )

                st.bar_chart(
                    chart_data.set_index(
                        "Sentiment"
                    )
                )

                # ------------------------------------------------
                # DOWNLOAD
                # ------------------------------------------------

                st.subheader(
                    "⬇️ Download Results"
                )

                csv_data = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Analyzed CSV",
                    data=csv_data,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Unable to process the CSV file."
            )

            st.write(
                f"Error details: {e}"
            )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.header("📚 How Sentiment AI Works")

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("1️⃣ Text Cleaning")

    st.write(
        "URLs, mentions, special characters and "
        "unnecessary spaces are removed from the text."
    )

with col2:

    st.subheader("2️⃣ TF-IDF")

    st.write(
        "TF-IDF converts text into numerical features "
        "that can be processed by the machine-learning model."
    )

with col3:

    st.subheader("3️⃣ Classification")

    st.write(
        "Logistic Regression classifies the text into "
        "Positive, Negative, or Neutral sentiment."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sentiment AI | Python • Streamlit • Pandas • Scikit-learn"
)