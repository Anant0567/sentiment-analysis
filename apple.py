import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentiment AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* ================= BACKGROUND ================= */

    .stApp {
        background: linear-gradient(
            135deg,
            #eef2ff 0%,
            #f8fafc 45%,
            #fce7f3 100%
        );
    }


    /* ================= HEADINGS ================= */

    h1 {
        color: #312e81 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #4338ca !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #4c1d95 !important;
    }


    /* ================= NORMAL TEXT ================= */

    p {
        color: #334155;
    }


    /* ================= SIDEBAR ================= */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #312e81,
            #4338ca,
            #7c3aed
        );
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: white !important;
    }


    /* ================= BUTTONS ================= */

    .stButton > button {
        border-radius: 12px !important;

        border: 1px solid #a5b4fc !important;

        background: white !important;

        color: #4338ca !important;

        font-weight: 700 !important;

        transition: all 0.25s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px);

        border-color: #7c3aed !important;

        color: #7c3aed !important;

        box-shadow:
            0 8px 20px
            rgba(124, 58, 237, 0.20);
    }


    /* ================= PRIMARY BUTTON ================= */

    .stButton > button[kind="primary"] {
        background: linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed,
            #db2777
        ) !important;

        color: white !important;

        border: none !important;

        animation:
            buttonGlow 3s ease-in-out infinite;
    }

    @keyframes buttonGlow {

        0% {
            box-shadow:
                0 5px 15px
                rgba(79, 70, 229, 0.20);
        }

        50% {
            box-shadow:
                0 8px 30px
                rgba(219, 39, 119, 0.35);
        }

        100% {
            box-shadow:
                0 5px 15px
                rgba(79, 70, 229, 0.20);
        }
    }


    /* ================= TEXT AREA ================= */

    [data-testid="stTextArea"] textarea {

        background-color: white !important;

        color: #1e293b !important;

        border: 2px solid #c7d2fe !important;

        border-radius: 14px !important;

        font-size: 16px !important;
    }

    [data-testid="stTextArea"] textarea:focus {

        border-color: #7c3aed !important;

        box-shadow:
            0 0 0 2px
            rgba(124, 58, 237, 0.12) !important;
    }


    /* ================= FILE UPLOADER ================= */

    [data-testid="stFileUploader"] {

        background: white;

        border-radius: 14px;

        border: 2px dashed #a5b4fc;

        padding: 10px;
    }


    /* ================= METRICS ================= */

    [data-testid="stMetric"] {

        background: white;

        border-radius: 15px;

        padding: 15px;

        border: 1px solid #e0e7ff;

        box-shadow:
            0 5px 18px
            rgba(0, 0, 0, 0.06);
    }


    /* ================= ALERTS ================= */

    [data-testid="stAlert"] {

        border-radius: 12px;
    }


    /* ================= RESULT ANIMATION ================= */

    @keyframes resultAppear {

        from {
            opacity: 0;
            transform:
                translateY(15px)
                scale(0.97);
        }

        to {
            opacity: 1;
            transform:
                translateY(0)
                scale(1);
        }
    }


    /* ================= SUMMARY BOX ================= */

    [data-testid="stAlert"] {

        animation:
            resultAppear 0.5s ease;
    }


    /* ================= FOOTER ================= */

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
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
    ("Amazing quality", "Positive"),
    ("Excellent product", "Positive"),
    ("Brilliant service", "Positive"),
    ("I am extremely pleased", "Positive"),
    ("This is fantastic", "Positive"),

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
    ("Very poor quality", "Negative"),
    ("Extremely disappointing", "Negative"),
    ("Bad service", "Negative"),
    ("Completely useless", "Negative"),
    ("I regret buying this", "Negative"),

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
    ("The product was delivered yesterday", "Neutral"),
    ("The application was installed", "Neutral"),
    ("The class ended at four", "Neutral"),
    ("The phone costs ten thousand rupees", "Neutral"),
    ("The package contains three items", "Neutral"),
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# TRAIN SENTIMENT MODEL
# ============================================================

@st.cache_resource
def train_model():

    data = pd.DataFrame(
        training_data,
        columns=[
            "text",
            "sentiment"
        ]
    )

    data["clean_text"] = (
        data["text"].apply(
            clean_text
        )
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(
        data["clean_text"]
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X,
        data["sentiment"]
    )

    return vectorizer, model


vectorizer, model = train_model()


# ============================================================
# SENTIMENT ANALYSIS FUNCTION
# ============================================================

def analyze_sentiment(text):

    cleaned = clean_text(text)

    if not cleaned:

        return "Neutral", 0.0

    X = vectorizer.transform(
        [cleaned]
    )

    prediction = model.predict(
        X
    )[0]

    probabilities = model.predict_proba(
        X
    )[0]

    confidence = (
        max(probabilities) * 100
    )

    return prediction, confidence


# ============================================================
# REVIEW SUMMARIZATION
# ============================================================

def summarize_review(
    text,
    sentence_count=2
):
    """
    Creates an extractive summary.

    The most important sentences are selected
    using TF-IDF scores.
    """

    text = str(text).strip()

    if not text:

        return ""

    # Split into sentences
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(
            sentence.strip().split()
        ) >= 4
    ]

    # Short review does not need summarization
    if len(sentences) <= sentence_count:

        return text

    cleaned_sentences = [
        clean_text(sentence)
        for sentence in sentences
    ]

    # Remove empty sentences
    valid_pairs = [
        (
            sentence,
            cleaned
        )
        for sentence, cleaned
        in zip(
            sentences,
            cleaned_sentences
        )
        if cleaned
    ]

    if not valid_pairs:

        return text

    sentences = [
        pair[0]
        for pair in valid_pairs
    ]

    cleaned_sentences = [
        pair[1]
        for pair in valid_pairs
    ]

    try:

        summary_vectorizer = (
            TfidfVectorizer(
                stop_words="english"
            )
        )

        matrix = (
            summary_vectorizer.fit_transform(
                cleaned_sentences
            )
        )

    except ValueError:

        return text

    # Calculate importance score
    sentence_scores = (
        matrix.sum(axis=1)
    )

    sentence_scores = [
        float(score)
        for score in sentence_scores
    ]

    # Rank sentences
    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda index:
            sentence_scores[index],
        reverse=True
    )

    # Select important sentences
    selected_indices = sorted(
        ranked_indices[
            :sentence_count
        ]
    )

    summary = " ".join(
        sentences[index]
        for index in selected_indices
    )

    return summary


# ============================================================
# SESSION STATE
# ============================================================

if "text_value" not in st.session_state:

    st.session_state.text_value = ""


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧠 Sentiment AI"
)

st.sidebar.caption(
    "Machine Learning + Review Summarization"
)

st.sidebar.divider()

mode = st.sidebar.radio(
    "Choose Analysis Mode",
    [
        "✍️ Analyze Text",
        "📁 Analyze CSV"
    ]
)

st.sidebar.divider()

st.sidebar.subheader(
    "🔬 Technology"
)

st.sidebar.write(
    "🐍 Python"
)

st.sidebar.write(
    "🎈 Streamlit"
)

st.sidebar.write(
    "📊 Pandas"
)

st.sidebar.write(
    "🧠 Scikit-learn"
)

st.sidebar.write(
    "🔢 TF-IDF"
)

st.sidebar.write(
    "📈 Logistic Regression"
)

st.sidebar.divider()

st.sidebar.info(
    "🔒 No external API is required."
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🧠 Sentiment AI"
)

st.subheader(
    "Analyze. Understand. Summarize."
)

st.info(
    "✨ AI-powered sentiment analysis "
    "with automatic customer review summarization."
)


# ============================================================
# TEXT ANALYSIS
# ============================================================

if mode == "✍️ Analyze Text":

    st.header(
        "✍️ Analyze Customer Review"
    )

    st.write(
        "Enter a customer review, comment, "
        "message, or any other text."
    )

    text_input = st.text_area(
        "Your review",
        value=st.session_state.text_value,
        height=190,
        placeholder=(
            "Example: I purchased this product "
            "last month. The design is excellent, "
            "but the battery life is disappointing..."
        )
    )

    st.session_state.text_value = (
        text_input
    )

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    st.subheader(
        "✨ Quick Examples"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success(
            "😊 **Positive**\n\n"
            "I love this product! "
            "The quality is amazing."
        )

        if st.button(
            "Use Positive Example",
            key="positive_button",
            use_container_width=True
        ):

            st.session_state.text_value = (
                "I really enjoyed this product. "
                "The quality is excellent and "
                "the service was fantastic."
            )

            st.rerun()

    with col2:

        st.error(
            "😞 **Negative**\n\n"
            "This product is terrible. "
            "I am very disappointed."
        )

        if st.button(
            "Use Negative Example",
            key="negative_button",
            use_container_width=True
        ):

            st.session_state.text_value = (
                "This product is terrible. "
                "The battery drains very quickly "
                "and the service was disappointing."
            )

            st.rerun()

    with col3:

        st.info(
            "😐 **Neutral**\n\n"
            "The package arrived today."
        )

        if st.button(
            "Use Neutral Example",
            key="neutral_button",
            use_container_width=True
        ):

            st.session_state.text_value = (
                "The package arrived today "
                "and contains two items."
            )

            st.rerun()

    st.write("")

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze_button = st.button(
        "🚀 Analyze Review",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not text_input.strip():

            st.warning(
                "⚠️ Please enter a review first."
            )

        else:

            with st.spinner(
                "🧠 Analyzing your review..."
            ):

                sentiment, confidence = (
                    analyze_sentiment(
                        text_input
                    )
                )

            st.divider()

            st.header(
                "📊 Analysis Result"
            )

            # ------------------------------------------------
            # SENTIMENT RESULT
            # ------------------------------------------------

            if sentiment == "Positive":

                st.success(
                    "😊 POSITIVE SENTIMENT"
                )

                explanation = (
                    "The review contains "
                    "mostly positive language."
                )

            elif sentiment == "Negative":

                st.error(
                    "😞 NEGATIVE SENTIMENT"
                )

                explanation = (
                    "The review contains "
                    "mostly negative language."
                )

            else:

                st.info(
                    "😐 NEUTRAL SENTIMENT"
                )

                explanation = (
                    "The review appears "
                    "relatively neutral."
                )

            st.write(
                explanation
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "🎯 Confidence",
                    f"{confidence:.1f}%"
                )

            with col2:

                st.metric(
                    "🧠 Sentiment",
                    sentiment
                )

            with col3:

                word_count = len(
                    text_input.split()
                )

                st.metric(
                    "📝 Word Count",
                    word_count
                )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.divider()

            st.header(
                "📋 Customer Review Summary"
            )

            if word_count >= 40:

                with st.spinner(
                    "✍️ Creating a short summary..."
                ):

                    summary = (
                        summarize_review(
                            text_input,
                            sentence_count=2
                        )
                    )

                st.info(
                    "💡 **Quick Summary**\n\n"
                    + summary
                )

                summary_words = len(
                    summary.split()
                )

                st.caption(
                    f"Original review: "
                    f"{word_count} words  |  "
                    f"Summary: "
                    f"{summary_words} words"
                )

            else:

                st.info(
                    "💡 This review is already "
                    "short, so a separate summary "
                    "is not necessary."
                )

            # ------------------------------------------------
            # ORIGINAL REVIEW
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📝 Original Customer Review"
            )

            st.write(
                text_input
            )


# ============================================================
# CSV ANALYSIS
# ============================================================

else:

    st.header(
        "📁 Analyze Customer Reviews from CSV"
    )

    st.write(
        "Upload a CSV file containing customer "
        "reviews. The application will analyze "
        "sentiment and create summaries for "
        "long reviews."
    )

    uploaded_file = st.file_uploader(
        "📤 Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"✅ CSV loaded successfully: "
                f"{len(data)} rows"
            )

            st.subheader(
                "📄 Uploaded Data"
            )

            st.dataframe(
                data,
                use_container_width=True
            )

            # ------------------------------------------------
            # FIND TEXT COLUMN
            # ------------------------------------------------

            possible_columns = [
                "text",
                "review",
                "comment",
                "message",
                "content",
                "sentence",
                "customer_review",
                "customer_review_text"
            ]

            available_columns = [
                column
                for column in possible_columns
                if column in data.columns
            ]

            if available_columns:

                selected_column = (
                    available_columns[0]
                )

            else:

                selected_column = (
                    st.selectbox(
                        "📝 Select the review column",
                        data.columns
                    )
                )

            st.write(
                f"Selected column: "
                f"**{selected_column}**"
            )

            # ------------------------------------------------
            # CSV ANALYZE BUTTON
            # ------------------------------------------------

            csv_button = st.button(
                "🚀 Analyze All Reviews",
                type="primary",
                use_container_width=True
            )

            if csv_button:

                sentiments = []

                confidence_values = []

                summaries = []

                progress = st.progress(
                    0
                )

                total_rows = len(
                    data
                )

                # --------------------------------------------
                # PROCESS EACH REVIEW
                # --------------------------------------------

                for index, value in enumerate(
                    data[selected_column]
                ):

                    text = str(value)

                    # Sentiment
                    sentiment, confidence = (
                        analyze_sentiment(
                            text
                        )
                    )

                    sentiments.append(
                        sentiment
                    )

                    confidence_values.append(
                        round(
                            confidence,
                            2
                        )
                    )

                    # Summary
                    word_count = len(
                        text.split()
                    )

                    if word_count >= 40:

                        summary = (
                            summarize_review(
                                text,
                                sentence_count=2
                            )
                        )

                    else:

                        summary = text

                    summaries.append(
                        summary
                    )

                    # Progress
                    progress.progress(
                        (index + 1)
                        / total_rows
                    )

                # --------------------------------------------
                # CREATE RESULT DATA
                # --------------------------------------------

                result_data = data.copy()

                result_data[
                    "Sentiment"
                ] = sentiments

                result_data[
                    "Confidence (%)"
                ] = confidence_values

                result_data[
                    "Review Summary"
                ] = summaries

                st.success(
                    "🎉 Analysis completed successfully!"
                )

                # --------------------------------------------
                # RESULTS
                # --------------------------------------------

                st.subheader(
                    "📊 Analysis Results"
                )

                st.dataframe(
                    result_data,
                    use_container_width=True
                )

                # --------------------------------------------
                # SUMMARY COUNTS
                # --------------------------------------------

                counts = (
                    result_data[
                        "Sentiment"
                    ].value_counts()
                )

                st.subheader(
                    "📈 Sentiment Summary"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "😊 Positive",
                        int(
                            counts.get(
                                "Positive",
                                0
                            )
                        )
                    )

                with col2:

                    st.metric(
                        "😞 Negative",
                        int(
                            counts.get(
                                "Negative",
                                0
                            )
                        )
                    )

                with col3:

                    st.metric(
                        "😐 Neutral",
                        int(
                            counts.get(
                                "Neutral",
                                0
                            )
                        )
                    )

                # --------------------------------------------
                # CHART
                # --------------------------------------------

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
                            counts.get(
                                "Positive",
                                0
                            ),
                            counts.get(
                                "Negative",
                                0
                            ),
                            counts.get(
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

                # --------------------------------------------
                # DOWNLOAD
                # --------------------------------------------

                csv_output = (
                    result_data
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                )

                st.download_button(
                    "⬇️ Download Results with Summaries",
                    data=csv_output,
                    file_name=(
                        "sentiment_analysis_results.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                "❌ There was a problem "
                "reading the CSV file."
            )

            st.code(
                str(error)
            )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.header(
    "⚙️ How Sentiment AI Works"
)

st.write(
    "The application combines sentiment classification "
    "with automatic review summarization."
)

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.subheader(
        "1️⃣ Clean"
    )

    st.write(
        "The review is cleaned by removing "
        "URLs, mentions and unnecessary characters."
    )

with step2:

    st.subheader(
        "2️⃣ TF-IDF"
    )

    st.write(
        "Text is converted into numerical "
        "features using TF-IDF."
    )

with step3:

    st.subheader(
        "3️⃣ Sentiment"
    )

    st.write(
        "Logistic Regression predicts "
        "Positive, Negative or Neutral."
    )

with step4:

    st.subheader(
        "4️⃣ Summary"
    )

    st.write(
        "Important sentences from long "
        "reviews are selected to create a short summary."
    )


# ============================================================
# PROJECT FEATURES
# ============================================================

st.divider()

st.header(
    "✨ Project Features"
)

feature1, feature2, feature3 = st.columns(3)

with feature1:

    st.success(
        "📝 **Text Analysis**\n\n"
        "Analyze individual customer reviews "
        "and messages."
    )

with feature2:

    st.info(
        "📁 **CSV Analysis**\n\n"
        "Analyze multiple customer reviews "
        "at the same time."
    )

with feature3:

    st.warning(
        "📋 **Review Summarization**\n\n"
        "Long reviews are reduced to their "
        "most important sentences."
    )


feature4, feature5, feature6 = st.columns(3)

with feature4:

    st.success(
        "🎯 **Confidence Score**\n\n"
        "See how confident the model is "
        "about its prediction."
    )

with feature5:

    st.info(
        "📊 **Visualization**\n\n"
        "View the distribution of positive, "
        "negative and neutral reviews."
    )

with feature6:

    st.warning(
        "⬇️ **Export Results**\n\n"
        "Download sentiment and summary "
        "results as a CSV file."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 Sentiment AI | "
    "Python • Streamlit • Pandas • Scikit-learn | "
    "Sentiment Analysis + Review Summarization"
)