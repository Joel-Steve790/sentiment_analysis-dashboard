"""
=====================================================================
Real-Time Sentiment Analysis Dashboard
=====================================================================
Streamlit dashboard supporting:
  - Single text analysis (all 4 models simultaneously)
  - Batch CSV upload and analysis
  - Sentiment trend visualization

Install:
    pip install streamlit plotly

Run:
    streamlit run dashboard.py

Model files required (all in same folder as this script):
    sentiment_model.pkl          — Logistic Regression
    tfidf_vectorizer.pkl         — LR vectorizer
    svm_model.pkl                — SVM
    svm_tfidf_vectorizer.pkl     — SVM vectorizer
    nb_model.pkl                 — Naive Bayes
    nb_tfidf_vectorizer.pkl      — NB vectorizer
    distilbert_sentiment_best.pt — DistilBERT weights
    distilbert_tokenizer_saved/  — DistilBERT tokenizer folder
=====================================================================
"""

import re
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# Download NLTK resources silently
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# =====================================================================
# PREPROCESSING (must match training scripts exactly)
# =====================================================================
LEMMATIZER = WordNetLemmatizer()
NEGATION_WORDS = {
    "not", "no", "nor", "don", "dont", "didn", "didnt", "doesn", "doesnt",
    "isn", "isnt", "wasn", "wasnt", "aren", "arent", "weren", "werent",
    "wouldn", "wouldnt", "couldn", "couldnt", "shouldn", "shouldnt",
    "won", "wont", "cant", "cannot", "never", "neither"
}
STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS


def convert_emoticons(text):
    text = re.sub(r"[:=;][\-o\*']?[)\]dD]+", " emoticonhappy ", text)
    text = re.sub(r"[:=;][\-o\*']?[(\[/]+", " emoticonsad ", text)
    return text


def preprocess_text(text):
    """TF-IDF preprocessing — must match training exactly."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = convert_emoticons(text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in text.split()
        if word not in STOP_WORDS and (len(word) > 2 or word in NEGATION_WORDS)
    ]
    return " ".join(tokens)


def light_clean(text):
    """DistilBERT preprocessing — minimal, preserves context."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =====================================================================
# MODEL LOADING (cached so models load only once per session)
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_classical_models():
    """Load LR, SVM, NB models and their vectorizers."""
    models = {}
    try:
        with open("sentiment_model.pkl", "rb") as f:
            models["Logistic Regression"] = {"model": pickle.load(f)}
        with open("tfidf_vectorizer.pkl", "rb") as f:
            models["Logistic Regression"]["vectorizer"] = pickle.load(f)
    except FileNotFoundError:
        models["Logistic Regression"] = None

    try:
        with open("svm_model.pkl", "rb") as f:
            models["SVM"] = {"model": pickle.load(f)}
        with open("svm_tfidf_vectorizer.pkl", "rb") as f:
            models["SVM"]["vectorizer"] = pickle.load(f)
    except FileNotFoundError:
        models["SVM"] = None

    try:
        with open("nb_model.pkl", "rb") as f:
            models["Naive Bayes"] = {"model": pickle.load(f)}
        with open("nb_tfidf_vectorizer.pkl", "rb") as f:
            models["Naive Bayes"]["vectorizer"] = pickle.load(f)
    except FileNotFoundError:
        models["Naive Bayes"] = None

    return models


@st.cache_resource(show_spinner=False)
def load_distilbert():
    """Load fine-tuned DistilBERT model and tokenizer."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("distilbert_tokenizer_saved")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        state_dict = torch.load(
            "distilbert_sentiment_best.pt",
            map_location=device
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return {"model": model, "tokenizer": tokenizer, "device": device}
    except Exception:
        return None


# =====================================================================
# PREDICTION FUNCTIONS
# =====================================================================
def predict_classical(text, model_dict):
    """Predict using LR / SVM / NB."""
    if model_dict is None:
        return None, None
    cleaned = preprocess_text(text)
    vec = model_dict["vectorizer"].transform([cleaned])
    prediction = model_dict["model"].predict(vec)[0]
    proba = model_dict["model"].predict_proba(vec)[0]
    label = "Positive" if prediction == 1 else "Negative"
    confidence = proba[prediction]
    return label, float(confidence)


def predict_distilbert(text, bert_dict):
    """Predict using fine-tuned DistilBERT."""
    if bert_dict is None:
        return None, None
    try:
        import torch
        cleaned = light_clean(text)
        encoding = bert_dict["tokenizer"](
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        ).to(bert_dict["device"])
        with torch.no_grad():
            outputs = bert_dict["model"](**encoding)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
        label = "Positive" if pred == 1 else "Negative"
        confidence = float(probs[pred])
        return label, confidence
    except Exception:
        return None, None


def predict_all_models(text, classical_models, bert_dict):
    """Run all 4 models on a single input text."""
    results = {}
    for name, model_dict in classical_models.items():
        label, conf = predict_classical(text, model_dict)
        results[name] = {"label": label, "confidence": conf}
    label, conf = predict_distilbert(text, bert_dict)
    results["DistilBERT"] = {"label": label, "confidence": conf}
    return results


# =====================================================================
# STREAMLIT PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.title("💬 Sentiment Analysis")
    st.markdown("**Real-Time Model for Organizations**")
    st.divider()

    st.markdown("### Models")
    st.markdown("""
    - 🔵 Logistic Regression
    - 🟠 Support Vector Machine
    - 🟢 Naive Bayes
    - 🟣 DistilBERT (fine-tuned)
    """)

    st.divider()
    st.markdown("### Dataset")
    st.markdown("Trained on Sentiment140 — 1.6M tweets")

    st.divider()
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Single Text Analysis", "Batch CSV Analysis", "Sentiment Trend"],
        label_visibility="collapsed"
    )

# =====================================================================
# LOAD MODELS (with loading spinner)
# =====================================================================
with st.spinner("Loading models... (first load only, cached after)"):
    classical_models = load_classical_models()
    bert_dict = load_distilbert()

# Show which models loaded successfully in sidebar
with st.sidebar:
    st.divider()
    st.markdown("### Model Status")
    status_icons = {True: "✅", False: "❌"}
    for name, m in classical_models.items():
        st.markdown(f"{status_icons[m is not None]} {name}")
    st.markdown(f"{status_icons[bert_dict is not None]} DistilBERT")


# =====================================================================
# HELPER: Render a single model result card
# =====================================================================
def render_result_card(model_name, label, confidence, col):
    color = "#2ecc71" if label == "Positive" else "#e74c3c"
    emoji = "😊" if label == "Positive" else "😞"
    with col:
        st.markdown(f"""
        <div style="
            background: {color}22;
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 8px;
        ">
            <p style="margin:0; font-size:13px; color:#888;">{model_name}</p>
            <p style="margin:4px 0; font-size:22px; font-weight:bold; color:{color};">
                {emoji} {label}
            </p>
            <p style="margin:0; font-size:14px; color:#555;">
                Confidence: <strong>{confidence:.1%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# PAGE 1: SINGLE TEXT ANALYSIS
# =====================================================================
if page == "Single Text Analysis":
    st.title("💬 Single Text Sentiment Analysis")
    st.markdown("Enter any customer review or text below to analyze its sentiment across all four models.")

    text_input = st.text_area(
        "Enter text to analyze",
        placeholder="Type or paste a customer review here...",
        height=120
    )

    analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_btn and text_input.strip():
        with st.spinner("Analyzing..."):
            results = predict_all_models(text_input.strip(), classical_models, bert_dict)

        st.markdown("### Results")
        cols = st.columns(4)
        model_names = ["Logistic Regression", "SVM", "Naive Bayes", "DistilBERT"]

        for i, name in enumerate(model_names):
            result = results.get(name)
            if result and result["label"]:
                render_result_card(name, result["label"], result["confidence"], cols[i])
            else:
                with cols[i]:
                    st.warning(f"{name} not loaded")

        # Confidence bar chart
        st.markdown("### Confidence Comparison")
        chart_data = []
        for name in model_names:
            result = results.get(name)
            if result and result["label"]:
                chart_data.append({
                    "Model": name,
                    "Confidence": result["confidence"] * 100,
                    "Sentiment": result["label"]
                })

        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            color_map = {"Positive": "#2ecc71", "Negative": "#e74c3c"}
            fig = px.bar(
                df_chart,
                x="Model",
                y="Confidence",
                color="Sentiment",
                color_discrete_map=color_map,
                text="Confidence",
                range_y=[0, 100]
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                yaxis_title="Confidence (%)",
                showlegend=True,
                height=350,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Show cleaned text (useful for debugging/understanding)
        with st.expander("See preprocessed text (TF-IDF models)"):
            st.code(preprocess_text(text_input.strip()))

    elif analyze_btn:
        st.warning("Please enter some text before analyzing.")


# =====================================================================
# PAGE 2: BATCH CSV ANALYSIS
# =====================================================================
elif page == "Batch CSV Analysis":
    st.title("📂 Batch CSV Sentiment Analysis")
    st.markdown("""
    Upload a CSV file containing customer reviews.
    The file must have a column containing the review text.
    """)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded:** {len(df):,} rows, {len(df.columns)} columns")
        st.dataframe(df.head(5), use_container_width=True)

        # Let user pick which column has the text
        text_col = st.selectbox(
            "Select the column containing review text",
            options=df.columns.tolist()
        )

        # Let user pick which models to run (DistilBERT is slow on large batches)
        model_options = st.multiselect(
            "Select models to run (DistilBERT is slower on large batches)",
            options=["Logistic Regression", "SVM", "Naive Bayes", "DistilBERT"],
            default=["Logistic Regression", "SVM", "Naive Bayes"]
        )

        max_rows = st.slider(
            "Max rows to analyze (limit for large files)",
            min_value=10,
            max_value=min(1000, len(df)),
            value=min(100, len(df)),
            step=10
        )

        run_batch = st.button("Run Batch Analysis", type="primary", use_container_width=True)

        if run_batch:
            df_sample = df[[text_col]].head(max_rows).copy()
            df_sample = df_sample.dropna(subset=[text_col])
            df_sample[text_col] = df_sample[text_col].astype(str)

            progress_bar = st.progress(0, text="Analyzing reviews...")

            for i, model_name in enumerate(model_options):
                labels, confidences = [], []
                texts = df_sample[text_col].tolist()

                if model_name == "DistilBERT":
                    for text in texts:
                        label, conf = predict_distilbert(text, bert_dict)
                        labels.append(label or "Error")
                        confidences.append(conf or 0.0)
                else:
                    for text in texts:
                        label, conf = predict_classical(text, classical_models[model_name])
                        labels.append(label or "Error")
                        confidences.append(conf or 0.0)

                df_sample[f"{model_name}_Sentiment"] = labels
                df_sample[f"{model_name}_Confidence"] = [f"{c:.1%}" for c in confidences]
                progress_bar.progress((i + 1) / len(model_options),
                                      text=f"Completed {model_name}...")

            progress_bar.empty()
            st.success(f"Analysis complete — {len(df_sample):,} reviews processed")
            st.dataframe(df_sample, use_container_width=True)

            # Download results
            csv_out = df_sample.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_out,
                file_name="sentiment_results.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Summary stats
            st.markdown("### Summary")
            summary_cols = st.columns(len(model_options))
            for i, model_name in enumerate(model_options):
                col_name = f"{model_name}_Sentiment"
                if col_name in df_sample.columns:
                    counts = df_sample[col_name].value_counts()
                    pos = counts.get("Positive", 0)
                    neg = counts.get("Negative", 0)
                    total = pos + neg
                    with summary_cols[i]:
                        st.markdown(f"**{model_name}**")
                        st.metric("Positive", f"{pos} ({pos/total:.0%})" if total > 0 else "0")
                        st.metric("Negative", f"{neg} ({neg/total:.0%})" if total > 0 else "0")

            # Store in session state for trend page
            st.session_state["batch_results"] = df_sample
            st.session_state["batch_model_options"] = model_options
            st.info("Go to **Sentiment Trend** in the sidebar to visualize these results.")


# =====================================================================
# PAGE 3: SENTIMENT TREND
# =====================================================================
elif page == "Sentiment Trend":
    st.title("📈 Sentiment Trend Analysis")
    st.markdown("Visualize how sentiment changes across your batch of reviews.")

    # Check if batch results exist from page 2
    if "batch_results" not in st.session_state:
        st.info("No batch results found. Please go to **Batch CSV Analysis** first, upload a file, and run the analysis.")
        st.stop()

    df_results = st.session_state["batch_results"]
    model_options = st.session_state.get("batch_model_options", [])

    if not model_options:
        st.warning("No model results found.")
        st.stop()

    # Model selector for trend chart
    selected_model = st.selectbox(
        "Select model for trend visualization",
        options=model_options
    )

    sentiment_col = f"{selected_model}_Sentiment"

    if sentiment_col not in df_results.columns:
        st.warning("Selected model results not found.")
        st.stop()

    # Add row index for x-axis
    df_plot = df_results.copy().reset_index(drop=True)
    df_plot["Review #"] = df_plot.index + 1

    # --- Chart 1: Sentiment Distribution Pie ---
    st.markdown("### Overall Sentiment Distribution")
    counts = df_plot[sentiment_col].value_counts()

    fig_pie = go.Figure(data=[go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker_colors=["#2ecc71" if l == "Positive" else "#e74c3c"
                       for l in counts.index.tolist()],
        hole=0.4,
        textinfo="label+percent"
    )])
    fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Chart 2: Rolling Sentiment Trend ---
    st.markdown("### Sentiment Trend (Rolling Average)")

    window = st.slider("Rolling window size (reviews)", min_value=3, max_value=20, value=5)

    df_plot["Sentiment_Numeric"] = df_plot[sentiment_col].map(
        {"Positive": 1, "Negative": 0}
    )
    df_plot["Rolling_Avg"] = df_plot["Sentiment_Numeric"].rolling(
        window=window, min_periods=1
    ).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_plot["Review #"],
        y=df_plot["Rolling_Avg"] * 100,
        mode="lines",
        name="% Positive (rolling avg)",
        line=dict(color="#3498db", width=2),
        fill="tozeroy",
        fillcolor="rgba(52, 152, 219, 0.1)"
    ))
    fig_trend.add_hline(
        y=50, line_dash="dash",
        line_color="gray",
        annotation_text="50% neutral line"
    )
    fig_trend.update_layout(
        xaxis_title="Review Number",
        yaxis_title="% Positive Sentiment",
        yaxis=dict(range=[0, 100]),
        height=350,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- Chart 3: Sentiment per Review (scatter) ---
    st.markdown("### Per-Review Sentiment")

    color_map = {"Positive": "#2ecc71", "Negative": "#e74c3c"}
    fig_scatter = px.scatter(
        df_plot,
        x="Review #",
        y="Sentiment_Numeric",
        color=sentiment_col,
        color_discrete_map=color_map,
        labels={"Sentiment_Numeric": "Sentiment (1=Pos, 0=Neg)"},
        height=300
    )
    fig_scatter.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Chart 4: All models side-by-side (if multiple) ---
    if len(model_options) > 1:
        st.markdown("### Model Agreement Comparison")
        st.markdown("Shows what percentage of reviews each model classified as Positive.")

        comparison_data = []
        for m in model_options:
            col = f"{m}_Sentiment"
            if col in df_results.columns:
                pos_pct = (df_results[col] == "Positive").mean() * 100
                comparison_data.append({"Model": m, "% Positive": pos_pct})

        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            fig_comp = px.bar(
                df_comp,
                x="Model",
                y="% Positive",
                color="% Positive",
                color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                range_y=[0, 100],
                text="% Positive",
                height=350
            )
            fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_comp.update_layout(margin=dict(t=20, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_comp, use_container_width=True)

    # Show raw data
    with st.expander("View raw results table"):
        st.dataframe(df_results, use_container_width=True)
