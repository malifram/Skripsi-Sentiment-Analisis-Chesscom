import re
import string

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Analisis Sentimen Chess.com",
    page_icon="♟️",
    layout="centered",
)

# =========================================================
# LOAD RESOURCE (model, vectorizer, stemmer, stopwords)
# Di-cache supaya cuma dimuat sekali, tidak berulang setiap interaksi
# =========================================================
@st.cache_resource
def load_nltk_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    stop_indonesia = set(stopwords.words("indonesian"))
    stop_english = set(stopwords.words("english"))
    stopword_list = stop_indonesia.union(stop_english)

    negation_words = {"tidak", "bukan", "kurang", "gak", "ga"}
    stopword_list = stopword_list - negation_words

    return stopword_list


@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()


@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model_sentimen_terbaik.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


stopword_list = load_nltk_resources()
stemmer = load_stemmer()
model, vectorizer = load_model_and_vectorizer()


# =========================================================
# FUNGSI PREPROCESSING
# Mengikuti tahapan yang sama seperti saat training model
# =========================================================
def preprocess_kalimat(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopword_list]

    text_stemmed = stemmer.stem(" ".join(tokens))
    return text_stemmed


def predict_sentiment(text: str):
    text_clean = preprocess_kalimat(text)
    text_tfidf = vectorizer.transform([text_clean])
    prediction = model.predict(text_tfidf)[0]

    # LinearSVC tidak punya predict_proba secara default,
    # tapi decision_function bisa dipakai sebagai indikator "keyakinan" model
    # (semakin jauh dari 0, semakin yakin model pada kelas tersebut)
    confidence = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(text_tfidf)[0]
        confidence = dict(zip(model.classes_, scores))

    return prediction, text_clean, confidence


# =========================================================
# UI STREAMLIT
# =========================================================
st.title("♟️ Analisis Sentimen Ulasan Chess.com")
st.caption("Prototipe pengujian model Machine Learning (TF-IDF + SVM) untuk klasifikasi sentimen ulasan aplikasi.")

st.divider()

with st.expander("ℹ️ Tentang aplikasi ini"):
    st.write(
        """
        Aplikasi ini menguji model **SVM (Linear SVC)** yang dilatih menggunakan data ulasan
        aplikasi Chess.com dari Google Play Store. Label sentimen (`positive`, `negative`, `neutral`)
        dihasilkan melalui pendekatan **lexicon-based sentiment analysis** yang telah dikurasi,
        kemudian dipakai untuk melatih beberapa model machine learning — dan SVM terpilih sebagai
        model dengan performa terbaik.
        """
    )

st.subheader("Masukkan Kalimat Ulasan")
kalimat_input = st.text_area(
    label="Tulis ulasan atau kalimat yang ingin diuji sentimennya:",
    placeholder="Contoh: gamenya bagus banget, tapi sering nge-lag pas lawan main...",
    height=120,
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_btn = st.button("🔍 Prediksi Sentimen", use_container_width=True)

if predict_btn:
    if not kalimat_input.strip():
        st.warning("Tolong masukkan kalimat terlebih dahulu.")
    else:
        with st.spinner("Menganalisis kalimat..."):
            prediction, text_clean, confidence = predict_sentiment(kalimat_input)

        st.divider()
        st.subheader("Hasil Prediksi")

        color_map = {"positive": "green", "negative": "red", "neutral": "gray"}
        icon_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}

        label_color = color_map.get(prediction, "gray")
        label_icon = icon_map.get(prediction, "")

        st.markdown(
            f"### {label_icon} Sentimen: :{label_color}[{prediction.upper()}]"
        )

        with st.expander("Detail proses"):
            st.write("**Teks setelah preprocessing:**")
            st.code(text_clean if text_clean else "(kosong setelah preprocessing)")

            if confidence:
                st.write("**Skor keyakinan model per kelas** (decision function, bukan probabilitas):")
                st.json({k: round(float(v), 4) for k, v in confidence.items()})

st.divider()
st.caption("Dibuat sebagai prototipe pengujian model untuk keperluan skripsi.")
