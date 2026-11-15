import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Klasifikasi Naive Bayes - Machine Learning UNP Kediri",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 22px 26px;
        border-radius: 12px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 26px;
        font-weight: 700;
    }
    .card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1e3a8a;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
    <h1>Aplikasi Klasifikasi Naive Bayes - Kelompok 2</h1>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_datasets():
    candidate_dirs = [
        os.path.dirname(__file__),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        r"D:\TUGAS KULIAH\Machine Learning\TUGAS DEMO 2\NB",
    ]
    df_iris = None
    df_bal = None
    for d in candidate_dirs:
        p_iris = os.path.join(d, "iris.csv")
        p_bal = os.path.join(d, "balance-scale.csv")
        if df_iris is None and os.path.exists(p_iris):
            df_iris = pd.read_csv(p_iris)
        if df_bal is None and os.path.exists(p_bal):
            df_bal = pd.read_csv(p_bal)
    return df_iris, df_bal


df_iris, df_bal = load_datasets()

with st.sidebar:
    st.header("Pengaturan Model")
    dataset_choice = st.selectbox(
        "Pilih Dataset:",
        ["Iris Dataset", "Balance Scale Dataset"],
    )
    st.markdown("---")
    st.subheader("Parameter Pembagian Data")
    split_mode = st.radio(
        "Rasio Data Latih : Uji",
        ["80% : 20% (Skenario 1)", "70% : 30% (Skenario 2)", "60% : 40% (Skenario 3)"],
    )
    if split_mode == "80% : 20% (Skenario 1)":
        test_size = 0.20
        nama_skenario = "Skenario 1"
    elif split_mode == "70% : 30% (Skenario 2)":
        test_size = 0.30
        nama_skenario = "Skenario 2"
    else:
        test_size = 0.40
        nama_skenario = "Skenario 3"
    st.markdown("---")
    st.info("Jenis Naive Bayes: GaussianNB")
    st.caption("Kelompok 2")

if dataset_choice == "Iris Dataset":
    df_active = df_iris
    target_col = "Species"
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    default_vals = [5.1, 3.5, 1.4, 0.2]
else:
    df_active = df_bal
    target_col = "Class"
    feature_cols = ["L-Weight", "L-Distance", "R-Weight", "R-Distance"]
    default_vals = [4, 3, 2, 1]

if df_active is None:
    st.error("File dataset tidak ditemukan. Pastikan iris.csv dan balance-scale.csv ada di folder aplikasi.")
    st.stop()

# Pemisahan fitur mengikuti notebook Arif: iloc[:, 1:5] dan target Species / Class
x = np.array(df_active[feature_cols].values, dtype=np.float64)
y = np.array(df_active[target_col].tolist())
class_labels = sorted(list(set(y)))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=3)
nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)
y_predict = nb_clf.predict(x_test)
akurasi_aktif = accuracy_score(y_test, y_predict)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Eksplorasi Data",
        "Evaluasi Model",
        "Tiga Skenario",
        "Klasifikasi Data Baru",
    ]
)

with tab1:
    st.subheader(f"Dataset Aktif: {dataset_choice}")
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Total Sampel", f"{df_active.shape[0]} baris")
    col_d2.metric("Jumlah Fitur", f"{len(feature_cols)} atribut")
    col_d3.metric("Data Latih / Uji", f"{len(x_train)} / {len(x_test)}")

    st.markdown("#### Preview Data")
    st.dataframe(df_active.head(10), use_container_width=True)

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"#### Sebaran Kelas (`{target_col}`)")
        vc = df_active[target_col].value_counts().reset_index()
        vc.columns = [target_col, "Jumlah"]
        st.dataframe(vc, use_container_width=True)
    with col_info2:
        st.markdown("#### Statistik Deskriptif")
        st.dataframe(df_active[feature_cols].describe(), use_container_width=True)

with tab2:
    st.subheader("Hasil Evaluasi Model GaussianNB")
    col_m1, col_m2 = st.columns(2)
    col_m1.markdown(
        f'<div class="card"><p style="margin:0;color:#64748b;">Akurasi Model Aktif</p>'
        f'<div class="metric-value">{akurasi_aktif * 100:.2f}%</div>'
        f'<p style="margin:0;font-size:12px;color:#059669;">{nama_skenario} | '
        f'Split {int((1 - test_size) * 100)}:{int(test_size * 100)}</p></div>',
        unsafe_allow_html=True,
    )
    col_m2.markdown(
        '<div class="card"><p style="margin:0;color:#64748b;">Jenis Naive Bayes</p>'
        '<div class="metric-value" style="font-size:22px;">GaussianNB</div>'
        '<p style="margin:0;font-size:12px;color:#64748b;">random_state = 3</p></div>',
        unsafe_allow_html=True,
    )

    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        st.markdown("#### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_predict, labels=class_labels)
        im = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xticks(range(len(class_labels)))
        ax_cm.set_yticks(range(len(class_labels)))
        ax_cm.set_xticklabels(class_labels, rotation=30, ha="right")
        ax_cm.set_yticklabels(class_labels)
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Aktual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig_cm.colorbar(im, ax=ax_cm)
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
    with col_eval2:
        st.markdown("#### Classification Report")
        rep = classification_report(y_test, y_predict, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(rep).transpose(), use_container_width=True)
        st.caption(f"Test Accuration : {akurasi_aktif}")

with tab3:
    st.subheader("Eksperimen Tiga Skenario Train-Test Split")
    st.caption("GaussianNB, random_state=3, sama seperti notebook Arif.")

    skenario_split = {
        "Skenario 1 (Train 80% : Test 20%)": 0.2,
        "Skenario 2 (Train 70% : Test 30%)": 0.3,
        "Skenario 3 (Train 60% : Test 40%)": 0.4,
    }
    rows = []
    summary_text = ""
    for nama, ukuran_test in skenario_split.items():
        xtr, xts, ytr, yts = train_test_split(x, y, test_size=ukuran_test, random_state=3)
        pred_s = GaussianNB().fit(xtr, ytr).predict(xts)
        acc = accuracy_score(yts, pred_s)
        rows.append(
            {
                "Skenario": nama,
                "Jenis Naive Bayes": "GaussianNB",
                "Akurasi": f"{acc * 100:.2f}%",
                "Nilai Akurasi": acc,
            }
        )
        summary_text += f"{nama}\n -> Akurasi : {acc:.4f} ({acc * 100:.2f}%)\n\n"

    df_hasil = pd.DataFrame(rows)
    terbaik = df_hasil.loc[df_hasil["Nilai Akurasi"].idxmax()]
    st.code(summary_text, language="text")
    st.dataframe(df_hasil.drop(columns=["Nilai Akurasi"]), use_container_width=True)
    st.success(
        f"Konfigurasi akurasi tertinggi pada {dataset_choice}: {terbaik['Skenario']} "
        f"dengan {terbaik['Jenis Naive Bayes']} ({terbaik['Akurasi']})."
    )

with tab4:
    st.subheader("Klasifikasi Data Baru")
    st.write("Masukkan nilai fitur untuk diuji pada ketiga skenario sekaligus.")
    cols_in = st.columns(len(feature_cols))
    input_values = []
    for idx_f, f_name in enumerate(feature_cols):
        with cols_in[idx_f]:
            if dataset_choice == "Balance Scale Dataset":
                val = st.number_input(
                    f_name, min_value=1, max_value=5, value=int(default_vals[idx_f]), step=1
                )
            else:
                val = st.number_input(
                    f_name, min_value=0.1, max_value=10.0, value=float(default_vals[idx_f]), step=0.1
                )
            input_values.append(val)

    if st.button("Prediksi Data Baru", use_container_width=True):
        input_data = np.array([input_values], dtype=np.float64)
        hasil = []
        for nama, ukuran_test in {
            "Skenario 1": 0.2,
            "Skenario 2": 0.3,
            "Skenario 3": 0.4,
        }.items():
            xtr, _, ytr, _ = train_test_split(x, y, test_size=ukuran_test, random_state=3)
            pred_new = GaussianNB().fit(xtr, ytr).predict(input_data)[0]
            hasil.append({"Skenario": nama, "Prediksi": pred_new})
            st.write(f"Predict Result {nama} : {pred_new}")
        st.dataframe(pd.DataFrame(hasil), use_container_width=True)
