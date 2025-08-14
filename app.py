# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

import plotly.express as px

st.set_page_config(page_title="Clustering Playground", layout="wide")
st.title("🧩 Clustering Playground (Robust Loader)")

# ---------- Helpers ----------
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ARABIC_DIGITS  = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_numeric_text(s: pd.Series) -> pd.Series:
    """
    Clean numeric-looking strings and convert to float:
    - remove LTR/RTL marks and spaces
    - convert Persian/Arabic digits to ASCII
    - drop '.' thousands separators
    - convert decimal ',' to '.'
    - finally coerce to numeric
    """
    s = s.astype(str)
    s = s.str.replace("\u200f", "", regex=False).str.replace("\u200e", "", regex=False)
    s = s.str.translate(PERSIAN_DIGITS).str.translate(ARABIC_DIGITS)
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace(".", "", regex=False)   # remove thousands dot
    s = s.str.replace(",", ".", regex=False)  # decimal comma -> dot
    return pd.to_numeric(s, errors="coerce")

def coerce_numeric_columns(df: pd.DataFrame, min_ratio: float = 0.6):
    """
    Convert object/string columns to numeric if at least `min_ratio` of values
    can be parsed as numbers. Returns (converted_df, report_df).
    """
    report = {}
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            report[col] = 1.0
            continue
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            coerced = normalize_numeric_text(out[col])
            ratio = coerced.notna().mean()
            report[col] = float(ratio)
            if ratio >= min_ratio:
                out[col] = coerced
        else:
            report[col] = 0.0
    report_df = pd.DataFrame({"coercible_to_numeric_ratio": pd.Series(report)}).sort_values(
        "coercible_to_numeric_ratio", ascending=False
    )
    return out, report_df

@st.cache_data(show_spinner=False)
def robust_read_csv(file_bytes: bytes | None) -> pd.DataFrame:
    """
    Load CSV with automatic delimiter detection and multiple encodings.
    If no file is uploaded, read dataset.csv next to this script.
    """
    encodings_to_try = ["utf-8-sig", "utf-8", "cp1256", "latin1"]
    if file_bytes is None:
        default_path = Path(__file__).parent / "dataset.csv"
        return pd.read_csv(default_path, sep=None, engine="python")
    else:
        last_err = None
        for enc in encodings_to_try:
            try:
                buf = io.BytesIO(file_bytes)
                return pd.read_csv(buf, sep=None, engine="python", encoding=enc)
            except Exception as e:
                last_err = e
                continue
        raise last_err if last_err else RuntimeError("Failed to read CSV.")

# ---------- UI: Upload & Load ----------
uploaded = st.file_uploader(
    "Upload a CSV file (optional). If nothing is uploaded, dataset.csv will be used.",
    type=["csv"]
)

try:
    if uploaded is not None:
        df = robust_read_csv(uploaded.getvalue())
    else:
        df = robust_read_csv(None)
except Exception as e:
    st.error(f"CSV read error: {e}")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head())

# ---------- Numeric coercion controls ----------
with st.expander("⚙️ Numeric parsing settings"):
    min_ratio = st.slider(
        "Minimum numeric parse ratio to convert text columns → numeric",
        min_value=0.5, max_value=0.95, value=0.6, step=0.05
    )
    aggressive = st.checkbox(
        "Aggressive conversion (try harder to coerce text columns to numbers)",
        value=True
    )

# Apply conversion if needed
df_processed, coercion_report = coerce_numeric_columns(
    df, min_ratio=min_ratio if aggressive else 0.8
)

st.caption("Columns’ ability to be coerced to numeric (closer to 1 is better):")
st.dataframe(coercion_report)

# ---------- Select numeric columns ----------
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found after processing. Check delimiter/format of your CSV.")
    st.stop()

cols = st.multiselect(
    "Select feature columns for clustering:",
    numeric_cols,
    default=numeric_cols[: min(5, len(numeric_cols))]
)
if not cols:
    st.warning("Select at least one column.")
    st.stop()

# ---------- Prepare X ----------
X_raw = df_processed[cols].dropna()
if X_raw.empty:
    st.error("No data left after dropping NaNs.")
    st.stop()

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ---------- Sidebar: Algorithm ----------
with st.sidebar:
    st.header("Algorithm settings")
    algo = st.selectbox("Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])

    if algo == "KMeans":
        k = st.slider("Number of clusters (k)", 2, 15, 4)
        random_state = st.number_input("random_state", value=42, step=1)
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    elif algo == "DBSCAN":
        eps = st.number_input("eps", value=0.5, min_value=0.05, step=0.05, format="%.2f")
        min_samples = st.slider("min_samples", 2, 50, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        k = st.slider("Number of clusters", 2, 15, 4)
        linkage = st.selectbox("linkage", ["ward", "complete", "average", "single"])
        # Note: 'ward' works best with Euclidean distance and continuous variables
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)

# ---------- Run ----------
fit_btn = st.button("Run clustering")
if fit_btn:
    labels = model.fit_predict(X)

    # Silhouette score (ignore DBSCAN noise labeled as -1)
    sil_text = "Not computed"
    try:
        mask = labels != -1
        if mask.sum() >= 2 and len(set(labels[mask])) > 1:
            sil = silhouette_score(X[mask], labels[mask])
            sil_text = f"{sil:.3f}"
    except Exception:
        pass

    # Attach cluster labels
    df_out = df_processed.copy()
    df_out.loc[X_raw.index, "cluster"] = labels

    st.subheader("Results")
    left, right = st.columns(2)
    with left:
        st.write("Points per cluster:")
        st.write(pd.Series(labels).value_counts().sort_index())
    with right:
        st.metric("Silhouette score", sil_text)

    # 2D visualization via PCA
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    plot_df = pd.DataFrame({"pc1": X2[:, 0], "pc2": X2[:, 1], "cluster": labels.astype(str)})
    fig = px.scatter(plot_df, x="pc1", y="pc2", color="cluster", title="2D projection (PCA)")
    st.plotly_chart(fig, use_container_width=True)

    # Download clustered data
    st.subheader("Download clustered data")
    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download output CSV", data=csv, file_name="clustered_output.csv", mime="text/csv")

st.caption(
    "Note: If no file is uploaded, dataset.csv next to this script will be used. "
    "This version automatically detects delimiter/encoding and normalizes numeric formats."
)