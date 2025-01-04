import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# -- Configure the Streamlit page
st.set_page_config(
    page_title="PDAC Subtype Classification (GSE71729)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Session State for Data
# -----------------------------
for var in [
    "gene_expression_data", "metadata", "gene_list_data",
    "X_train", "X_test", "y_train", "y_test",
    "rf_model", "rf_model_tuned", "svm_model", "svm_model_tuned",
    "rf_best_params", "svm_best_params"
]:
    if var not in st.session_state:
        st.session_state[var] = None

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_data():
    """Preprocess data: filter metadata, align with gene expression, and optionally filter by gene list."""
    try:
        gene_expression_data = st.session_state["gene_expression_data"]
        metadata = st.session_state["metadata"]
        gene_list_data = st.session_state["gene_list_data"]

        # Filter metadata for "Primary" samples
        filtered_metadata = metadata[metadata['tissue type:ch2'] == 'Primary']
        sample_ids = filtered_metadata['geo_accession'].values

        # Align gene expression data with filtered metadata
        gene_expression_data_filtered = gene_expression_data[['ID'] + list(sample_ids)]
        gene_expression_data_filtered.set_index('ID', inplace=True)

        # Filter by gene list if provided
        if gene_list_data is not None and not gene_list_data.empty:
            genes_of_interest = gene_list_data['gene'].values
            gene_expression_data_filtered = gene_expression_data_filtered.loc[
                gene_expression_data_filtered.index.intersection(genes_of_interest)
            ]

        # Extract labels
        target_labels = filtered_metadata['tumor_subtype_0na_1classical_2basal:ch2'].astype(int).values

        # Construct X, y
        X = gene_expression_data_filtered.T
        y = target_labels

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save to session state
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        st.success("Data preprocessing completed successfully!")
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        logging.error(f"Preprocessing error: {e}")

def train_rf_all_features():
    """Train a Random Forest model using all features."""
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)
        st.session_state["rf_model"] = rf_model
        st.success("Random Forest (All Features) training complete!")
    except Exception as e:
        st.error(f"Random Forest training failed: {e}")
        logging.error(f"RF training error: {e}")

def fine_tune_rf():
    """Fine-tune the Random Forest model using GridSearchCV."""
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please preprocess the data before fine-tuning.")
            return

        logging.info("Starting RF fine-tuning...")
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        logging.info("RF fine-tuning complete.")

        st.session_state["rf_model_tuned"] = grid_search.best_estimator_
        st.session_state["rf_best_params"] = grid_search.best_params_
        st.success(f"RF fine-tuning complete! Best params: {grid_search.best_params_}")
    except Exception as e:
        st.error(f"RF fine-tuning failed: {e}")
        logging.error(f"RF fine-tuning error: {e}")

def train_svm_all_features():
    """Train an SVM model using all features."""
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(X_train, y_train)
        st.session_state["svm_model"] = svm_model
        st.success("SVM (All Features) training complete!")
    except Exception as e:
        st.error(f"SVM training failed: {e}")
        logging.error(f"SVM training error: {e}")

def fine_tune_svm():
    """Fine-tune the SVM model using GridSearchCV."""
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please preprocess the data before fine-tuning.")
            return

        logging.info("Starting SVM fine-tuning...")
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": [0.01, 0.1, 1]
        }
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        logging.info("SVM fine-tuning complete.")

        st.session_state["svm_model_tuned"] = grid_search.best_estimator_
        st.session_state["svm_best_params"] = grid_search.best_params_
        st.success(f"SVM fine-tuning complete! Best params: {grid_search.best_params_}")
    except Exception as e:
        st.error(f"SVM fine-tuning failed: {e}")
        logging.error(f"SVM fine-tuning error: {e}")

def evaluate_model(model_key, title):
    """Evaluate the specified model on the test set."""
    if model_key not in st.session_state or st.session_state[model_key] is None:
        st.warning(f"{title} model is not trained. Train or fine-tune it first.")
        return {}

    try:
        model = st.session_state[model_key]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        st.metric(label=f"{title} Accuracy", value=f"{acc:.4f}")
        st.write(f"**Classification Report for {title}**")
        st.table(pd.DataFrame(report).T)
        display_confusion_matrix(cm, title=f"{title} - Confusion Matrix")

        return {"accuracy": acc, "report": report, "confusion_matrix": cm}
    except Exception as e:
        st.error(f"Failed to evaluate {title}: {e}")
        logging.error(f"{title} evaluation error: {e}")
        return {}

def analyze_stability(model_key, model_name):
    """Analyze the stability of the model using cross-validation."""
    if model_key not in st.session_state or st.session_state[model_key] is None:
        st.warning(f"{model_name} model is not trained. Train or fine-tune it first.")
        return

    try:
        model = st.session_state[model_key]
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        st.write(f"**{model_name} Stability Analysis**")
        st.write(f"Mean Accuracy: {mean_score:.4f}")
        st.write(f"Standard Deviation: {std_score:.4f}")
        st.write(f"Confidence Interval (95%): [{mean_score - 1.96 * std_score:.4f}, {mean_score + 1.96 * std_score:.4f}]")
    except Exception as e:
        st.error(f"Stability analysis failed for {model_name}: {e}")
        logging.error(f"{model_name} stability analysis error: {e}")

def display_confusion_matrix(cm, title):
    """Display a confusion matrix as a heatmap."""
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

# -------------------------------------------------
# Streamlit Layout
# -------------------------------------------------
st.title("PDAC Subtype Classification with Random Forest and SVM")

st.sidebar.header("Data Upload")
gene_expr_file = st.sidebar.file_uploader("Gene Expression CSV", type=["csv"])
metadata_file = st.sidebar.file_uploader("Metadata CSV", type=["csv"])
gene_list_file = st.sidebar.file_uploader("Gene List CSV (Optional)", type=["csv"])

if gene_expr_file is not None:
    st.session_state["gene_expression_data"] = pd.read_csv(gene_expr_file)
if metadata_file is not None:
    st.session_state["metadata"] = pd.read_csv(metadata_file)
if gene_list_file is not None:
    st.session_state["gene_list_data"] = pd.read_csv(gene_list_file)

if st.sidebar.button("Preprocess Data"):
    preprocess_data()

tab_rf, tab_svm, tab_compare = st.tabs(["Random Forest", "SVM", "Compare Models"])

# Random Forest Tab
with tab_rf:
    st.header("Random Forest Modeling")
    if st.button("Train RF (All Features)"):
        train_rf_all_features()
    if st.button("Fine-Tune RF"):
        fine_tune_rf()
    if st.button("Evaluate RF (Tuned)"):
        evaluate_model("rf_model_tuned", "Random Forest (Tuned)")
    if st.button("Analyze RF Stability"):
        analyze_stability("rf_model_tuned", "Random Forest (Tuned)")

# SVM Tab
with tab_svm:
    st.header("SVM Modeling")
    if st.button("Train SVM (All Features)"):
        train_svm_all_features()
    if st.button("Fine-Tune SVM"):
        fine_tune_svm()
    if st.button("Evaluate SVM (Tuned)"):
        evaluate_model("svm_model_tuned", "SVM (Tuned)")
    if st.button("Analyze SVM Stability"):
        analyze_stability("svm_model_tuned", "SVM (Tuned)")

# Compare Models Tab
with tab_compare:
    st.header("Model Comparison")
    rf_results = evaluate_model("rf_model_tuned", "Random Forest (Tuned)")
    svm_results = evaluate_model("svm_model_tuned", "SVM (Tuned)")
    st.write("Stability Analysis:")
    analyze_stability("rf_model_tuned", "Random Forest (Tuned)")
    analyze_stability("svm_model_tuned", "SVM (Tuned)")
