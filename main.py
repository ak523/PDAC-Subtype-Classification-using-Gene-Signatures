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
session_vars = [
    "gene_expression_data", "metadata", "gene_list_data",
    "X_train", "X_test", "y_train", "y_test",
    "important_features", "X_train_filtered", "X_test_filtered",
    "rf_model_filtered", "rf_model_filtered_tuned", "rf_filtered_best_params",
    "svm_model_filtered", "svm_model_filtered_tuned", "svm_filtered_best_params"
]

for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = None

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_data():
    """
    Preprocess the data:
    1) Filter metadata to include only 'Primary' samples
    2) Optionally filter gene expression data by the signature file (if provided)
    3) Split into train/test sets
    """
    try:
        gene_expression_data = st.session_state["gene_expression_data"]
        metadata = st.session_state["metadata"]
        gene_list_data = st.session_state["gene_list_data"]

        # Filter metadata for primary samples
        filtered_metadata = metadata[metadata['tissue type:ch2'] == 'Primary']
        sample_ids = filtered_metadata['geo_accession'].values

        # Align gene expression data with metadata
        gene_expression_filtered = gene_expression_data[['ID'] + list(sample_ids)]
        gene_expression_filtered.set_index('ID', inplace=True)

        # Filter by gene list (signatures) if provided
        if gene_list_data is not None:
            # Expecting a column named "Gene" in the signature file
            genes_of_interest = gene_list_data['Gene'].unique().tolist()
            gene_expression_filtered = gene_expression_filtered.loc[
                gene_expression_filtered.index.intersection(genes_of_interest)
            ]

        # Extract labels (tumor_subtype_0na_1classical_2basal:ch2)
        target_labels = filtered_metadata['tumor_subtype_0na_1classical_2basal:ch2'].astype(int).values

        # Construct X, y
        X = gene_expression_filtered.T
        y = target_labels

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Update session state
        st.session_state.update({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        })

        st.success("Data preprocessing completed successfully!")
    except Exception as e:
        st.error(f"Data preprocessing failed: {e}")
        logging.error(f"Data preprocessing error: {e}")

def identify_important_features():
    """
    Identify important features using Random Forest feature importances.
    This step is also our 'feature selection' step.
    """
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]

        if X_train is None or y_train is None:
            st.warning("Please preprocess the data first.")
            return

        # Train a Random Forest to get feature importances
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)

        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Gene": X_train.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Let the user select how many top features to keep
        top_n = st.sidebar.slider("Select Top N Important Features", min_value=5, max_value=50, value=20)
        top_features = feature_importance_df.head(top_n)

        # Filter the original X_train/X_test to keep only the top features
        st.session_state["X_train_filtered"] = X_train[top_features["Gene"]]
        st.session_state["X_test_filtered"] = st.session_state["X_test"][top_features["Gene"]]
        st.session_state["important_features"] = top_features

        # Display the results
        st.subheader(f"Top {top_n} Important Genes")
        st.dataframe(top_features)

        st.bar_chart(top_features.set_index("Gene")["Importance"])

        # Option to download the important genes
        csv_data = top_features.to_csv(index=False)
        st.download_button(
            "Download Important Genes",
            data=csv_data,
            file_name="important_genes.csv",
            mime="text/csv"
        )

        st.success(f"Identified top {top_n} important genes!")
    except Exception as e:
        st.error(f"Failed to identify important features: {e}")
        logging.error(f"Feature selection error: {e}")

# ---- Random Forest Training / Tuning ----
def train_rf_filtered():
    """
    Train a Random Forest model using filtered (important) features.
    """
    try:
        X_train = st.session_state["X_train_filtered"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please identify important features first.")
            return

        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)
        st.session_state["rf_model_filtered"] = rf_model
        st.success("Random Forest (Filtered Features) training complete!")
    except Exception as e:
        st.error(f"Training failed: {e}")
        logging.error(f"RF training error: {e}")

def fine_tune_rf_filtered():
    """
    Fine-tune the Random Forest model using GridSearchCV on filtered features.
    """
    try:
        X_train = st.session_state["X_train_filtered"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please identify important features first.")
            return

        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
        rf_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)

        st.session_state["rf_model_filtered_tuned"] = grid_search.best_estimator_
        st.session_state["rf_filtered_best_params"] = grid_search.best_params_

        st.success(f"RF Fine-tuning complete! Best parameters: {grid_search.best_params_}")
    except Exception as e:
        st.error(f"RF Fine-tuning failed: {e}")
        logging.error(f"RF fine-tuning error: {e}")

# ---- SVM Training / Tuning ----
def train_svm_filtered():
    """
    Train an SVM model using filtered (important) features.
    """
    try:
        X_train = st.session_state["X_train_filtered"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please identify important features first.")
            return

        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(X_train, y_train)
        st.session_state["svm_model_filtered"] = svm_model
        st.success("SVM (Filtered Features) training complete!")
    except Exception as e:
        st.error(f"SVM training failed: {e}")
        logging.error(f"SVM training error: {e}")

def fine_tune_svm_filtered():
    """
    Fine-tune the SVM model using GridSearchCV on filtered features.
    """
    try:
        X_train = st.session_state["X_train_filtered"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please identify important features first.")
            return

        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": [0.01, 0.1, 1]
        }
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)

        st.session_state["svm_model_filtered_tuned"] = grid_search.best_estimator_
        st.session_state["svm_filtered_best_params"] = grid_search.best_params_

        st.success(f"SVM Fine-tuning complete! Best parameters: {grid_search.best_params_}")
    except Exception as e:
        st.error(f"SVM Fine-tuning failed: {e}")
        logging.error(f"SVM fine-tuning error: {e}")

# ---- Evaluation / Comparison ----
def evaluate_model(model_key, title):
    """
    Evaluate the specified model on the test set and display metrics.
    """
    model = st.session_state.get(model_key)
    X_test = st.session_state["X_test_filtered"]
    y_test = st.session_state["y_test"]

    if model is None:
        st.warning(f"{title} is not trained or fine-tuned yet.")
        return None
    if X_test is None or y_test is None:
        st.warning("Test data is missing. Please preprocess the data first.")
        return None

    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader(f"{title} - Evaluation Metrics")
        st.metric("Accuracy", f"{acc:.4f}")
        st.write("Classification Report:")
        st.table(pd.DataFrame(report).T)

        st.write("Confusion Matrix:")
        display_confusion_matrix(cm, title)

        return {"accuracy": acc, "report": report, "confusion_matrix": cm}
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        logging.error(f"Evaluation error for {title}: {e}")
        return None

def display_confusion_matrix(cm, title):
    """
    Display a confusion matrix as a heatmap.
    """
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

def analyze_stability(model_key, model_name):
    """
    Analyze the stability of the model using cross-validation on the training set.
    Shows mean accuracy and standard deviation.
    """
    model = st.session_state.get(model_key)
    X_train = st.session_state.get("X_train_filtered")
    y_train = st.session_state.get("y_train")

    if model is None:
        st.warning(f"{model_name} is not trained or fine-tuned yet.")
        return
    if X_train is None or y_train is None:
        st.warning("Training data is missing. Please preprocess and select important features first.")
        return

    try:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        st.subheader(f"{model_name} - Stability Analysis")
        st.write(f"**Mean Accuracy:** {mean_score:.4f}")
        st.write(f"**Standard Deviation:** {std_score:.4f}")
        conf_int_low = mean_score - 1.96 * std_score
        conf_int_high = mean_score + 1.96 * std_score
        st.write(f"**95% Confidence Interval:** [{conf_int_low:.4f}, {conf_int_high:.4f}]")
    except Exception as e:
        st.error(f"Stability analysis failed: {e}")
        logging.error(f"Stability analysis error for {model_name}: {e}")

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title("PDAC Subtype Classification (GSE71729) using Random Forest & SVM")

# Sidebar for Data Upload
st.sidebar.header("Data Upload")
gene_expr_file = st.sidebar.file_uploader("Gene Expression CSV", type=["csv"])
metadata_file = st.sidebar.file_uploader("Metadata CSV", type=["csv"])
signature_file = st.sidebar.file_uploader("Gene Signatures CSV (Optional)", type=["csv"])

# Load uploaded data into session state
if gene_expr_file is not None:
    st.session_state["gene_expression_data"] = pd.read_csv(gene_expr_file)
if metadata_file is not None:
    st.session_state["metadata"] = pd.read_csv(metadata_file)
if signature_file is not None:
    # Expecting a column named 'Gene' in the signature file
    st.session_state["gene_list_data"] = pd.read_csv(signature_file)

# Button to preprocess data
if st.sidebar.button("Preprocess Data"):
    preprocess_data()

# Create tabs for workflow steps
tab_identify, tab_train_rf, tab_train_svm, tab_evaluate, tab_compare = st.tabs([
    "1. Identify Important Genes",
    "2. Train Random Forest",
    "3. Train SVM",
    "4. Evaluate Models",
    "5. Compare & Stability"
])

# 1. Identify Important Genes
with tab_identify:
    st.header("Identify Important Genes")
    st.write("This step uses a Random Forest to find the top N most important genes.")
    if st.button("Identify Important Genes"):
        identify_important_features()

# 2. Train Random Forest
with tab_train_rf:
    st.header("Train & Fine-Tune Random Forest on Filtered Features")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Train RF (Filtered Features)"):
            train_rf_filtered()
    with col2:
        if st.button("Fine-Tune RF"):
            fine_tune_rf_filtered()

# 3. Train SVM
with tab_train_svm:
    st.header("Train & Fine-Tune SVM on Filtered Features")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Train SVM (Filtered Features)"):
            train_svm_filtered()
    with col2:
        if st.button("Fine-Tune SVM"):
            fine_tune_svm_filtered()

# 4. Evaluate Models
with tab_evaluate:
    st.header("Evaluate Models on Test Set (Filtered Features)")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Evaluate RF (Filtered Features)"):
            rf_results = evaluate_model("rf_model_filtered_tuned", "Random Forest (Filtered + Tuned)")
    with col2:
        if st.button("Evaluate SVM (Filtered Features)"):
            svm_results = evaluate_model("svm_model_filtered_tuned", "SVM (Filtered + Tuned)")

# 5. Compare & Stability
with tab_compare:
    st.header("Compare Performance & Analyze Stability")

    # Compare Performance (e.g., side-by-side metrics)
    st.subheader("Performance Comparison")
    rf_perf = None
    svm_perf = None

    if st.session_state["rf_model_filtered_tuned"] is not None:
        rf_perf = evaluate_model("rf_model_filtered_tuned", "Random Forest (Filtered + Tuned)")
    else:
        st.write("No tuned Random Forest model available to compare.")

    if st.session_state["svm_model_filtered_tuned"] is not None:
        svm_perf = evaluate_model("svm_model_filtered_tuned", "SVM (Filtered + Tuned)")
    else:
        st.write("No tuned SVM model available to compare.")

    if rf_perf and svm_perf:
        # Create a small comparison dataframe for accuracy
        comparison_df = pd.DataFrame({
            "Model": ["Random Forest", "SVM"],
            "Accuracy": [rf_perf["accuracy"], svm_perf["accuracy"]]
        })
        st.write("**Accuracy Comparison**")
        st.table(comparison_df)

    # Stability Analysis
    st.subheader("Stability Analysis (Cross-Validation)")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Analyze RF Stability"):
            analyze_stability("rf_model_filtered_tuned", "Random Forest (Filtered + Tuned)")
    with col2:
        if st.button("Analyze SVM Stability"):
            analyze_stability("svm_model_filtered_tuned", "SVM (Filtered + Tuned)")
