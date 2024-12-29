import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

# -- Configure the Streamlit page
st.set_page_config(
    page_title="PDAC Subtype Classification (GSE71729)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Session State for Data
# -----------------------------
# We store data in session state so that app interactions don't reset everything.
for var in [
    "gene_expression_data", "metadata", "gene_list_data",
    "X_train", "X_test", "y_train", "y_test",
    "rf_model", "rf_model_top", "svm_model", "svm_model_top",
    "top_genes", "X_train_top", "X_test_top"
]:
    if var not in st.session_state:
        st.session_state[var] = None

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_data():
    """
    1. Filter metadata for primary samples.
    2. Align gene expression with sample IDs.
    3. Filter gene expression by genes of interest (if provided).
    4. Extract labels.
    5. Split train-test.
    """
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

        # Filter by gene_list_data (if provided and not empty)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2, 
                                                            random_state=42)

        # Save in session state
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        st.success("Data preprocessing completed successfully!")
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")

def train_rf_all_features():
    """
    Train a Random Forest model using all features in X_train.
    """
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)
        st.session_state["rf_model"] = rf_model
        st.success("Random Forest (All Features) training complete!")
    except Exception as e:
        st.error(f"Random Forest training failed: {e}")

def evaluate_rf_all_features():
    """
    Evaluate the trained Random Forest (all features) on the test set and return results as a dict.
    """
    results = {}
    try:
        rf_model = st.session_state["rf_model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        y_pred = rf_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        results["accuracy"] = acc
        results["report"] = report
        results["confusion_matrix"] = cm
    except Exception as e:
        st.error(f"Failed to evaluate Random Forest (all features): {e}")
    return results

def identify_top_genes(N=20):
    """
    Identify top N important genes from the trained RF model using feature_importances_.
    """
    try:
        rf_model = st.session_state["rf_model"]
        X_train = st.session_state["X_train"]
        if not rf_model:
            st.warning("Train the RF model (all features) first!")
            return

        feature_importances = rf_model.feature_importances_
        important_genes_df = pd.DataFrame({
            'Gene': X_train.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        top_genes = important_genes_df.head(N)['Gene'].values
        st.session_state["top_genes"] = top_genes

        # Display bar chart for top features
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=important_genes_df.head(N),
            x='Importance', y='Gene', ax=ax,
            palette='Blues_r'
        )
        ax.set_title(f"Top {N} Genes by Importance")
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Gene")
        st.pyplot(fig)

        return top_genes
    except Exception as e:
        st.error(f"Top genes extraction failed: {e}")

def train_rf_top_genes():
    """
    Train a Random Forest model with only the top genes.
    """
    try:
        top_genes = st.session_state["top_genes"]
        if top_genes is None:
            st.warning("Please identify top genes first!")
            return

        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        X_train_top = X_train[top_genes]
        X_test_top = X_test[top_genes]

        st.session_state["X_train_top"] = X_train_top
        st.session_state["X_test_top"] = X_test_top
        y_train = st.session_state["y_train"]

        rf_model_top = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model_top.fit(X_train_top, y_train)
        st.session_state["rf_model_top"] = rf_model_top
        st.success("Random Forest (Top Genes) training complete!")
    except Exception as e:
        st.error(f"Random Forest (top genes) training failed: {e}")

def evaluate_rf_top_genes():
    """
    Evaluate the trained Random Forest (top genes) on the test set.
    """
    results = {}
    try:
        rf_model_top = st.session_state["rf_model_top"]
        X_test_top = st.session_state["X_test_top"]
        y_test = st.session_state["y_test"]

        y_pred_top = rf_model_top.predict(X_test_top)
        acc_top = accuracy_score(y_test, y_pred_top)
        report_top = classification_report(y_test, y_pred_top, output_dict=True)
        cm_top = confusion_matrix(y_test, y_pred_top)

        results["accuracy"] = acc_top
        results["report"] = report_top
        results["confusion_matrix"] = cm_top
    except Exception as e:
        st.error(f"Failed to evaluate Random Forest (top genes): {e}")
    return results

def train_svm_all_features():
    """
    Train an SVM model using all features.
    """
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(X_train, y_train)
        st.session_state["svm_model"] = svm_model
        st.success("SVM (All Features) training complete!")
    except Exception as e:
        st.error(f"SVM training failed: {e}")

def evaluate_svm_all_features():
    """
    Evaluate the trained SVM (all features) on the test set.
    """
    results = {}
    try:
        svm_model = st.session_state["svm_model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        y_pred_svm = svm_model.predict(X_test)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
        cm_svm = confusion_matrix(y_test, y_pred_svm)

        results["accuracy"] = acc_svm
        results["report"] = report_svm
        results["confusion_matrix"] = cm_svm
    except Exception as e:
        st.error(f"Failed to evaluate SVM (all features): {e}")
    return results

def train_svm_top_genes():
    """
    Train an SVM model with only the top N genes.
    """
    try:
        top_genes = st.session_state["top_genes"]
        if top_genes is None:
            st.warning("Please identify top genes first!")
            return

        X_train_top = st.session_state["X_train_top"]
        y_train = st.session_state["y_train"]

        svm_model_top = SVC(kernel='linear', probability=True, random_state=42)
        svm_model_top.fit(X_train_top, y_train)
        st.session_state["svm_model_top"] = svm_model_top
        st.success("SVM (Top Genes) training complete!")
    except Exception as e:
        st.error(f"SVM (top genes) training failed: {e}")

def evaluate_svm_top_genes():
    """
    Evaluate the trained SVM (top genes) on the test set.
    """
    results = {}
    try:
        svm_model_top = st.session_state["svm_model_top"]
        X_test_top = st.session_state["X_test_top"]
        y_test = st.session_state["y_test"]

        y_pred_svm_top = svm_model_top.predict(X_test_top)
        acc_svm_top = accuracy_score(y_test, y_pred_svm_top)
        report_svm_top = classification_report(y_test, y_pred_svm_top, output_dict=True)
        cm_svm_top = confusion_matrix(y_test, y_pred_svm_top)

        results["accuracy"] = acc_svm_top
        results["report"] = report_svm_top
        results["confusion_matrix"] = cm_svm_top
    except Exception as e:
        st.error(f"Failed to evaluate SVM (top genes): {e}")
    return results

def display_confusion_matrix(cm, title):
    """
    Display a single confusion matrix as a heatmap with the given title.
    """
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

def show_metrics(results, model_name):
    """
    Display accuracy, classification report, and confusion matrix for a given model result dict.
    """
    if not results:
        st.write(f"No results available for {model_name}.")
        return

    accuracy = results["accuracy"]
    report_dict = results["report"]
    cm = results["confusion_matrix"]

    st.metric(label=f"{model_name} Accuracy", value=f"{accuracy:.4f}")

    # Show classification report in a more structured way
    st.write(f"**Classification Report for {model_name}**")
    st.table(pd.DataFrame(report_dict).T)

    # Show confusion matrix
    display_confusion_matrix(cm, title=f"{model_name} - Confusion Matrix")


# -------------------------------------------------
# Streamlit Layout
# -------------------------------------------------
st.title("PDAC Subtype Classification (GSE71729)")

st.markdown("""
This application allows you to:
1. **Upload** three CSV files (Gene Expression, Metadata, and an optional Gene List).
2. **Preprocess** the data.
3. **Train** Random Forest and SVM models (all features vs. top features).
4. **View** classification reports and confusion matrices.
5. **Compare** the performance of the models.

---
""")

# ---------------------------------------
# Sidebar for File Upload & Preprocessing
# ---------------------------------------
with st.sidebar:
    st.header("Data Upload")
    gene_expr_file = st.file_uploader("Gene Expression CSV", type=["csv"])
    metadata_file = st.file_uploader("Metadata CSV", type=["csv"])
    gene_list_file = st.file_uploader("Gene List CSV (Optional)", type=["csv"])

    if gene_expr_file is not None:
        st.session_state["gene_expression_data"] = pd.read_csv(gene_expr_file)
        st.success("Gene Expression data loaded successfully!")

    if metadata_file is not None:
        st.session_state["metadata"] = pd.read_csv(metadata_file)
        st.success("Metadata loaded successfully!")

    if gene_list_file is not None:
        st.session_state["gene_list_data"] = pd.read_csv(gene_list_file)
        st.success("Gene list data loaded successfully!")

    if st.button("Preprocess Data"):
        if (st.session_state["gene_expression_data"] is None) or (st.session_state["metadata"] is None):
            st.warning("Please provide both Gene Expression and Metadata files.")
        else:
            preprocess_data()

# ---------------------------------------
# Model Training and Evaluation Tabs
# ---------------------------------------
tab_rf, tab_svm, tab_compare = st.tabs(["Random Forest", "SVM", "Compare Models"])

# ---------------------------------------
# Tab: Random Forest
# ---------------------------------------
with tab_rf:
    st.header("Random Forest Modeling")
    col_rf1, col_rf2 = st.columns(2)

    with col_rf1:
        st.subheader("1. Train RF (All Features)")
        if st.button("Train RF (All Features)", key="train_rf_all"):
            if (st.session_state["X_train"] is None) or (st.session_state["y_train"] is None):
                st.warning("Please preprocess data first.")
            else:
                train_rf_all_features()

        st.subheader("2. Evaluate RF (All Features)")
        if st.button("Evaluate RF (All Features)", key="eval_rf_all"):
            rf_all_results = evaluate_rf_all_features()
            show_metrics(rf_all_results, "Random Forest (All Features)")

    with col_rf2:
        st.subheader("3. Identify Top Genes (RF)")
        if st.button("Identify Top Genes", key="identify_genes"):
            if st.session_state["rf_model"] is None:
                st.warning("Train RF (All Features) first!")
            else:
                identify_top_genes(N=20)  # default top 20

        st.subheader("4. Train RF (Top Genes)")
        if st.button("Train RF (Top Genes)", key="train_rf_top"):
            train_rf_top_genes()

        st.subheader("5. Evaluate RF (Top Genes)")
        if st.button("Evaluate RF (Top Genes)", key="eval_rf_top"):
            rf_top_results = evaluate_rf_top_genes()
            show_metrics(rf_top_results, "Random Forest (Top Genes)")


# ---------------------------------------
# Tab: SVM
# ---------------------------------------
with tab_svm:
    st.header("SVM Modeling")
    col_svm1, col_svm2 = st.columns(2)

    with col_svm1:
        st.subheader("1. Train SVM (All Features)")
        if st.button("Train SVM (All Features)", key="train_svm_all"):
            if (st.session_state["X_train"] is None) or (st.session_state["y_train"] is None):
                st.warning("Please preprocess data first.")
            else:
                train_svm_all_features()

        st.subheader("2. Evaluate SVM (All Features)")
        if st.button("Evaluate SVM (All Features)", key="eval_svm_all"):
            svm_all_results = evaluate_svm_all_features()
            show_metrics(svm_all_results, "SVM (All Features)")

    with col_svm2:
        st.subheader("3. Train SVM (Top Genes)")
        if st.button("Train SVM (Top Genes)", key="train_svm_top"):
            train_svm_top_genes()

        st.subheader("4. Evaluate SVM (Top Genes)")
        if st.button("Evaluate SVM (Top Genes)", key="eval_svm_top"):
            svm_top_results = evaluate_svm_top_genes()
            show_metrics(svm_top_results, "SVM (Top Genes)")


# ---------------------------------------
# Tab: Compare Models
# ---------------------------------------
with tab_compare:
    st.header("Model Comparison")

    st.markdown("""
    Here, you can compare the confusion matrices and metrics for all four models side by side.
    """)

    # We'll retrieve the results for each model (if they exist) and display them.
    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.subheader("Random Forest Comparison")
        rf_all_results = evaluate_rf_all_features()
        rf_top_results = evaluate_rf_top_genes()

        if rf_all_results.get("confusion_matrix") is not None:
            display_confusion_matrix(
                rf_all_results["confusion_matrix"], 
                "RF (All Features) - Confusion Matrix"
            )
        if rf_top_results.get("confusion_matrix") is not None:
            display_confusion_matrix(
                rf_top_results["confusion_matrix"],
                "RF (Top Genes) - Confusion Matrix"
            )

    with col_cm2:
        st.subheader("SVM Comparison")
        svm_all_results = evaluate_svm_all_features()
        svm_top_results = evaluate_svm_top_genes()

        if svm_all_results.get("confusion_matrix") is not None:
            display_confusion_matrix(
                svm_all_results["confusion_matrix"], 
                "SVM (All Features) - Confusion Matrix"
            )
        if svm_top_results.get("confusion_matrix") is not None:
            display_confusion_matrix(
                svm_top_results["confusion_matrix"],
                "SVM (Top Genes) - Confusion Matrix"
            )

    st.markdown("""
    **Hint:** The metrics and classification reports for each model are shown in the RF/SVM tabs. 
    Use this comparison to get a quick visual sense of where each model might be struggling.
    """)

