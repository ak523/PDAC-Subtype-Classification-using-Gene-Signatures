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
    "X_train_filtered", "X_test_filtered", "important_features",
    "rf_model_full", "rf_model_filtered",
    "rf_model_full_tuned", "rf_model_filtered_tuned",
    "svm_model_full", "svm_model_filtered",
    "svm_model_full_tuned", "svm_model_filtered_tuned",
    "rf_full_best_params", "rf_filtered_best_params",
    "svm_full_best_params", "svm_filtered_best_params"
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

        if gene_expression_data is None or metadata is None:
            st.warning("Please upload Gene Expression and Metadata files.")
            return

        # Filter metadata for primary samples
        filtered_metadata = metadata[metadata['tissue type:ch2'] == 'Primary']
        sample_ids = filtered_metadata['geo_accession'].values

        # Align gene expression data with metadata
        gene_expression_filtered = gene_expression_data[['ID'] + list(sample_ids)]
        gene_expression_filtered.set_index('ID', inplace=True)

        # Filter by gene list (signatures) if provided
        if gene_list_data is not None:
            genes_of_interest = gene_list_data['Gene'].unique().tolist()
            gene_expression_filtered = gene_expression_filtered.loc[
                gene_expression_filtered.index.intersection(genes_of_interest)
            ]

        # Extract labels (tumor_subtype_0na_1classical_2basal:ch2)
        target_labels = filtered_metadata['tumor_subtype_0na_1classical_2basal:ch2'].astype(int).values

        # Construct X, y
        X = gene_expression_filtered.T
        y = target_labels

        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
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
    Identify important features using Random Forest feature importances (on the *full* feature set).
    """
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]

        if X_train is None or y_train is None:
            st.warning("Please preprocess the data first.")
            return

        # Train a Random Forest to get feature importances
        rf_model_full = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model_full.fit(X_train, y_train)
        st.session_state["rf_model_full"] = rf_model_full

        importances = rf_model_full.feature_importances_
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

# ---- Training / Tuning - Random Forest (Full) ----
def train_rf_full():
    """
    Train a Random Forest model on the FULL features (no filtering).
    """
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please preprocess the data first.")
            return

        rf_model_full = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
        rf_model_full.fit(X_train, y_train)
        st.session_state["rf_model_full"] = rf_model_full
        st.success("Random Forest (Full Features) training complete!")
    except Exception as e:
        st.error(f"Training failed: {e}")
        logging.error(f"RF full training error: {e}")

def fine_tune_rf_full():
    """
    Fine-tune the Random Forest model on the FULL features using GridSearchCV.
    """
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please train RF (Full) first.")
            return

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        st.session_state["rf_model_full_tuned"] = grid_search.best_estimator_
        st.session_state["rf_full_best_params"] = grid_search.best_params_

        st.success(f"RF (Full) Fine-tuning complete! Best parameters: {grid_search.best_params_}")
    except Exception as e:
        st.error(f"RF (Full) Fine-tuning failed: {e}")
        logging.error(f"RF full fine-tuning error: {e}")

# ---- Training / Tuning - Random Forest (Filtered) ----
def train_rf_filtered():
    """
    Train a Random Forest model on the SELECTED (important) features.
    """
    try:
        X_train = st.session_state["X_train_filtered"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please identify important features first.")
            return

        rf_model_filtered = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
        rf_model_filtered.fit(X_train, y_train)
        st.session_state["rf_model_filtered"] = rf_model_filtered
        st.success("Random Forest (Filtered Features) training complete!")
    except Exception as e:
        st.error(f"Training failed: {e}")
        logging.error(f"RF filtered training error: {e}")

def fine_tune_rf_filtered():
    """
    Fine-tune the Random Forest model on SELECTED features using GridSearchCV.
    """
    try:
        X_train = st.session_state["X_train_filtered"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please identify important features and train the filtered RF first.")
            return

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        st.session_state["rf_model_filtered_tuned"] = grid_search.best_estimator_
        st.session_state["rf_filtered_best_params"] = grid_search.best_params_

        st.success(f"RF (Filtered) Fine-tuning complete! Best parameters: {grid_search.best_params_}")
    except Exception as e:
        st.error(f"RF (Filtered) Fine-tuning failed: {e}")
        logging.error(f"RF filtered fine-tuning error: {e}")

# ---- Training / Tuning - SVM (Full) ----
def train_svm_full():
    """
    Train an SVM model using the FULL feature set.
    """
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please preprocess the data first.")
            return

        svm_model_full = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
        svm_model_full.fit(X_train, y_train)
        st.session_state["svm_model_full"] = svm_model_full
        st.success("SVM (Full Features) training complete!")
    except Exception as e:
        st.error(f"SVM (Full) training failed: {e}")
        logging.error(f"SVM (Full) training error: {e}")

def fine_tune_svm_full():
    """
    Fine-tune the SVM model on the FULL features using GridSearchCV.
    """
    try:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please train SVM (Full) first.")
            return

        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "class_weight": ["balanced"],
            "gamma": ["scale", 0.1, 1]
        }
        svm_model = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        st.session_state["svm_model_full_tuned"] = grid_search.best_estimator_
        st.session_state["svm_full_best_params"] = grid_search.best_params_

        st.success(f"SVM (Full) Fine-tuning complete! Best parameters: {grid_search.best_params_}")
    except Exception as e:
        st.error(f"SVM (Full) Fine-tuning failed: {e}")
        logging.error(f"SVM full fine-tuning error: {e}")

# ---- Training / Tuning - SVM (Filtered) ----
def train_svm_filtered():
    """
    Train an SVM model using only the SELECTED (important) features.
    """
    try:
        X_train = st.session_state["X_train_filtered"]
        y_train = st.session_state["y_train"]
        if X_train is None or y_train is None:
            st.warning("Please identify important features first.")
            return

        svm_model_filtered = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
        svm_model_filtered.fit(X_train, y_train)
        st.session_state["svm_model_filtered"] = svm_model_filtered
        st.success("SVM (Filtered Features) training complete!")
    except Exception as e:
        st.error(f"SVM (Filtered) training failed: {e}")
        logging.error(f"SVM filtered training error: {e}")

def fine_tune_svm_filtered():
    """
    Fine-tune the SVM model using the SELECTED features with GridSearchCV.
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
            "class_weight": ["balanced"],
            "gamma": ["scale", 0.1, 1]
        }
        svm_model = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        st.session_state["svm_model_filtered_tuned"] = grid_search.best_estimator_
        st.session_state["svm_filtered_best_params"] = grid_search.best_params_

        st.success(f"SVM (Filtered) Fine-tuning complete! Best parameters: {grid_search.best_params_}")
    except Exception as e:
        st.error(f"SVM (Filtered) Fine-tuning failed: {e}")
        logging.error(f"SVM filtered fine-tuning error: {e}")

# ---- Evaluation Functions ----
def evaluate_model(model_key, title, full_or_filtered="full"):
    """
    Evaluate the specified model on the test set and display metrics.
    """
    model = st.session_state.get(model_key)
    if full_or_filtered == "filtered":
        X_test = st.session_state["X_test_filtered"]
    else:
        X_test = st.session_state["X_test"]
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

def analyze_stability(model_key, model_name, full_or_filtered="full"):
    """
    Analyze the stability of the model using cross-validation on the training set.
    """
    model = st.session_state.get(model_key)
    if full_or_filtered == "filtered":
        X_train = st.session_state.get("X_train_filtered")
    else:
        X_train = st.session_state.get("X_train")
    y_train = st.session_state.get("y_train")

    if model is None:
        st.warning(f"{model_name} is not trained or fine-tuned yet.")
        return
    if X_train is None or y_train is None:
        st.warning("Training data is missing. Please preprocess and/or select important features first.")
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
st.title("PDAC Subtype Classification (GSE71729) — Enhanced Models")

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
    # Expecting a column named 'Gene'
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
    st.header("Identify Important Genes (using Random Forest Feature Importances)")
    st.write("This step uses an RF model on the full feature set to find the top N most important genes.")
    if st.button("Identify Important Genes"):
        identify_important_features()

# 2. Train Random Forest
with tab_train_rf:
    st.header("Train & Fine-Tune Random Forest")
    st.markdown("#### Full Feature Set")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train RF (Full)"):
            train_rf_full()
    with col2:
        if st.button("Fine-Tune RF (Full)"):
            fine_tune_rf_full()

    st.markdown("---")
    st.markdown("#### Filtered (Top N) Features")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Train RF (Filtered)"):
            train_rf_filtered()
    with col4:
        if st.button("Fine-Tune RF (Filtered)"):
            fine_tune_rf_filtered()

# 3. Train SVM
with tab_train_svm:
    st.header("Train & Fine-Tune SVM")
    st.markdown("#### Full Feature Set")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train SVM (Full)"):
            train_svm_full()
    with col2:
        if st.button("Fine-Tune SVM (Full)"):
            fine_tune_svm_full()

    st.markdown("---")
    st.markdown("#### Filtered (Top N) Features")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Train SVM (Filtered)"):
            train_svm_filtered()
    with col4:
        if st.button("Fine-Tune SVM (Filtered)"):
            fine_tune_svm_filtered()

# 4. Evaluate Models
with tab_evaluate:
    st.header("Evaluate Models on Test Set")
    st.markdown("Evaluate the trained or fine-tuned models on the test set. Choose a model below:")
    col1, col2 = st.columns(2)

    # Evaluate RF (Full)
    with col1:
        if st.button("Evaluate RF (Full)"):
            evaluate_model("rf_model_full_tuned", "Random Forest (Full + Tuned)", full_or_filtered="full")

    # Evaluate RF (Filtered)
    with col2:
        if st.button("Evaluate RF (Filtered)"):
            evaluate_model("rf_model_filtered_tuned", "Random Forest (Filtered + Tuned)", full_or_filtered="filtered")

    # Evaluate SVM (Full)
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Evaluate SVM (Full)"):
            evaluate_model("svm_model_full_tuned", "SVM (Full + Tuned)", full_or_filtered="full")

    # Evaluate SVM (Filtered)
    with col4:
        if st.button("Evaluate SVM (Filtered)"):
            evaluate_model("svm_model_filtered_tuned", "SVM (Filtered + Tuned)", full_or_filtered="filtered")

# 5. Compare & Stability
with tab_compare:
    st.header("Compare Performance & Analyze Stability")
    st.write("Here you can compare final accuracies side by side and also do cross-validation stability checks.")

    # A quick function to do side-by-side model evaluations
    def quick_compare():
        results = []

        # Evaluate each model if present
        model_info = [
            ("rf_model_full_tuned", "RF (Full + Tuned)", "full"),
            ("rf_model_filtered_tuned", "RF (Filtered + Tuned)", "filtered"),
            ("svm_model_full_tuned", "SVM (Full + Tuned)", "full"),
            ("svm_model_filtered_tuned", "SVM (Filtered + Tuned)", "filtered"),
        ]
        for key, name, subset in model_info:
            eval_result = evaluate_model(key, name, full_or_filtered=subset)
            if eval_result is not None:
                results.append((name, eval_result["accuracy"]))
        # Display as a table
        if results:
            df_compare = pd.DataFrame(results, columns=["Model", "Accuracy"])
            st.write("### Accuracy Comparison")
            st.table(df_compare)
            # Bar chart
            fig, ax = plt.subplots()
            sns.barplot(x="Model", y="Accuracy", data=df_compare, ax=ax)
            ax.set_ylim(0,1)
            ax.set_title("Accuracy Comparison")
            st.pyplot(fig)

    if st.button("Compare All Tuned Models"):
        quick_compare()

    st.subheader("Stability Analysis (Cross-Validation)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze RF (Full) Stability"):
            analyze_stability("rf_model_full_tuned", "RF (Full + Tuned)", "full")
        if st.button("Analyze RF (Filtered) Stability"):
            analyze_stability("rf_model_filtered_tuned", "RF (Filtered + Tuned)", "filtered")

    with col2:
        if st.button("Analyze SVM (Full) Stability"):
            analyze_stability("svm_model_full_tuned", "SVM (Full + Tuned)", "full")
        if st.button("Analyze SVM (Filtered) Stability"):
            analyze_stability("svm_model_filtered_tuned", "SVM (Filtered + Tuned)", "filtered")
