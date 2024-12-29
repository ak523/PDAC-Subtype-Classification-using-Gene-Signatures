import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

# -----------------------------
# Global Variables (for demo)
# -----------------------------
gene_expression_data = None
metadata = None
gene_list_data = None

# Model objects
rf_model = None
rf_model_top = None
svm_model = None
svm_model_top = None

# Training/Test sets
X_train, X_test, y_train, y_test = None, None, None, None
X_train_top, X_test_top = None, None
top_genes = None

# -----------------------------
# Helper Functions
# -----------------------------
def load_gene_expression_csv():
    """
    Load the gene expression CSV file.
    """
    global gene_expression_data
    filepath = filedialog.askopenfilename(
        title="Select Gene Expression CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if filepath:
        try:
            gene_expression_data = pd.read_csv(filepath)
            messagebox.showinfo("Success", f"Gene Expression data loaded:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

def load_metadata_csv():
    """
    Load the metadata CSV file.
    """
    global metadata
    filepath = filedialog.askopenfilename(
        title="Select Metadata CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if filepath:
        try:
            metadata = pd.read_csv(filepath)
            messagebox.showinfo("Success", f"Metadata loaded:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

def load_gene_list_csv():
    """
    Load the gene list CSV file (containing genes of interest).
    """
    global gene_list_data
    filepath = filedialog.askopenfilename(
        title="Select Gene List CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if filepath:
        try:
            gene_list_data = pd.read_csv(filepath)
            messagebox.showinfo("Success", f"Gene list data loaded:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

def preprocess_data():
    """
    Preprocess data:
    1. Filter metadata for primary samples.
    2. Align gene expression with sample IDs.
    3. Filter gene expression by genes of interest (if provided).
    4. Extract labels.
    5. Train-Test Split
    """
    global gene_expression_data, metadata, gene_list_data
    global X_train, X_test, y_train, y_test
    
    if gene_expression_data is None or metadata is None:
        messagebox.showerror("Error", "Please load the Gene Expression and Metadata CSVs first.")
        return

    try:
        # Filter metadata to keep only primary samples
        filtered_metadata = metadata[metadata['tissue type:ch2'] == 'Primary']

        # Align gene expression data with filtered metadata
        sample_ids = filtered_metadata['geo_accession'].values
        gene_expression_data_filtered = gene_expression_data[['ID'] + list(sample_ids)]
        gene_expression_data_filtered.set_index('ID', inplace=True)

        # Filter gene expression data to include only genes in gene_list_data
        if gene_list_data is not None and not gene_list_data.empty:
            genes_of_interest = gene_list_data['gene'].values
            gene_expression_data_filtered = gene_expression_data_filtered.loc[
                gene_expression_data_filtered.index.intersection(genes_of_interest)
            ]

        # Extract target labels
        target_labels = filtered_metadata['tumor_subtype_0na_1classical_2basal:ch2'].astype(int).values

        # Construct X and y
        X = gene_expression_data_filtered.T  # Transpose so that rows = samples, cols = genes
        y = target_labels

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        messagebox.showinfo("Success", "Data preprocessed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Preprocessing failed:\n{e}")

def run_random_forest():
    """
    Train a Random Forest model using all features.
    """
    global rf_model, X_train, X_test, y_train, y_test
    if X_train is None or X_test is None:
        messagebox.showerror("Error", "Please preprocess the data first.")
        return
    try:
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Create a classification report
        report = classification_report(y_test, y_pred)
        msg = f"Random Forest (All Features) Accuracy: {acc:.4f}\n\n{report}"
        results_text.delete('1.0', tk.END)
        results_text.insert(tk.END, msg)

    except Exception as e:
        messagebox.showerror("Error", f"Random Forest training failed:\n{e}")

def identify_top_genes():
    """
    Identify top N important genes from trained RF model.
    """
    global rf_model, top_genes, X_train
    if rf_model is None:
        messagebox.showerror("Error", "Please train the Random Forest model first.")
        return
    try:
        N = 20  # You can modify as needed
        feature_importances = rf_model.feature_importances_
        important_genes_df = pd.DataFrame({
            'Gene': X_train.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        top_genes = important_genes_df.head(N)['Gene'].values

        msg = f"Top {N} Important Genes:\n"
        msg += "\n".join(top_genes)
        results_text.delete('1.0', tk.END)
        results_text.insert(tk.END, msg)

    except Exception as e:
        messagebox.showerror("Error", f"Top genes extraction failed:\n{e}")

def run_random_forest_top_genes():
    """
    Train a Random Forest model using only the top N genes.
    """
    global rf_model_top, X_train, X_test, X_train_top, X_test_top
    global top_genes, y_train, y_test

    if top_genes is None:
        messagebox.showerror("Error", "Please identify top genes first.")
        return

    try:
        X_train_top = X_train[top_genes]
        X_test_top = X_test[top_genes]

        rf_model_top = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model_top.fit(X_train_top, y_train)
        y_pred_top = rf_model_top.predict(X_test_top)
        acc_top = accuracy_score(y_test, y_pred_top)
        report_top = classification_report(y_test, y_pred_top)

        msg = f"Random Forest (Top {len(top_genes)} Genes) Accuracy: {acc_top:.4f}\n\n{report_top}"
        results_text.delete('1.0', tk.END)
        results_text.insert(tk.END, msg)
    except Exception as e:
        messagebox.showerror("Error", f"Random Forest (top genes) training failed:\n{e}")

def run_svm_all_features():
    """
    Train an SVM model using all features.
    """
    global svm_model, X_train, X_test, y_train, y_test
    if X_train is None or X_test is None:
        messagebox.showerror("Error", "Please preprocess the data first.")
        return
    try:
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        report_svm = classification_report(y_test, y_pred_svm)

        msg = f"SVM (All Features) Accuracy: {acc_svm:.4f}\n\n{report_svm}"
        results_text.delete('1.0', tk.END)
        results_text.insert(tk.END, msg)

    except Exception as e:
        messagebox.showerror("Error", f"SVM training failed:\n{e}")

def run_svm_top_genes():
    """
    Train an SVM model using only the top N genes.
    """
    global svm_model_top, X_train_top, X_test_top, y_train, y_test
    if X_train_top is None or X_test_top is None:
        messagebox.showerror("Error", "Please run 'identify_top_genes' and 'run_random_forest_top_genes' first so that top genes sets are created.")
        return
    try:
        svm_model_top = SVC(kernel='linear', probability=True, random_state=42)
        svm_model_top.fit(X_train_top, y_train)
        y_pred_svm_top = svm_model_top.predict(X_test_top)
        acc_svm_top = accuracy_score(y_test, y_pred_svm_top)
        report_svm_top = classification_report(y_test, y_pred_svm_top)

        msg = f"SVM (Top Genes) Accuracy: {acc_svm_top:.4f}\n\n{report_svm_top}"
        results_text.delete('1.0', tk.END)
        results_text.insert(tk.END, msg)

    except Exception as e:
        messagebox.showerror("Error", f"SVM (top genes) training failed:\n{e}")

def compare_confusion_matrices():
    """
    Display confusion matrices for both Random Forest and SVM (all features and top features).
    """
    global rf_model, rf_model_top, svm_model, svm_model_top
    global X_test, X_test_top, y_test

    if any(model is None for model in [rf_model, rf_model_top, svm_model, svm_model_top]):
        messagebox.showerror("Error", "Please train all four models first (RF all, RF top, SVM all, SVM top).")
        return

    try:
        y_pred_rf_all = rf_model.predict(X_test)
        y_pred_rf_top = rf_model_top.predict(X_test_top)
        y_pred_svm_all = svm_model.predict(X_test)
        y_pred_svm_top = svm_model_top.predict(X_test_top)

        cm_rf_all = confusion_matrix(y_test, y_pred_rf_all)
        cm_rf_top = confusion_matrix(y_test, y_pred_rf_top)
        cm_svm_all = confusion_matrix(y_test, y_pred_svm_all)
        cm_svm_top = confusion_matrix(y_test, y_pred_svm_top)

        # We can display them in a popup or print in the text widget
        output_str = StringIO()
        output_str.write("=== Confusion Matrices ===\n\n")
        output_str.write("Random Forest (All Features):\n")
        output_str.write(str(cm_rf_all) + "\n\n")
        output_str.write("Random Forest (Top Genes):\n")
        output_str.write(str(cm_rf_top) + "\n\n")
        output_str.write("SVM (All Features):\n")
        output_str.write(str(cm_svm_all) + "\n\n")
        output_str.write("SVM (Top Genes):\n")
        output_str.write(str(cm_svm_top) + "\n\n")

        results_text.delete('1.0', tk.END)
        results_text.insert(tk.END, output_str.getvalue())

    except Exception as e:
        messagebox.showerror("Error", f"Error comparing confusion matrices:\n{e}")

# -----------------------------
# Tkinter UI
# -----------------------------
def create_main_window():
    window = tk.Tk()
    window.title("PDAC Subtype Classification (GSE71729)")

    # Set window size
    window.geometry("950x700")

    # Notebook/Tabbed Interface (optional)
    tab_control = ttk.Notebook(window)

    # Tab 1: Data Loading
    tab_data_loading = ttk.Frame(tab_control)
    tab_control.add(tab_data_loading, text="1. Load Data")

    # Tab 2: Preprocessing and Modeling
    tab_modeling = ttk.Frame(tab_control)
    tab_control.add(tab_modeling, text="2. Modeling")

    tab_control.pack(expand=1, fill="both")

    # -------------
    # Tab 1 Layout
    # -------------
    ttk.Label(tab_data_loading, text="Load CSV Files for the Analysis", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, padx=10, pady=10)

    load_gene_expr_btn = ttk.Button(tab_data_loading, text="Load Gene Expression CSV", command=load_gene_expression_csv)
    load_gene_expr_btn.grid(row=1, column=0, padx=10, pady=10)

    load_metadata_btn = ttk.Button(tab_data_loading, text="Load Metadata CSV", command=load_metadata_csv)
    load_metadata_btn.grid(row=1, column=1, padx=10, pady=10)

    load_gene_list_btn = ttk.Button(tab_data_loading, text="Load Gene List CSV", command=load_gene_list_csv)
    load_gene_list_btn.grid(row=1, column=2, padx=10, pady=10)

    preprocess_btn = ttk.Button(tab_data_loading, text="Preprocess Data", command=preprocess_data)
    preprocess_btn.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

    # -------------
    # Tab 2 Layout
    # -------------
    ttk.Label(tab_modeling, text="Model Training and Evaluation", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, padx=10, pady=10)

    # Row 1: Random Forest
    rf_all_btn = ttk.Button(tab_modeling, text="Train RF (All Features)", command=run_random_forest)
    rf_all_btn.grid(row=1, column=0, padx=10, pady=10)

    identify_top_btn = ttk.Button(tab_modeling, text="Identify Top Genes (RF)", command=identify_top_genes)
    identify_top_btn.grid(row=1, column=1, padx=10, pady=10)

    rf_top_btn = ttk.Button(tab_modeling, text="Train RF (Top Genes)", command=run_random_forest_top_genes)
    rf_top_btn.grid(row=1, column=2, padx=10, pady=10)

    # Row 2: SVM
    svm_all_btn = ttk.Button(tab_modeling, text="Train SVM (All Features)", command=run_svm_all_features)
    svm_all_btn.grid(row=2, column=0, padx=10, pady=10)

    svm_top_btn = ttk.Button(tab_modeling, text="Train SVM (Top Genes)", command=run_svm_top_genes)
    svm_top_btn.grid(row=2, column=1, padx=10, pady=10)

    compare_matrices_btn = ttk.Button(tab_modeling, text="Compare Confusion Matrices", command=compare_confusion_matrices)
    compare_matrices_btn.grid(row=2, column=2, padx=10, pady=10)

    # -------------
    # Results Section
    # -------------
    global results_text
    results_label = ttk.Label(tab_modeling, text="Results/Output:")
    results_label.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="W")

    results_text = tk.Text(tab_modeling, width=100, height=20)
    results_text.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

    return window

def main():
    window = create_main_window()
    window.mainloop()

if __name__ == "__main__":
    main()
