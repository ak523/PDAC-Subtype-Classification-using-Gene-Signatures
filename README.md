# PDAC Subtype Classification Using Gene Signatures

This repository hosts a **Streamlit** application for classifying Pancreatic Ductal Adenocarcinoma (PDAC) subtypes based on gene-expression signatures. The application facilitates PDAC subtype prediction, potentially aiding both research and clinical workflows.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Live Demo](#live-demo)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Directory Structure](#directory-structure)  
7. [References & Data Sources](#references--data-sources)  
8. [License](#license)  
9. [Contact](#contact)

---

## Overview

Pancreatic Ductal Adenocarcinoma (PDAC) is a highly aggressive cancer that can be categorized into multiple subtypes based on gene expression patterns. Proper classification can lead to better treatment decisions and more accurate prognoses.

**Key Objectives**:
- Use curated gene signatures to classify PDAC subtypes.  
- Provide a user-friendly web application for real-time subtype predictions.  

The web application is built with [Streamlit](https://streamlit.io/), enabling you to upload your own gene-expression data and obtain immediate subtype predictions.

---

## Features

- **Interactive Web Interface**: Easy-to-use interface to upload your data and view results.  
- **Real-Time Predictions**: Instantly classify PDAC subtypes by uploading CSV/TSV datasets.  
- **Visualization Tools**: View plots such as PCA, gene expression distributions, and other analytics.  
- **Explainability**: Inspect the top genes contributing to each classification for interpretability.

---

## Live Demo

A publicly accessible version of this app is available at:

[**PDAC Subtype Classification Using Gene Signatures**](https://pdac-subtype-classification-using-gene-signatures-wpaarrp8hf6c.streamlit.app/)

Visit the above link to:
- Upload your gene-expression file.  
- Get immediate classification results.  
- Explore additional plots and metrics.

---

## Installation

> **Note**: If you only need to use the hosted app, installation is not required. For local development, follow these steps:

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/pdac-subtype-classification.git
   cd pdac-subtype-classification
   ```

2. **Create and Activate a Virtual Environment** (optional but recommended)  
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   Make sure `requirements.txt` includes:
   - `streamlit`
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - and any other packages your code relies on

4. **Run the Application**  
   ```bash
   streamlit run app.py
   ```
   Once the server starts, open the provided local URL (e.g., `http://localhost:8501`) in your browser to access the app.

---

## Usage

1. **Prepare Your Data**  
   - Ensure the file has gene names and sample identifiers in the correct orientation.  
   - Confirm that the column names or row names match expected formats (see the app instructions).

2. **Upload the Data**  
   - Go to the “Upload File” section in the app.  
   - Drag and drop your CSV or TSV file or click to browse.

3. **View Results**  
   - The model will classify each sample into its predicted PDAC subtype.  
   - You can also see confidence scores or probability estimates, if available.

4. **Explore Additional Visualizations**  
   - Check PCA plots to see how samples cluster.  
   - Review gene-level expression patterns or boxplots for each subtype.

---

## Directory Structure

Below is an example layout for this project. Adjust as needed:

```
pdac-subtype-classification/
├── .devcontainer/                # Dev container configuration (optional, for VS Code or Docker)
├── .ipynb_checkpoints/           # Jupyter notebook checkpoints (auto-generated)
├── Data/                         # Directory containing your PDAC-related datasets
│   ├── GSE71729_expression.csv   # Example gene expression data
│   ├── GSE71729_metadata.csv     # Example metadata (Primary samples, tumor subtypes, etc.)
│   └── gene_signatures.csv       # Example signature file (lists of genes)
├── PDAC.ipynb                    # Jupyter notebook for data exploration and model experimentation
├── README.md                     # Project documentation (this file)
├── UI.py                         # Streamlit or UI logic (front-end of the web application)
├── main.py                       # Main entry point for running the app
└── requirements.txt              # Python dependencies (libraries needed to run the project)
```

---

## References & Data Sources

1. **TCGA (The Cancer Genome Atlas)** – [https://www.cancer.gov/tcga](https://www.cancer.gov/tcga)  
2. **GEO (Gene Expression Omnibus)** – [https://www.ncbi.nlm.nih.gov/geo/](https://www.ncbi.nlm.nih.gov/geo/)  
3. **Key Publications**:  
   - Bailey et al. (2016). *Nature*, 531, 47–52.  
   - Collisson et al. (2011). *Nature Medicine*, 17(4), 500–503.

---

## License

This project is licensed under the [MIT License](LICENSE).  
Feel free to modify and distribute under these terms.

---

## Contact

If you have any questions, suggestions, or would like to collaborate:
- **GitHub Issues**: Create an issue for bugs, enhancements, or feature requests.
- **Email**: [aryankadiya@gmail.com](mailto:aryankadiya@gmail.com)

We welcome community contributions! Consider opening a pull request for new features or improvements.

---

*Thank you for using PDAC Subtype Classification Using Gene Signatures.*
