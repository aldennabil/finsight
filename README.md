# FinSight: Advanced Customer Segmentation & Transaction Categorization

Welcome to **FinSight**, a comprehensive data science project designed for financial analysis, wealth management, customer segmentation, and transaction categorization. By combining advanced Natural Language Processing (NLP) models with robust dimensionality reduction techniques and advanced clustering algorithms, FinSight transforms raw transaction streams and customer demographic datasets into high-value, actionable segment insights.

---

## 📂 Repository Structure

The project workspace is organized as follows:

```filepath
DATA-SCIENCE/
├── .gitignore
├── README.md
├── data/
│   ├── Data Transaksi.csv                    # Raw Transaction Data
│   ├── df_transaksi(5).csv                   # Cleaned Transaction Data
│   ├── df_nasabah(3).csv                     # Customer Profiles & Demographics
│   ├── mcc_mapping.csv                       # Merchant Category Code Mapping Reference
│   ├── Analysis ready.csv                    # Processed and scaled feature set for modelling
│   ├── tfidf_vectorizer.pkl                  # Trained TF-IDF Vectorizer
│   └── nlp_model_nb.pkl                      # Trained Multinomial Naive Bayes Model
└── notebook/
    ├── Exploration and NLP.ipynb             # Exploratory Data Analysis, MCC Mapping & NLP Model Pipeline
    └── Dimensionality Reduction and Modelling.ipynb # Scaling, PCA, UMAP, and Advanced Clustering Algorithms
```

---

## 🛠️ Pipelines & Methodology

### 1. Exploratory Data Analysis & NLP Categorization Pipeline
Located in `notebook/Exploration and NLP.ipynb`, this pipeline addresses data cleaning, merchant category code mapping, and resolving transaction ambiguity.

*   **Cyclical Time Feature Engineering:** Trigonometric normalization of transaction timestamps (hours, days of the week, month) into cyclical sine/cosine representation (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`) to capture natural timing patterns.
*   **MCC Mapping & Rule-based Classification:** Seamlessly maps merchant transactions to detail categories based on standard `mcc_mapping.csv`. Transactions under MCC 4829 are dynamically marked as ambiguous P2P transfers.
*   **NLP Pipeline for Ambiguous Transactions:**
    *   **Text Preprocessing:** Case-folding and clean normalization of raw transaction strings and transfer notes.
    *   **Feature Extraction:** Text vectorization using **TF-IDF Vectorizer** (with Unigrams and Bigrams).
    *   **Machine Learning Model:** Evaluates and trains a **Multinomial Naive Bayes (MultinomialNB)** model, achieving a **97% overall accuracy** on classifying ambiguous transaction notes (e.g., separating grocery buy-backs, gaming top-ups, utility payments, or casual dining from P2P transfers).
    *   **Exportable Artifacts:** Model weights and vectorizers are automatically serialized (`nlp_model_nb.pkl`, `tfidf_vectorizer.pkl`) for downstream deployment.

---

### 2. Dimensionality Reduction & Advanced Customer Segmentation
Located in `notebook/Dimensionality Reduction and Modelling.ipynb`, this pipeline translates granular transaction profiles into distinct, high-level behavioral segmentations.

*   **Feature Transformation & Outlier Handling:**
    *   Log-transformation (`np.log1p`) applied to extremely skewed spend counts and surgings (e.g., weekend spend surges, wants frequency, balance volatility).
    *   **Winsorization** (capping extreme values to 1st and 99th percentiles) to neutralize extreme outliers without deleting customer records.
    *   **Standard Scaling** via `StandardScaler` to uniformize Euclidean distance metrics.
*   **Dimensionality Reduction:**
    *   **Principal Component Analysis (PCA):** Reduces the 10-dimensional space to components retaining **95% of total variance** to eliminate noise.
    *   **UMAP (Uniform Manifold Approximation and Projection):** Provides low-dimensional non-linear mapping for visual validation.
*   **Clustering Models & Evaluations:**
    *   Explores **K-Means**, **Agglomerative Hierarchical Clustering**, and **Gaussian Mixture Models (GMM)**.
    *   Rigorous cluster-validation utilizing four metric indexes:
        1.  **Silhouette Score** (for cluster tightness and separation).
        2.  **Davies-Bouldin Index (DBI)** (for inter-cluster similarity).
        3.  **Calinski-Harabasz Index (CHI)** (variance ratio criterion).
        4.  **Adjusted Rand Index (ARI)** (evaluation against baseline configurations).

---

## 🚀 How to Run the Project

### Prerequisites
Make sure you have Python 3.8+ and the following packages installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn umap-learn scipy tqdm ipywidgets
```

### Steps
1.  **Run the Exploratory and NLP Pipeline:**
    Open and run `notebook/Exploration and NLP.ipynb` from cell to cell. This will read the raw transactions in `data/`, perform MCC mapping, and train/save the NLP Naive Bayes pipeline.
2.  **Run the Dimensionality Reduction and Clustering Models:**
    Open and run `notebook/Dimensionality Reduction and Modelling.ipynb`. It will load the processed outputs (`Analysis ready.csv`), perform PCA and UMAP, and execute the clustering algorithms.

---
*Developed under the DBS-FinSight Capstone project to leverage transaction classification and customer financial behavior segmentation.*
