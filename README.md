# 🥋 TKD Physical Evaluation – Dashboard & Analytics

A premium analytics platform built with **Streamlit** to track, analyze, and visualize physical evaluation data for Taekwondo athletes. This tool leverages **Principal Component Analysis (PCA)** to transform raw physical test results into meaningful performance dimensions: **Endurance**, **Speed**, **Strength**, and **Flexibility**.

---

## ✨ Overview

This project was developed to provide coaches and athletes with a clear, data-driven view of physical progress. By processing Excel-based evaluation records, the dashboard generates interactive radar charts that allow for:
- **Individual Progress Tracking:** Compare an athlete's performance from the beginning of the season to mid-season.
- **Category Benchmarking:** Compare average performance across different age groups and genders (*Escalões*).
- **Multi-Dimensional Analysis:** View homogenized scores (0–1 scale) derived from complex physical datasets.

---

## 🚀 Key Features

*   **Holistic Dimension Scoring:** Automatically groups test results into four core athletic pillars:
    *   **Endurance:** Beep test (*Vaivém*), 1-minute kicks.
    *   **Speed:** 4x10m shuttle, 20m sprint, reaction times.
    *   **Strength:** Horizontal/vertical jumps, abdominals, push-ups.
    *   **Flexibility:** Sit-and-Reach, front/side splits, kick amplitude.
*   **Intelligent Data Processing:**
    *   **PCA Integration:** Uses Scikit-learn to identify the most significant performance factors.
    *   **Smart Imputation:** Handles missing data by filling values based on category averages.
    *   **Dual Scaling:** Supports both "Local Scaling" (relative to the athlete's category) and "Global Scaling" (relative to the entire team).
*   **Interactive Visualizations:** High-quality radar charts powered by **Plotly** for intuitive performance comparison.
*   **Modular Architecture:** Easily adaptable for different evaluation phases (Start/Middle/End of season).

---

## 🛠️ Tech Stack

- **Frontend/App Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning/Stats:** [Scikit-learn](https://scikit-learn.org/) (PCA, Scalers, Imputers)
- **Data Visualization:** [Plotly](https://plotly.com/)
- **File Handling:** [OpenPyXL](https://openpyxl.readthedocs.io/) (Excel processing)

---

## 📋 Committed Project Files

| File | Description |
| :--- | :--- |
| `notebooks/dashboard_avaliacoes_fisicas_tce.py` | The main Streamlit dashboard application. |
| `requirements.txt` | Project dependencies and library versions. |
| `.gitignore` | Standard Git exclusion rules. |

---

## ⚙️ Setup & Installation

Follow these steps to run the dashboard locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/anacoelho92/tkd-physical-evaluation.git
    cd tkd-physical-evaluation
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run notebooks/dashboard_avaliacoes_fisicas_tce.py
    ```

---

## 📊 Data Requirements

The application expects an Excel file (`.xlsx`) with the following structure:
- **Sheet Name:** `Avaliações Físicas`
- **Header:** Starting at row 2 (index 1).
- **Columns:** Must include athlete metadata (Atleta, Sexo, Idade, Categoria, Fase) and physical test columns (e.g., *4x10m*, *Impulsão horizontal*, etc.).

You can place your data in the `data/` directory or specify a custom path directly in the dashboard sidebar.

---

## 🛡️ License

This project is intended for internal use and coaching staff within the TCE Taekwondo team.
