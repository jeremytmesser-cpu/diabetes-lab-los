# Lab Tests and Length of Stay in Diabetes Hospitalizations

A [Marimo](https://marimo.io) notebook that explores one question about hospital stays for patients with diabetes:

> **How is the number of lab procedures performed during a stay related to the length of that stay, and does this relationship differ across admission types (Elective, Emergency, Urgent)?**

## What the notebook does

`analysis.py` is the notebook. Every code cell is preceded by a markdown cell explaining why the code was written. It covers:

- **Data operations:** filtering by column values, sorting, grouping by `admission_type`, and adding computed columns (`stay_length` bins and `labs_per_day`)
- **Visualizations:** a histogram, a boxplot, a scatter plot, and a group-comparison chart (correlation by admission type with 95% confidence intervals)
- **Statistics:** Pearson and Spearman correlations, per-group correlations, and a z-test comparing correlations between groups
- **Conclusions** that pull their numbers from the computed results

Key finding: more lab procedures go with longer stays in every admission type (overall r = 0.34), and the link is strongest for elective admissions (r = 0.46) compared with emergency (r = 0.32) and urgent (r = 0.33). This is a correlation, not evidence that lab tests cause longer stays.

## Data

`data/diabetes_130.csv` is a simplified 3,000-stay extract of the Diabetes 130-US Hospitals dataset (130 US hospitals, 1999-2008), provided for this course.

- Original data: <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
- Citation: Strack et al. (2014). *Impact of HbA1c measurement on hospital readmission rates.* BioMed Research International. Data: Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

Columns used for the research question: `admission_type`, `time_in_hospital`, `num_lab_procedures`.

## How to run it

You need Python 3.10 or newer.

```bash
# 1. Get the code
git clone https://github.com/jeremytmesser-cpu/diabetes-lab-los.git
cd diabetes-lab-los

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook in the editor...
marimo edit analysis.py

#    ...or view it as a read-only app
marimo run analysis.py
```

Run the commands from the repository root: the notebook reads `data/diabetes_130.csv` with a relative path. Running the notebook also saves the four charts to `figures/`.

## Project layout

```
analysis.py        Marimo notebook (the analysis)
data/              diabetes_130.csv
figures/           charts saved by the notebook
requirements.txt   Python dependencies
```
