# PhD Applications Analysis Tool

This tool analyzes and visualizes PhD application data using Python and Plotly to provide interactive insights into applicant demographics, academic qualifications, test scores, and relationships between various metrics.

## Features

- **Data Cleaning**: Standardizes country names, handles inconsistent encodings, converts data types, and prepares data for analysis.
- **Descriptive Analysis**: 
  - Interactive plots for demographics (citizenship, student types, applicant types).
  - Interactive plots for academic qualifications (GPAs, degree types, subject areas).
  - Interactive plots for test scores (English proficiency tests, GRE scores).
- **Relationship Analysis**:
  - Interactive scatter plots for GPA vs. test scores.
  - Interactive scatter plot for GRE verbal vs. quantitative scores.
  - Interactive box plots for country-specific comparisons.
- **Interactive Dashboard**: A single HTML file (`phd_applications_dashboard.html`) containing all interactive visualizations with explanations, generated using Plotly.

## Setup and Usage

1.  **Install Dependencies**: Ensure you have Python installed. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data File**: Place your PhD application data in a CSV file named `PhD Applications.csv` in the same directory as the script.

3.  **Run Analysis**: Execute the Python script from your terminal:
    ```bash
    python analyze_phd_applications.py
    ```

4.  **View Results**:
    - A cleaned version of the data is saved as `cleaned_phd_applications.csv`.
    - Summary statistics are saved in `summary_statistics.csv`.
    - Open the **interactive dashboard** `phd_applications_dashboard.html` in a web browser to view and explore the visualizations.

## Data Requirements

The script expects a CSV file (`PhD Applications.csv`) with columns roughly matching the original example, including:
- Application ID
- Citizenship Country
- Student Type
- Applicant Type
- Institution
- GPA
- Degree
- Subject Area
- GRE Verbal/Quantitative scores
- English Proficiency Test Type and Score

## Output Files

- **Cleaned Data**: `cleaned_phd_applications.csv`
- **Summary Statistics**: `summary_statistics.csv`
- **Interactive Dashboard**: `phd_applications_dashboard.html` (Contains all visualizations) 