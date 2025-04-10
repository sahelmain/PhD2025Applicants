import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import warnings
import scipy.stats as stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set default Plotly template
import plotly.io as pio
pio.templates.default = "plotly_dark"

# Try different encodings
encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
df = None
for encoding in encodings:
    try:
        # Read data with specified encoding
        df = pd.read_csv('PhD Applications.csv', skiprows=1, encoding=encoding)
        logging.info(f"Successfully read file with {encoding} encoding")
        break
    except Exception as e:
        logging.error(f"Failed to read with {encoding} encoding: {e}")

if df is None:
    logging.critical("Error: Could not read the CSV file with any of the attempted encodings.")
    exit()

# Data Cleaning
# -----------------

# Display initial data types
logging.info("Initial data types:")
logging.info(df.dtypes)

# Fix column names (remove trailing spaces)
df.columns = df.columns.str.strip()

# Remove leading/trailing spaces from string columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]

# Validate and convert numeric columns to appropriate types
# GPA column
if df['GPA'].apply(lambda x: isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()).all():
    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')
else:
    logging.warning("Non-numeric values found in GPA column, converting to NaN.")
    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')

if df['GPA 2'].apply(lambda x: isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()).all():
    df['GPA 2'] = pd.to_numeric(df['GPA 2'], errors='coerce')
else:
    logging.warning("Non-numeric values found in GPA 2 column, converting to NaN.")
    df['GPA 2'] = pd.to_numeric(df['GPA 2'], errors='coerce')

# GRE scores
if df['Verbal'].apply(lambda x: isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()).all():
    df['Verbal'] = pd.to_numeric(df['Verbal'], errors='coerce')
else:
    logging.warning("Non-numeric values found in Verbal column, converting to NaN.")
    df['Verbal'] = pd.to_numeric(df['Verbal'], errors='coerce')

if df['Quantitative'].apply(lambda x: isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()).all():
    df['Quantitative'] = pd.to_numeric(df['Quantitative'], errors='coerce')
else:
    logging.warning("Non-numeric values found in Quantitative column, converting to NaN.")
    df['Quantitative'] = pd.to_numeric(df['Quantitative'], errors='coerce')

# English Test Score
if df['Score'].apply(lambda x: isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()).all():
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
else:
    logging.warning("Non-numeric values found in Score column, converting to NaN.")
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

# Standardize country names
country_mapping = {
    'China ': 'China',
    'Bangladesh ': 'Bangladesh',
    'Iran ': 'Iran',
    'Pakistan ': 'Pakistan',
    'Nepal ': 'Nepal',
    'Nigeria ': 'Nigeria',
    'Viet nam': 'Vietnam',
    'Viet Nam': 'Vietnam'
}
df['Citizenship Country'] = df['Citizenship Country'].replace(country_mapping)

# Display converted data types
logging.info("Converted data types:")
logging.info(df.dtypes)

# Show basic statistics to verify conversion
logging.info("Basic statistics for numeric columns:")
logging.info(df[['GPA', 'GPA 2', 'Verbal', 'Quantitative', 'Score']].describe())

# Save cleaned data
df.to_csv('cleaned_phd_applications.csv', index=False)
logging.info("Data cleaning completed. Saved to 'cleaned_phd_applications.csv'")

# Add LinkedIn URL field and sample values for demonstration
df['LinkedIn_URL'] = None  # Initialize empty LinkedIn URL field

# For demonstration, let's add some sample LinkedIn URLs for a few candidates
# In a real implementation, these would be found through manual research or API search
linkedin_sample_data = {
    0: "https://www.linkedin.com/in/sample-profile-1/",
    5: "https://www.linkedin.com/in/sample-profile-2/",
    10: "https://www.linkedin.com/in/sample-profile-3/",
    15: "https://www.linkedin.com/in/sample-profile-4/",
    20: "https://www.linkedin.com/in/sample-profile-5/"
}

# Assign sample URLs to a few rows
for idx, url in linkedin_sample_data.items():
    if idx < len(df):
        df.at[idx, 'LinkedIn_URL'] = url

# Machine Learning Analysis
# ------------------------
logging.info("Starting Machine Learning Analysis...")

# Preparing data for clustering
ml_features = ['GPA', 'Score']  # Basic features everyone should have
# Add GRE scores if available
if df['Verbal'].notna().sum() > 5 and df['Quantitative'].notna().sum() > 5:
    ml_features.extend(['Verbal', 'Quantitative'])

# Extract data for clustering
cluster_df = df[ml_features].copy()

# Impute missing values
imputer = SimpleImputer(strategy='mean')
cluster_data = imputer.fit_transform(cluster_df)

# Scale the data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Determine optimal number of clusters using the Elbow Method
inertia = []
k_range = range(2, 6)  # Try 2-5 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Create elbow curve
fig_elbow = px.line(x=list(k_range), y=inertia, 
                    title='Elbow Method for Optimal k',
                    labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'},
                    template='plotly_dark')
fig_elbow.update_layout(
    dragmode='zoom',
    hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
    plot_bgcolor='rgba(25,25,25,1)',
    paper_bgcolor='rgba(25,25,25,0)',
    margin=dict(l=40, r=40, t=60, b=40),
    title_font=dict(size=20, color='#F28E2B')
)
fig_elbow.add_annotation(
    x=3, y=inertia[1],  # Position may need adjustment
    text="Optimal k",
    showarrow=True,
    arrowhead=1,
    ax=50, ay=-50
)

# Apply K-means with the optimal k
optimal_k = 3  # Based on elbow method visualization
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Perform PCA for visualization if we have more than 2 features
if len(ml_features) > 2:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = clusters
    
    # Create PCA visualization
    fig_pca = px.scatter(
        pca_df, x='PCA1', y='PCA2', 
        color='Cluster',
        color_continuous_scale='viridis',
        title='PCA: Applicant Clusters',
        labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
        template='plotly_dark'
    )
    
    # Get the explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    fig_pca.update_layout(
        annotations=[
            dict(
                x=0.5, y=1.05,
                text=f"Explained Variance: PC1 {explained_variance[0]:.2%}, PC2 {explained_variance[1]:.2%}",
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(size=12)
            )
        ],
        dragmode='zoom',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color='#F28E2B')
    )
else:
    # If only 2 features, visualize directly
    cluster_viz_df = pd.DataFrame(scaled_data, columns=ml_features)
    cluster_viz_df['Cluster'] = clusters
    
    fig_pca = px.scatter(
        cluster_viz_df, x=ml_features[0], y=ml_features[1], 
        color='Cluster',
        color_continuous_scale='viridis',
        title=f'Clusters of Applicants by {ml_features[0]} and {ml_features[1]}',
        labels={ml_features[0]: ml_features[0], ml_features[1]: ml_features[1]},
        template='plotly_dark'
    )
    fig_pca.update_layout(
        dragmode='zoom',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color='#F28E2B')
    )

# Analyze the clusters
cluster_stats = df.groupby('Cluster')[ml_features].mean().reset_index()
cluster_counts = df['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']
cluster_stats = cluster_stats.merge(cluster_counts, on='Cluster')

# Create clustered parallel coordinates plot
fig_parallel_clusters = go.Figure()
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    dimensions = [dict(range=[df[col].min(), df[col].max()],
                        label=col, values=cluster_data[col]) 
                   for col in ml_features if cluster_data[col].notna().sum() > 0]
    
    if dimensions:  # Only add if we have valid dimensions
        fig_parallel_clusters.add_trace(
            go.Parcoords(
                line=dict(color=f'rgb({50 + i*70}, {100 + i*50}, {150 + i*30})'),
                dimensions=dimensions,
                name=f'Cluster {i}'
            )
        )

fig_parallel_clusters.update_layout(
    title='Parallel Coordinates Plot by Cluster',
    plot_bgcolor='rgba(25,25,25,1)',
    paper_bgcolor='rgba(25,25,25,0)',
    title_font=dict(size=20, color='#F28E2B'),
    margin=dict(l=80, r=80, t=80, b=80)
)

# Create radar chart to compare clusters
fig_radar_clusters = go.Figure()

# Normalize the features for radar chart
radar_features = ml_features.copy()
radar_df = df[radar_features].copy()
radar_max = radar_df.max()
radar_min = radar_df.min()
radar_df = (radar_df - radar_min) / (radar_max - radar_min)  # Scale to 0-1

# Add application count as a feature
radar_df['Application Count'] = 1  # Each row is one application

# Create radar chart traces for each cluster
for cluster_id in range(optimal_k):
    cluster_means = radar_df[df['Cluster'] == cluster_id].mean().reset_index()
    cluster_means.columns = ['Feature', 'Value']
    
    fig_radar_clusters.add_trace(go.Scatterpolar(
        r=cluster_means['Value'],
        theta=cluster_means['Feature'],
        fill='toself',
        name=f'Cluster {cluster_id}'
    ))

fig_radar_clusters.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    title='Cluster Characteristic Comparison',
    title_font=dict(size=20, color='#F28E2B'),
    paper_bgcolor='rgba(25,25,25,0)',
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    )
)

logging.info(f"Machine Learning Analysis completed. Identified {optimal_k} distinct clusters of applicants.")

# Store ML plots in the plots_html dictionary
plots_html = {}
plots_html['elbow_curve'] = fig_elbow.to_html(full_html=False, include_plotlyjs='cdn')
plots_html['cluster_pca'] = fig_pca.to_html(full_html=False, include_plotlyjs=False)
plots_html['parallel_clusters'] = fig_parallel_clusters.to_html(full_html=False, include_plotlyjs=False)
plots_html['radar_clusters'] = fig_radar_clusters.to_html(full_html=False, include_plotlyjs=False)

# Color scheme
primary_color = "#4E79A7"
secondary_color = "#F28E2B"
accent_colors = ["#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]

# Dictionary to store plot HTML
plots_html['title_section'] = f"""
<div class="title-container">
    <div class="dashboard-title">PhD Applications Analysis</div>
    <div class="dashboard-subtitle">Interactive Exploration of Computer Science PhD Applicant Data</div>
    
    <div class="filter-controls">
        <div class="filter-row">
            <div class="filter-item">
                <label for="country-filter">Country:</label>
                <select id="country-filter" class="filter-select">
                    <option value="all">All Countries</option>
                    {' '.join([f'<option value="{country}">{country}</option>' for country in sorted(df['Citizenship Country'].dropna().unique())])}
                </select>
            </div>
            <div class="filter-item">
                <label for="term-filter">Term:</label>
                <select id="term-filter" class="filter-select">
                    <option value="all">All Terms</option>
                    {' '.join([f'<option value="{term}">{term}</option>' for term in sorted(df['Intended Term'].dropna().unique())])}
                </select>
            </div>
        </div>
        <div class="filter-row">
            <div class="filter-item">
                <label for="applicant-type-filter">Applicant Type:</label>
                <select id="applicant-type-filter" class="filter-select">
                    <option value="all">All Types</option>
                    {' '.join([f'<option value="{app_type}">{app_type}</option>' for app_type in sorted(df['Applicant Type'].dropna().unique())])}
                </select>
            </div>
            <div class="filter-item">
                <label for="student-type-filter">Student Type:</label>
                <select id="student-type-filter" class="filter-select">
                    <option value="all">All Student Types</option>
                    {' '.join([f'<option value="{stu_type}">{stu_type}</option>' for stu_type in sorted(df['Student Type'].dropna().unique())])}
                </select>
            </div>
        </div>
        <button id="apply-filters" class="action-button">Apply Filters</button>
        <button id="reset-filters" class="action-button secondary">Reset</button>
    </div>
    
    <div class="stats-highlights">
        <div class="stat-item" data-value="{len(df)}">
            <div class="stat-value counter">{len(df)}</div>
            <div class="stat-label">Total Applications</div>
        </div>
        <div class="stat-item" data-value="{df['Citizenship Country'].nunique()}">
            <div class="stat-value counter">{df['Citizenship Country'].nunique()}</div>
            <div class="stat-label">Countries</div>
        </div>
        <div class="stat-item">
            <div class="stat-value counter-decimal">{df['GPA'].mean():.2f}</div>
            <div class="stat-label">Avg GPA</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{df['Test Type'].value_counts().index[0]}</div>
            <div class="stat-label">Top Test</div>
        </div>
    </div>
</div>
"""

# 1. DESCRIPTIVE ANALYSIS
# -----------------------

# 1.1 Citizenship Distribution
citizenship_counts = df['Citizenship Country'].value_counts()
citizenship_counts = pd.DataFrame({'Country': citizenship_counts.index, 'Count': citizenship_counts.values})

student_type_counts = df['Student Type'].value_counts()
student_type_counts = pd.DataFrame({'Type': student_type_counts.index, 'Count': student_type_counts.values})

applicant_type_counts = df['Applicant Type'].value_counts()
applicant_type_counts = pd.DataFrame({'Type': applicant_type_counts.index, 'Count': applicant_type_counts.values})

# Create plot figures
citizenship_fig = px.bar(citizenship_counts, 
                        x='Country', y='Count',
                        title='Distribution of Applicants by Country',
                        labels={'Count': 'Number of Applicants'})

student_type_fig = px.bar(student_type_counts, 
                         x='Type', y='Count',
             title='Distribution of Applicants by Student Type',
                         labels={'Count': 'Number of Applicants'})

applicant_type_fig = px.bar(applicant_type_counts, 
                           x='Type', y='Count',
             title='Distribution of Applicants by Applicant Type',
                           labels={'Count': 'Number of Applicants'})

# GPA Distribution
gpa_data = df['GPA'].dropna()
gpa_fig = px.histogram(gpa_data, 
                      nbins=20,
                   title='Distribution of First Institution GPA',
                      labels={'value': 'GPA', 'count': 'Number of Applicants'},
                      opacity=0.8)

# Add mean and median lines
gpa_fig.add_vline(x=gpa_data.mean(), 
                  line_dash="dash", 
                  line_color="#F28E2B",
              annotation_text=f"Mean: {gpa_data.mean():.2f}",
                  annotation_position="top right")
gpa_fig.add_vline(x=gpa_data.median(), 
                  line_dash="dash", 
                  line_color="#59A14F",
              annotation_text=f"Median: {gpa_data.median():.2f}",
                  annotation_position="top left")

# Subject Area Distribution
subject_area_counts = df['Subject Area'].value_counts()
subject_area_counts = pd.DataFrame({'Area': subject_area_counts.index, 'Count': subject_area_counts.values})
subject_area_fig = px.bar(subject_area_counts.head(15), 
                         y='Area', x='Count',
                         orientation='h',
             title='Top 15 Subject Areas of Applicants',
                         labels={'Count': 'Number of Applicants'})

# Degree Type Distribution
degree_counts = df['Degree'].value_counts()
degree_counts = pd.DataFrame({'Degree': degree_counts.index, 'Count': degree_counts.values})
degree_type_fig = px.bar(degree_counts,
                        x='Degree', y='Count',
             title='Distribution of Degree Types',
                        labels={'Count': 'Number of Applicants'})

# English Test Distribution
test_type_counts = df['Test Type'].value_counts()
test_type_counts = pd.DataFrame({'Test': test_type_counts.index, 'Count': test_type_counts.values})
english_test_fig = px.bar(test_type_counts,
                         x='Test', y='Count',
                         title='Distribution of English Proficiency Test Types',
                         labels={'Count': 'Number of Applicants'})

# GRE Score Distributions
verbal_data = df['Verbal'].dropna()
gre_verbal_fig = px.histogram(verbal_data,
                             nbins=15,
                       title='Distribution of GRE Verbal Scores',
                             labels={'value': 'Verbal Score', 'count': 'Number of Applicants'},
                             opacity=0.8)

quant_data = df['Quantitative'].dropna()
gre_quant_fig = px.histogram(quant_data,
                            nbins=15,
                       title='Distribution of GRE Quantitative Scores',
                            labels={'value': 'Quantitative Score', 'count': 'Number of Applicants'},
                            opacity=0.8)

# Relationship plots
gpa_vs_gre_verbal_fig = px.scatter(df.dropna(subset=['GPA', 'Verbal']),
                                  x='GPA', y='Verbal',
                                  title='GPA vs GRE Verbal Score',
                                  trendline='ols')

gpa_vs_gre_quant_fig = px.scatter(df.dropna(subset=['GPA', 'Quantitative']),
                                 x='GPA', y='Quantitative',
                                 title='GPA vs GRE Quantitative Score',
                                 trendline='ols')

gre_verbal_vs_quant_fig = px.scatter(df.dropna(subset=['Verbal', 'Quantitative']),
                                    x='Verbal', y='Quantitative',
                                    title='GRE Verbal vs Quantitative Scores',
                                    trendline='ols')

# Configure plot layouts for consistent sizing
plot_config = {
    'displayModeBar': True,
    'responsive': True,
    'scrollZoom': True
}

plot_layout = {
    'template': 'plotly_dark',
    'paper_bgcolor': 'rgba(25,25,25,0)',
    'plot_bgcolor': 'rgba(25,25,25,1)',
    'font': {'family': 'Roboto', 'color': '#f2f5fa'},
    'title': {'font': {'size': 20, 'color': '#F28E2B'}},
    'margin': {'l': 40, 'r': 40, 't': 60, 'b': 40},
    'hoverlabel': {'font': {'size': 12, 'family': 'Roboto'}, 'bgcolor': 'rgba(0,0,0,0.8)'},
    'dragmode': 'zoom',
    'height': 400,
    'autosize': True
}

# Update all plot layouts
for fig in [citizenship_fig, student_type_fig, applicant_type_fig,
            gpa_fig, subject_area_fig, degree_type_fig,
            english_test_fig, gre_verbal_fig, gre_quant_fig,
            gpa_vs_gre_verbal_fig, gpa_vs_gre_quant_fig, gre_verbal_vs_quant_fig]:
    fig.update_layout(**plot_layout)

# Convert plots to HTML with consistent config
plots_html = {
    'citizenship_distribution': citizenship_fig.to_html(config=plot_config, full_html=False),
    'student_type_distribution': student_type_fig.to_html(config=plot_config, full_html=False),
    'applicant_type_distribution': applicant_type_fig.to_html(config=plot_config, full_html=False),
    'gpa_distribution': gpa_fig.to_html(config=plot_config, full_html=False),
    'subject_area_distribution': subject_area_fig.to_html(config=plot_config, full_html=False),
    'degree_type_distribution': degree_type_fig.to_html(config=plot_config, full_html=False),
    'english_test_distribution': english_test_fig.to_html(config=plot_config, full_html=False),
    'gre_verbal_distribution': gre_verbal_fig.to_html(config=plot_config, full_html=False),
    'gre_quant_distribution': gre_quant_fig.to_html(config=plot_config, full_html=False),
    'gpa_vs_gre_verbal': gpa_vs_gre_verbal_fig.to_html(config=plot_config, full_html=False),
    'gpa_vs_gre_quant': gpa_vs_gre_quant_fig.to_html(config=plot_config, full_html=False),
    'gre_verbal_vs_quant': gre_verbal_vs_quant_fig.to_html(config=plot_config, full_html=False)
}

# Create the complete HTML content with the plots embedded
html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhD Applications Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --bg-color: #1a1a1a;
            --text-color: #f2f5fa;
            --primary-color: #F28E2B;
            --secondary-color: #59A14F;
            --accent-color: #4E79A7;
            --border-color: #2d2d2d;
        }}

        body {{
            font-family: 'Roboto', sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-color);
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .nav {{
            background: rgba(25, 25, 25, 0.9);
            padding: 15px 0;
            position: sticky;
            top: 0;
            z-index: 100;
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border-color);
        }}

        .nav-content {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .nav-links {{
            display: flex;
            gap: 20px;
        }}

        .nav-link {{
            color: var(--text-color);
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 4px;
            transition: background-color 0.3s;
        }}

        .nav-link:hover {{
            background-color: rgba(255, 255, 255, 0.1);
        }}

        .header {{
            text-align: center;
            padding: 40px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 40px;
        }}

        h1 {{
            color: var(--primary-color);
            font-size: 2.5em;
            margin: 0;
        }}

        .description {{
            color: var(--text-color);
            font-size: 1.1em;
            margin: 20px 0;
            line-height: 1.6;
        }}

        .section {{
            margin-bottom: 40px;
            padding: 20px;
            background-color: rgba(45, 45, 45, 0.5);
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        h2 {{
            color: var(--primary-color);
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}

        .plot-container {{
            margin-bottom: 30px;
            background: rgba(25, 25, 25, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            height: 450px;
            width: 100%;
            display: flex;
            flex-direction: column;
        }}

        .plot-container h3 {{
            color: var(--primary-color);
            margin: 0 0 10px 0;
            font-size: 1.4em;
        }}

        .plot-container p {{
            color: var(--text-color);
            margin: 0 0 15px 0;
            font-size: 1em;
            line-height: 1.5;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
            
            .plot-container {{
                height: 400px;
            }}

            .nav-links {{
                display: none;
            }}
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            border-top: 1px solid var(--border-color);
            margin-top: 40px;
            color: var(--text-color);
        }}
    </style>
</head>
<body>
    <nav class="nav">
        <div class="nav-content">
            <div class="nav-links">
                <a href="#demographics" class="nav-link">Demographics</a>
                <a href="#academics" class="nav-link">Academics</a>
                <a href="#test-scores" class="nav-link">Test Scores</a>
                <a href="#relationships" class="nav-link">Relationships</a>
            </div>
        </div>
    </nav>

    <div class="container">
        <div class="header">
            <h1>PhD Applications Analysis Dashboard</h1>
            <p class="description">
                Interactive visualization of PhD application data, including demographics,
                academic performance, and test scores.
            </p>
        </div>

        <div id="demographics" class="section">
            <h2>Demographics</h2>
            <div class="grid">
                <div class="plot-container">
                    <h3>Citizenship Distribution</h3>
                    {citizenship_distribution}
                </div>
                <div class="plot-container">
                    <h3>Student Type Distribution</h3>
                    {student_type_distribution}
                </div>
                <div class="plot-container">
                    <h3>Applicant Type Distribution</h3>
                    {applicant_type_distribution}
                </div>
            </div>
        </div>

        <div id="academics" class="section">
            <h2>Academic Performance</h2>
            <div class="grid">
                <div class="plot-container">
                    <h3>GPA Distribution</h3>
                    {gpa_distribution}
                </div>
                <div class="plot-container">
                    <h3>Subject Areas</h3>
                    {subject_area_distribution}
                </div>
                <div class="plot-container">
                    <h3>Degree Types</h3>
                    {degree_type_distribution}
                </div>
            </div>
        </div>

        <div id="test-scores" class="section">
            <h2>Test Scores</h2>
            <div class="grid">
                <div class="plot-container">
                    <h3>English Test Types</h3>
                    {english_test_distribution}
                </div>
                <div class="plot-container">
                    <h3>GRE Verbal Scores</h3>
                    {gre_verbal_distribution}
                </div>
                <div class="plot-container">
                    <h3>GRE Quantitative Scores</h3>
                    {gre_quant_distribution}
                </div>
            </div>
        </div>

        <div id="relationships" class="section">
            <h2>Score Relationships</h2>
            <div class="grid">
                <div class="plot-container">
                    <h3>GPA vs GRE Verbal</h3>
                    {gpa_vs_gre_verbal}
                </div>
                <div class="plot-container">
                    <h3>GPA vs GRE Quantitative</h3>
                    {gpa_vs_gre_quant}
                </div>
                <div class="plot-container">
                    <h3>GRE Verbal vs Quantitative</h3>
                    {gre_verbal_vs_quant}
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Created with Python, Plotly, and Pandas</p>
        </div>
    </div>
</body>
</html>'''

# Generate the dashboard
try:
    dashboard_html = html_template.format(**plots_html)
    with open('phd_applications_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    logging.info("Interactive dashboard created at 'phd_applications_dashboard.html'")
    print("Analysis completed! Check out the interactive dashboard.")
except Exception as e:
    logging.error(f"Error generating dashboard HTML: {str(e)}")
    print("Failed to generate dashboard. Check the logs for details.")