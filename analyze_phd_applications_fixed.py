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
import subprocess
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Initialize dictionary to store plot HTML
plots_html = {}

# Machine Learning Analysis
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

# Apply K-means with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Perform PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create PCA visualization
cluster_viz_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
cluster_viz_df['Cluster'] = clusters

# Create cluster visualization
fig_clusters = px.scatter(
    cluster_viz_df,
    x='PC1',
    y='PC2',
    color='Cluster',
    title='Applicant Clusters (PCA)',
    labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'},
    color_discrete_sequence=['#4E79A7', '#F28E2B', '#59A14F']
)

# Add explained variance ratio to the plot
explained_variance = pca.explained_variance_ratio_
fig_clusters.add_annotation(
    text=f'Explained Variance: PC1 {explained_variance[0]:.2%}, PC2 {explained_variance[1]:.2%}',
    xref='paper', yref='paper',
    x=0.5, y=1.05,
    showarrow=False
)

fig_clusters.update_layout(
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
    title_font=dict(size=20, color='#F28E2B'),
    showlegend=True,
    legend=dict(
        title='Cluster',
        bgcolor='rgba(50,50,50,0.8)',
        bordercolor='rgba(255,255,255,0.3)',
        borderwidth=1
    )
)

# Generate cluster summary
cluster_summary = []
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    summary = {
        'size': len(cluster_data),
        'avg_gpa': cluster_data['GPA'].mean(),
        'avg_score': cluster_data['Score'].mean(),
        'top_countries': cluster_data['Citizenship Country'].value_counts().head(3).index.tolist()
    }
    if 'Verbal' in ml_features:
        summary['avg_verbal'] = cluster_data['Verbal'].mean()
        summary['avg_quant'] = cluster_data['Quantitative'].mean()
    cluster_summary.append(summary)

# Create cluster summary text
cluster_summary_text = """
<div class="cluster-summary">
    <h3>Understanding the Clusters</h3>
    <p>The machine learning analysis has identified three distinct groups of applicants:</p>
    
    <div class="cluster-groups">
        <div class="cluster-group">
            <h4>Cluster 0: {}</h4>
            <ul>
                <li>Size: {} applicants</li>
                <li>Average GPA: {:.2f}</li>
                <li>Average Test Score: {:.2f}</li>
                <li>Top Countries: {}</li>
            </ul>
        </div>
        
        <div class="cluster-group">
            <h4>Cluster 1: {}</h4>
            <ul>
                <li>Size: {} applicants</li>
                <li>Average GPA: {:.2f}</li>
                <li>Average Test Score: {:.2f}</li>
                <li>Top Countries: {}</li>
            </ul>
        </div>
        
        <div class="cluster-group">
            <h4>Cluster 2: {}</h4>
            <ul>
                <li>Size: {} applicants</li>
                <li>Average GPA: {:.2f}</li>
                <li>Average Test Score: {:.2f}</li>
                <li>Top Countries: {}</li>
            </ul>
        </div>
    </div>
    
    <p class="cluster-note">Note: The clustering is based on academic metrics (GPA and test scores). 
    The visualization uses PCA to reduce dimensionality while preserving {:.1%} of the variance in the data.</p>
</div>
""".format(
    "High Academic Achievers" if cluster_summary[0]['avg_gpa'] > 3.5 else "Research/Experience Focus" if cluster_summary[0]['avg_gpa'] > 3.0 else "Diverse Background",
    cluster_summary[0]['size'],
    cluster_summary[0]['avg_gpa'],
    cluster_summary[0]['avg_score'],
    ", ".join(cluster_summary[0]['top_countries']),
    
    "High Academic Achievers" if cluster_summary[1]['avg_gpa'] > 3.5 else "Research/Experience Focus" if cluster_summary[1]['avg_gpa'] > 3.0 else "Diverse Background",
    cluster_summary[1]['size'],
    cluster_summary[1]['avg_gpa'],
    cluster_summary[1]['avg_score'],
    ", ".join(cluster_summary[1]['top_countries']),
    
    "High Academic Achievers" if cluster_summary[2]['avg_gpa'] > 3.5 else "Research/Experience Focus" if cluster_summary[2]['avg_gpa'] > 3.0 else "Diverse Background",
    cluster_summary[2]['size'],
    cluster_summary[2]['avg_gpa'],
    cluster_summary[2]['avg_score'],
    ", ".join(cluster_summary[2]['top_countries']),
    
    sum(explained_variance)
)

# Add cluster visualization to plots
plots_html['cluster_plot'] = fig_clusters.to_html(full_html=False, include_plotlyjs=False)
plots_html['cluster_summary'] = cluster_summary_text

logging.info("Machine Learning Analysis completed. Identified 3 distinct clusters of applicants.")

# Analyze the clusters
cluster_stats = df.groupby('Cluster')[ml_features].mean().reset_index()
cluster_counts = df['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']
cluster_stats = cluster_stats.merge(cluster_counts, on='Cluster')

# Create clustered parallel coordinates plot
fig_parallel_clusters = go.Figure()
for i in range(3):
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
for cluster_id in range(3):
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

logging.info(f"Machine Learning Analysis completed. Identified 3 distinct clusters of applicants.")

# Store ML plots in the plots_html dictionary
plots_html['cluster_plot'] = fig_clusters.to_html(full_html=False, include_plotlyjs=False)
plots_html['cluster_summary'] = cluster_summary_text

# Color scheme
primary_color = "#4E79A7"
secondary_color = "#F28E2B"
accent_colors = ["#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]

# Dictionary to store plot HTML
plots_html['title_section'] = f"""
<div class="title-container">
    <div class="dashboard-title">PhD Applications Analysis</div>
    <div class="dashboard-subtitle">Interactive Exploration of Computer Science PhD Applicant Data</div>
    
    <div class="stats-highlights">
        <div class="stat-item">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Total Applications</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{df['Citizenship Country'].nunique()}</div>
            <div class="stat-label">Countries</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{df['GPA'].mean():.2f}</div>
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
logging.info("Generating Country Distribution plot...")
try:
    country_counts = df['Citizenship Country'].value_counts()
    if not country_counts.empty:
        fig = px.bar(
            country_counts,
            x=country_counts.index,
            y=country_counts.values,
            title='Distribution of Applicants by Country',
            labels={'x': 'Country', 'y': 'Number of Applicants'}
        )
fig.update_layout(
    title_font=dict(size=20, color='#F28E2B')
)
        fig.update_traces(hovertemplate='Country: %{x}<br>Applicants: %{y}<extra></extra>')
        plots_html['country_distribution'] = fig.to_html(full_html=False, include_plotlyjs=False)
        logging.info("Country Distribution plot generated successfully.")
    else:
        plots_html['country_distribution'] = "<p>No data available for country distribution.</p>"
        logging.warning("No data found for country distribution plot.")
except Exception as e:
    plots_html['country_distribution'] = f"<p>Error generating country distribution plot: {e}</p>"
    logging.error(f"Error generating country distribution plot: {e}", exc_info=True)

# 1.2 Student Type Analysis
student_type_counts = df['Student Type'].value_counts().reset_index()
student_type_counts.columns = ['Student Type', 'Count']
fig = px.bar(student_type_counts, x='Student Type', y='Count', 
             title='Distribution of Applicants by Student Type',
             labels={'x': 'Student Type', 'y': 'Number of Applicants'},
             color='Count', color_continuous_scale='Viridis')
fig.update_layout(
    title_font=dict(size=20, color='#F28E2B'),
    hoverlabel=dict(font=dict(size=12, family="Roboto"), bgcolor='rgba(0,0,0,0.8)'),
    margin=dict(l=40, r=40, t=60, b=40),
    dragmode='zoom',
    plot_bgcolor='rgba(25,25,25,1)',
    paper_bgcolor='rgba(25,25,25,0)',
    coloraxis_colorbar=dict(title="Number of Applicants")
)
fig.update_traces(hovertemplate='<b>%{x}</b><br>Applications: %{y}<extra></extra>')
plots_html['student_type_distribution'] = fig.to_html(full_html=False, include_plotlyjs=False)

# 1.3 Applicant Type Analysis
applicant_type_counts = df['Applicant Type'].value_counts().reset_index()
applicant_type_counts.columns = ['Applicant Type', 'Count']
fig = px.bar(applicant_type_counts, x='Applicant Type', y='Count', 
             title='Distribution of Applicants by Applicant Type',
             labels={'Count': 'Number of Applicants'},
             color='Count',
             color_continuous_scale=px.colors.sequential.Inferno,
             template='plotly_dark')
fig.update_layout(
    dragmode='zoom',
    hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
    plot_bgcolor='rgba(25,25,25,1)',
    paper_bgcolor='rgba(25,25,25,0)',
    margin=dict(l=40, r=40, t=60, b=40),
    title_font=dict(size=20, color='#F28E2B')
)
fig.update_traces(hovertemplate='<b>%{x}</b><br>Applications: %{y}<extra></extra>')
plots_html['applicant_type_distribution'] = fig.to_html(full_html=False, include_plotlyjs=False)

# 1.4 English Proficiency Test Types
test_type_counts = df['Test Type'].value_counts().reset_index()
test_type_counts.columns = ['Test Type', 'Count']
test_type_counts = test_type_counts.dropna(subset=['Test Type'])
fig = px.bar(test_type_counts, x='Test Type', y='Count', 
             title='Distribution of English Proficiency Test Types',
             labels={'Count': 'Number of Applicants'},
             color='Count',
             color_continuous_scale=px.colors.sequential.Turbo,
             template='plotly_dark')
fig.update_layout(
    dragmode='zoom',
    hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
    plot_bgcolor='rgba(25,25,25,1)',
    paper_bgcolor='rgba(25,25,25,0)',
    margin=dict(l=40, r=40, t=60, b=40),
    title_font=dict(size=20, color='#F28E2B')
)
fig.update_traces(hovertemplate='<b>%{x}</b><br>Applications: %{y}<extra></extra>')
plots_html['english_test_distribution'] = fig.to_html(full_html=False, include_plotlyjs=False)

# 1.5 GPA Distribution
gpa_data = df['GPA'].dropna()
fig = px.histogram(gpa_data, x='GPA', nbins=20, 
                   title='Distribution of First Institution GPA',
                   labels={'GPA': 'GPA', 'count': 'Number of Applicants'},
                   color_discrete_sequence=['#4E79A7'],
                   opacity=0.8,
                   template='plotly_dark')
fig.add_vline(x=gpa_data.mean(), line_dash="dash", line_color="#F28E2B", 
              annotation_text=f"Mean: {gpa_data.mean():.2f}",
              annotation_position="top right",
              annotation_font_color="#F28E2B",
              annotation_font_size=14)
fig.add_vline(x=gpa_data.median(), line_dash="dash", line_color="#59A14F", 
              annotation_text=f"Median: {gpa_data.median():.2f}",
              annotation_position="top left",
              annotation_font_color="#59A14F",
              annotation_font_size=14)
# Add a KDE curve
kde_x = np.linspace(gpa_data.min(), gpa_data.max(), 100)
kde_y = gpa_data.plot.kde().get_lines()[0].get_ydata()[:100]
kde_y_scaled = kde_y * (len(gpa_data) * (gpa_data.max() - gpa_data.min()) / 20) / kde_y.max()  # Scale to match histogram height
fig.add_trace(go.Scatter(x=kde_x, y=kde_y_scaled, mode='lines', line=dict(color='#E15759', width=3), name='Density'))

fig.update_layout(
    dragmode='zoom',
    hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
    plot_bgcolor='rgba(25,25,25,1)',
    paper_bgcolor='rgba(25,25,25,0)',
    margin=dict(l=40, r=40, t=60, b=40),
    title_font=dict(size=20, color='#F28E2B'),
    bargap=0.1,
    hovermode='closest'
)
fig.update_traces(hovertemplate='GPA: %{x:.2f}<br>Count: %{y}<extra></extra>')
plots_html['gpa_distribution'] = fig.to_html(full_html=False, include_plotlyjs=False)

# 1.6 Subject Area Analysis
subject_counts = df['Subject Area'].value_counts().reset_index().head(15)
subject_counts.columns = ['Subject Area', 'Count']
subject_counts = subject_counts.dropna(subset=['Subject Area'])
fig = px.bar(subject_counts.sort_values('Count', ascending=True), 
             y='Subject Area', x='Count', orientation='h',
             title='Top 15 Subject Areas of Applicants',
             labels={'Count': 'Number of Applicants'},
             color='Count',
             color_continuous_scale='viridis',
             template='plotly_dark')
fig.update_layout(
    dragmode='zoom',
    hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
    plot_bgcolor='rgba(25,25,25,1)',
    paper_bgcolor='rgba(25,25,25,0)',
    margin=dict(l=40, r=40, t=60, b=40),
    title_font=dict(size=20, color='#F28E2B'),
    yaxis={'categoryorder':'total ascending'}
)
fig.update_traces(hovertemplate='<b>%{y}</b><br>Applications: %{x}<extra></extra>')
plots_html['subject_area_distribution'] = fig.to_html(full_html=False, include_plotlyjs=False)

# 1.7 Degree Type Analysis
degree_counts = df['Degree'].value_counts().reset_index()
degree_counts.columns = ['Degree', 'Count']
degree_counts = degree_counts.dropna(subset=['Degree'])
fig = px.bar(degree_counts, x='Degree', y='Count', 
             title='Distribution of Degree Types',
             labels={'Count': 'Number of Applicants'},
             color='Count',
             color_continuous_scale=px.colors.sequential.Plasma,
             template='plotly_dark')
fig.update_layout(
    xaxis_tickangle=-45,
    dragmode='zoom',
    hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
    plot_bgcolor='rgba(25,25,25,1)',
    paper_bgcolor='rgba(25,25,25,0)',
    margin=dict(l=40, r=40, t=60, b=40),
    title_font=dict(size=20, color='#F28E2B')
)
fig.update_traces(hovertemplate='<b>%{x}</b><br>Applications: %{y}<extra></extra>')
plots_html['degree_type_distribution'] = fig.to_html(full_html=False, include_plotlyjs=False)

# 1.8 GRE Scores (where available)
# Verbal Scores
verbal_data = df['Verbal'].dropna()
if len(verbal_data) > 0:
    fig = px.histogram(verbal_data, x='Verbal', nbins=15, 
                       title='Distribution of GRE Verbal Scores',
                       labels={'Verbal': 'Verbal Score', 'count': 'Number of Applicants'},
                       color_discrete_sequence=['#4E79A7'],
                       opacity=0.8,
                       template='plotly_dark')
    fig.add_vline(x=verbal_data.mean(), line_dash="dash", line_color="#F28E2B", 
                  annotation_text=f"Mean: {verbal_data.mean():.2f}",
                  annotation_position="top right",
                  annotation_font_color="#F28E2B",
                  annotation_font_size=14)
    fig.add_vline(x=verbal_data.median(), line_dash="dash", line_color="#59A14F", 
                  annotation_text=f"Median: {verbal_data.median():.2f}",
                  annotation_position="top left",
                  annotation_font_color="#59A14F",
                  annotation_font_size=14)
    
    # Add a KDE curve if we have enough data
    if len(verbal_data) > 5:
        kde_x = np.linspace(verbal_data.min(), verbal_data.max(), 100)
        kde_y = verbal_data.plot.kde().get_lines()[0].get_ydata()[:100]
        kde_y_scaled = kde_y * (len(verbal_data) * (verbal_data.max() - verbal_data.min()) / 15) / kde_y.max()
        fig.add_trace(go.Scatter(x=kde_x, y=kde_y_scaled, mode='lines', line=dict(color='#E15759', width=3), name='Density'))
    
    fig.update_layout(
        dragmode='zoom',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color='#F28E2B'),
        bargap=0.1,
        hovermode='closest'
    )
    fig.update_traces(hovertemplate='Verbal Score: %{x:.0f}<br>Count: %{y}<extra></extra>')
    plots_html['gre_verbal_distribution'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['gre_verbal_distribution'] = "<p>No GRE Verbal score data available.</p>"

# Quantitative Scores
quant_data = df['Quantitative'].dropna()
if len(quant_data) > 0:
    fig = px.histogram(quant_data, x='Quantitative', nbins=15, 
                       title='Distribution of GRE Quantitative Scores',
                       labels={'Quantitative': 'Quantitative Score', 'count': 'Number of Applicants'},
                       color_discrete_sequence=['#76B7B2'],
                       opacity=0.8,
                       template='plotly_dark')
    fig.add_vline(x=quant_data.mean(), line_dash="dash", line_color="#F28E2B", 
                  annotation_text=f"Mean: {quant_data.mean():.2f}",
                  annotation_position="top right",
                  annotation_font_color="#F28E2B",
                  annotation_font_size=14)
    fig.add_vline(x=quant_data.median(), line_dash="dash", line_color="#59A14F", 
                  annotation_text=f"Median: {quant_data.median():.2f}",
                  annotation_position="top left",
                  annotation_font_color="#59A14F",
                  annotation_font_size=14)
    
    # Add a KDE curve if we have enough data
    if len(quant_data) > 5:
        kde_x = np.linspace(quant_data.min(), quant_data.max(), 100)
        kde_y = quant_data.plot.kde().get_lines()[0].get_ydata()[:100]
        kde_y_scaled = kde_y * (len(quant_data) * (quant_data.max() - quant_data.min()) / 15) / kde_y.max()
        fig.add_trace(go.Scatter(x=kde_x, y=kde_y_scaled, mode='lines', line=dict(color='#E15759', width=3), name='Density'))
    
    fig.update_layout(
        dragmode='zoom',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color='#F28E2B'),
        bargap=0.1,
        hovermode='closest'
    )
    fig.update_traces(hovertemplate='Quantitative Score: %{x:.0f}<br>Count: %{y}<extra></extra>')
    plots_html['gre_quant_distribution'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['gre_quant_distribution'] = "<p>No GRE Quantitative score data available.</p>"

# 2. RELATIONSHIP ANALYSIS
# ------------------------

# 2.1 GPA vs English Proficiency Score (focusing on IELTS)
ielts_df = df[df['Test Type'] == 'IELTS'].dropna(subset=['GPA', 'Score'])
if len(ielts_df) > 1:
    fig = px.scatter(ielts_df, x='GPA', y='Score', 
                     title='Relationship Between GPA and IELTS Score',
                     labels={'Score': 'IELTS Score', 'GPA': 'GPA'},
                     color='GPA',
                     color_continuous_scale='viridis',
                     size='Score',
                     size_max=15,
                     template='plotly_dark',
                     trendline="ols")
    fig.update_layout(
        dragmode='zoom',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color='#F28E2B'),
        hovermode='closest'
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate='GPA: %{x:.2f}<br>IELTS Score: %{y:.1f}<extra></extra>'
    )
    plots_html['gpa_vs_ielts'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['gpa_vs_ielts'] = "<p>Not enough data to plot GPA vs IELTS.</p>"

# 2.2 GPA vs GRE Scores (where available)
# GPA vs Verbal
gre_df_verbal = df.dropna(subset=['GPA', 'Verbal'])
if len(gre_df_verbal) > 1:
    fig = px.scatter(gre_df_verbal, x='GPA', y='Verbal', 
                     title='Relationship Between GPA and GRE Verbal Score',
                     labels={'Verbal': 'GRE Verbal Score', 'GPA': 'GPA'},
                     color='Verbal',
                     color_continuous_scale='plasma',
                     size='GPA',
                     size_max=15,
                     template='plotly_dark',
                     trendline="ols")
    fig.update_layout(
        dragmode='zoom',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color='#F28E2B'),
        hovermode='closest'
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate='GPA: %{x:.2f}<br>Verbal Score: %{y:.0f}<extra></extra>'
    )
    plots_html['gpa_vs_gre_verbal'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['gpa_vs_gre_verbal'] = "<p>Not enough data to plot GPA vs GRE Verbal.</p>"

# GPA vs Quantitative
gre_df_quant = df.dropna(subset=['GPA', 'Quantitative'])
if len(gre_df_quant) > 1:
    fig = px.scatter(gre_df_quant, x='GPA', y='Quantitative', 
                     title='Relationship Between GPA and GRE Quantitative Score',
                     labels={'Quantitative': 'GRE Quantitative Score', 'GPA': 'GPA'},
                     color='Quantitative',
                     color_continuous_scale='inferno',
                     size='GPA',
                     size_max=15,
                     template='plotly_dark',
                     trendline="ols")
    fig.update_layout(
        dragmode='zoom',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color='#F28E2B'),
        hovermode='closest'
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate='GPA: %{x:.2f}<br>Quantitative Score: %{y:.0f}<extra></extra>'
    )
    plots_html['gpa_vs_gre_quant'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['gpa_vs_gre_quant'] = "<p>Not enough data to plot GPA vs GRE Quantitative.</p>"

# 2.3 GRE Verbal vs Quantitative Scores
gre_complete_df = df.dropna(subset=['Verbal', 'Quantitative'])
if len(gre_complete_df) > 1:
    fig = px.scatter(gre_complete_df, x='Verbal', y='Quantitative', 
                     title='Relationship Between GRE Verbal and Quantitative Scores',
                     labels={'Verbal': 'GRE Verbal Score', 'Quantitative': 'GRE Quantitative Score'},
                     color='GPA',
                     color_continuous_scale='turbo',
                     size=np.ones(len(gre_complete_df))*10,  # Uniform size for better visibility
                     template='plotly_dark',
                     trendline="ols")
    fig.update_layout(
        dragmode='zoom',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color='#F28E2B'),
        hovermode='closest'
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate='Verbal: %{x:.0f}<br>Quantitative: %{y:.0f}<br>GPA: %{marker.color:.2f}<extra></extra>'
    )
    plots_html['gre_verbal_vs_quant'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['gre_verbal_vs_quant'] = "<p>Not enough data to plot GRE Verbal vs Quantitative.</p>"

# 2.4 Country Comparison of GPAs (top 5 countries)
top_countries = df['Citizenship Country'].value_counts().head(5).index
country_gpa_df = df[df['Citizenship Country'].isin(top_countries)].dropna(subset=['GPA'])
if len(country_gpa_df) > 0:
    fig = px.box(country_gpa_df, x='Citizenship Country', y='GPA', 
                 title='Comparison of GPA Distributions by Country (Top 5 Countries)',
                 labels={'GPA': 'GPA', 'Citizenship Country': 'Country'},
                 color='Citizenship Country',
                 color_discrete_sequence=px.colors.qualitative.Bold,
                 points="all",
                 notched=True,
                 template='plotly_dark')
    fig.update_layout(
        xaxis_tickangle=-45,
        dragmode='zoom',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color='#F28E2B'),
        showlegend=False,
        hovermode='closest'
    )
    # Update the individual point traces
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    plots_html['country_gpa_comparison'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['country_gpa_comparison'] = "<p>Not enough data to compare GPAs by country.</p>"

# 2.5 Country Comparison of English Proficiency (IELTS)
ielts_country_df = df[(df['Test Type'] == 'IELTS') & 
                       (df['Citizenship Country'].isin(top_countries))].dropna(subset=['Score'])
if len(ielts_country_df) > 0:
    fig = px.box(ielts_country_df, x='Citizenship Country', y='Score', 
                 title='Comparison of IELTS Scores by Country (Top Countries)',
                 labels={'Score': 'IELTS Score', 'Citizenship Country': 'Country'},
                 color='Citizenship Country',
                 color_discrete_sequence=px.colors.qualitative.Safe,
                 points="all",
                 notched=True,
                 template='plotly_dark')
    fig.update_layout(
        xaxis_tickangle=-45,
        dragmode='zoom',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color='#F28E2B'),
        showlegend=False,
        hovermode='closest'
    )
    # Update the individual point traces
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    plots_html['country_ielts_comparison'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['country_ielts_comparison'] = "<p>Not enough data to compare IELTS scores by country.</p>"

# 3. SUMMARY STATISTICS
# ---------------------

# Calculate overall summary statistics
numeric_cols = ['GPA', 'GPA 2', 'Verbal', 'Quantitative', 'Score']
summary_stats = pd.DataFrame({
    'Count': df['GPA'].count(),
    'Mean': df['GPA'].dropna().mean(),
    'Median': df['GPA'].dropna().median(),
    'Std Dev': df['GPA'].dropna().std(),
    'Min': df['GPA'].dropna().min(),
    'Max': df['GPA'].dropna().max()
}, index=['GPA'])

# For English proficiency tests
for test in df['Test Type'].dropna().unique():
    test_scores = df[df['Test Type'] == test]['Score'].dropna()
    if len(test_scores) > 0:
        summary_stats.loc[f'{test} Score'] = [
            test_scores.count(),
            test_scores.mean(),
            test_scores.median(),
            test_scores.std(),
            test_scores.min(),
            test_scores.max()
        ]

# For GRE scores
for gre_type in ['Verbal', 'Quantitative']:
    scores = df[gre_type].dropna()
    if len(scores) > 0:
        summary_stats.loc[f'GRE {gre_type}'] = [
            scores.count(),
            scores.mean(),
            scores.median(),
            scores.std(),
            scores.min(),
            scores.max()
        ]

# Save summary statistics
summary_stats.to_csv('summary_statistics.csv')
logging.info("Analysis completed. Summary statistics saved.")

# Advanced Visualizations Section
# ------------------------------

# Create a Sunburst chart of subject areas by country
subject_country_df = df.dropna(subset=['Subject Area', 'Citizenship Country'])
if not subject_country_df.empty:
    # Count occurrences
    subject_country_counts = subject_country_df.groupby(['Citizenship Country', 'Subject Area']).size().reset_index(name='Count')
    # Keep only top 7 countries and top 10 subject areas for readability
    top_countries = subject_country_df['Citizenship Country'].value_counts().nlargest(7).index
    top_subjects = subject_country_df['Subject Area'].value_counts().nlargest(10).index
    filtered_counts = subject_country_counts[
        subject_country_counts['Citizenship Country'].isin(top_countries) & 
        subject_country_counts['Subject Area'].isin(top_subjects)
    ]
    
    # Create sunburst chart
    fig = px.sunburst(
        filtered_counts, 
        path=['Citizenship Country', 'Subject Area'], 
        values='Count',
        title='Distribution of Subject Areas by Country',
        color_discrete_sequence=px.colors.qualitative.Bold,
        template='plotly_dark'
    )
    fig.update_layout(
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        margin=dict(l=0, r=0, t=50, b=0),
        title_font=dict(size=20, color='#F28E2B'),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
    )
    plots_html['subject_by_country_sunburst'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['subject_by_country_sunburst'] = "<p>Not enough data to create the sunburst visualization.</p>"

# Create a World Map of applicants by country
country_counts = df['Citizenship Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Applicants']
if not country_counts.empty:
    # Fix any country name inconsistencies that might affect mapping
    country_name_fixes = {
        'Viet Nam': 'Vietnam',
        'Vietnam': 'Vietnam',
        'USA': 'United States',
        'United States': 'United States',
        'UK': 'United Kingdom',
        'United Kingdom': 'United Kingdom'
    }
    country_counts['Country'] = country_counts['Country'].replace(country_name_fixes)
    
    # Create choropleth map
    fig = px.choropleth(
        country_counts,
        locations='Country',
        locationmode='country names',
        color='Applicants',
        hover_name='Country',
        color_continuous_scale='plasma',
        title='Global Distribution of PhD Applicants',
        template='plotly_dark'
    )
    fig.update_layout(
        geo=dict(
            showcoastlines=True,
            coastlinecolor="White",
            showland=True,
            landcolor="rgb(30, 30, 30)",
            showocean=True,
            oceancolor="rgb(20, 20, 30)",
            showlakes=False,
            showrivers=False,
            projection_type='natural earth'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        title_font=dict(size=20, color='#F28E2B'),
        coloraxis_colorbar=dict(
            title="Applicants",
            tickfont=dict(color='white')
        ),
        paper_bgcolor='rgba(25,25,25,0)',
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
    )
    plots_html['world_map'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['world_map'] = "<p>Not enough data to create the world map visualization.</p>"

# Create a 3D Scatter plot for GPA, GRE Verbal, and GRE Quantitative
gre_3d_df = df.dropna(subset=['GPA', 'Verbal', 'Quantitative'])
if len(gre_3d_df) > 1:
    fig = px.scatter_3d(
        gre_3d_df,
        x='GPA',
        y='Verbal',
        z='Quantitative',
        color='GPA',
        color_continuous_scale='viridis',
        opacity=0.7,
        title='3D Relationship Between GPA, GRE Verbal, and GRE Quantitative Scores',
        labels={'GPA': 'GPA', 'Verbal': 'GRE Verbal', 'Quantitative': 'GRE Quantitative'},
        template='plotly_dark'
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='GPA',
            yaxis_title='GRE Verbal',
            zaxis_title='GRE Quantitative',
            xaxis=dict(gridcolor='gray', showbackground=False),
            yaxis=dict(gridcolor='gray', showbackground=False),
            zaxis=dict(gridcolor='gray', showbackground=False),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        title_font=dict(size=20, color='#F28E2B'),
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=12, font_family="Roboto"),
        paper_bgcolor='rgba(25,25,25,0)',
    )
    plots_html['gre_3d_scatter'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['gre_3d_scatter'] = "<p>Not enough data to create the 3D visualization.</p>"

# Create a radar chart comparing different metrics for top countries
top5_countries = df['Citizenship Country'].value_counts().nlargest(5).index
if len(top5_countries) > 0:
    radar_data = []
    
    for country in top5_countries:
        country_df = df[df['Citizenship Country'] == country]
        # Calculate metrics (handling NaNs appropriately)
        avg_gpa = country_df['GPA'].mean() if not country_df['GPA'].isna().all() else None
        avg_gpa2 = country_df['GPA 2'].mean() if not country_df['GPA 2'].isna().all() else None
        
        # For English scores, need to handle different test types
        ielts_scores = country_df[country_df['Test Type'] == 'IELTS']['Score']
        avg_ielts = ielts_scores.mean() if not ielts_scores.empty else None
        
        # Scale all metrics to 0-10 range for comparison
        metrics = {
            'GPA': avg_gpa/4*10 if avg_gpa is not None else 0,
            'GPA 2': avg_gpa2/4*10 if avg_gpa2 is not None else 0,
            'IELTS': avg_ielts if avg_ielts is not None else 0,
            'Applications': len(country_df)/max(df['Citizenship Country'].value_counts())*10
        }
        
        # Create radar chart trace
        radar_data.append(
            go.Scatterpolar(
                r=[metrics['GPA'], metrics['GPA 2'], metrics['IELTS'], metrics['Applications']],
                theta=['GPA', 'GPA 2', 'IELTS', 'Application Volume'],
                fill='toself',
                name=country
            )
        )
    
    # Create radar chart
    fig = go.Figure(data=radar_data)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        title='Country Comparison: Academic Metrics (Scaled)',
        template='plotly_dark',
        title_font=dict(size=20, color='#F28E2B'),
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(50,50,50,0.2)'
        ),
        paper_bgcolor='rgba(25,25,25,0)',
        margin=dict(l=40, r=40, t=50, b=40),
    )
    plots_html['country_radar'] = fig.to_html(full_html=False, include_plotlyjs=False)
else:
    plots_html['country_radar'] = "<p>Not enough data to create the radar chart.</p>"

# Parallel Coordinates Plot (Added)
logging.info("Generating Parallel Coordinates Plot...")
parallel_cols = ['GPA', 'Verbal', 'Quantitative']
df_parallel = df[parallel_cols].dropna()

if not df_parallel.empty:
    fig_parallel = go.Figure(data=
        go.Parcoords(
            line = dict(color = df_parallel['GPA'],
                       colorscale = 'Viridis', #Plasma, Viridis, Cividis
                       showscale = True,
                       cmin = df_parallel['GPA'].min(),
                       cmax = df_parallel['GPA'].max()),
            dimensions = list([
                dict(range = [df_parallel['GPA'].min(), df_parallel['GPA'].max()],
                     label = 'GPA', values = df_parallel['GPA']),
                dict(range = [df_parallel['Verbal'].min(), df_parallel['Verbal'].max()],
                     label = 'GRE Verbal', values = df_parallel['Verbal']),
                dict(range = [df_parallel['Quantitative'].min(), df_parallel['Quantitative'].max()],
                     label = 'GRE Quantitative', values = df_parallel['Quantitative'])
            ])
        )
    )

    fig_parallel.update_layout(
        title={
            'text': "Parallel Coordinates of GPA and GRE Scores",
            'font': {'size': 20, 'color': '#F28E2B'}
        },
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        font_color='white',
        margin=dict(l=40, r=40, t=80, b=40),
         hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=12,
            font_family="Roboto"
        )
    )
    plots_html['parallel_coordinates'] = fig_parallel.to_html(full_html=False, include_plotlyjs='cdn')
    logging.info("Parallel Coordinates Plot generated.")
else:
    logging.warning("Not enough data to generate Parallel Coordinates plot after dropping NaNs.")
    plots_html['parallel_coordinates'] = f"""
<div class="plot-container">
    <h3>Parallel Coordinates Plot: GPA vs. GRE Scores</h3>
    <p class="description">Could not generate Parallel Coordinates plot due to insufficient data (applicants missing GPA or GRE scores).</p>
</div>
"""

# Create a Heatmap to show relationships between Countries and Subject Areas
logging.info("Generating Country-Subject Heatmap...")

# Get top 5 countries and top 5 subject areas for readability
top_countries = df['Citizenship Country'].value_counts().head(5).index.tolist()
top_subjects = df['Subject Area'].value_counts().head(5).index.tolist()

# Remove NaN values
heatmap_df = df.dropna(subset=['Citizenship Country', 'Subject Area'])
heatmap_df = heatmap_df[heatmap_df['Citizenship Country'].isin(top_countries) & heatmap_df['Subject Area'].isin(top_subjects)]

if not heatmap_df.empty and len(heatmap_df) > 5:
    # Create a cross-tabulation of countries and subject areas
    heatmap_data = pd.crosstab(heatmap_df['Citizenship Country'], heatmap_df['Subject Area'])
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        colorbar=dict(title="Applicants"),
        hovertemplate='Country: %{y}<br>Subject: %{x}<br>Applicants: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Heatmap: Relationship Between Countries and Subject Areas",
        xaxis_title="Subject Area",
        yaxis_title="Country",
        font=dict(size=12),
        autosize=True,
        margin=dict(l=60, r=20, t=60, b=70),
        plot_bgcolor='rgba(25,25,25,1)',
        paper_bgcolor='rgba(25,25,25,0)',
        title_font=dict(size=20, color='#F28E2B'),
        font_color='white',
        xaxis={'tickangle': -45}
    )
    
    plots_html['country_subject_heatmap'] = fig.to_html(full_html=False, include_plotlyjs=False)
    logging.info("Country-Subject Heatmap generated successfully.")
else:
    plots_html['country_subject_heatmap'] = "<p>Not enough data to create a heatmap between countries and subject areas.</p>"
    logging.warning("Not enough data to generate Country-Subject Heatmap.")

# Define cluster_summary (ensure it's defined before using)
cluster_summary_text = plots_html.get('cluster_summary', "<p>Cluster analysis summary not available.</p>")

# 5. HTML DASHBOARD GENERATION
# ---------------------------
logging.info("Assembling the interactive HTML dashboard...")

# Define the complete HTML structure with unique string placeholders
html_template_structure = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhD Applications Analysis Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
    <style>
        /* Root Variables */
        :root {{
            --bg-color: #121212;
            --text-color: #fff;
            --text-muted-color: #aaa;
            --primary-color: #4E79A7;
            --secondary-color: #F28E2B;
            --accent-color: #59A14F;
            --light-bg-color: #f5f5f5;
            --light-text-color: #333;
            --light-text-muted-color: #666;
            --primary-gradient: linear-gradient(135deg, #4E79A7 0%, #3a5e85 100%);
            --secondary-gradient: linear-gradient(135deg, #F28E2B 0%, #e0741a 100%);
            --accent-gradient: linear-gradient(135deg, #59A14F 0%, #488240 100%);
            --dark-gradient: linear-gradient(135deg, #1e1e1e 0%, #121212 100%);
            --glass-bg: rgba(255,255,255,0.07);
            --glass-border: rgba(255,255,255,0.1);
            --card-shadow: 0 10px 20px rgba(0,0,0,0.1), 0 6px 6px rgba(0,0,0,0.1);
            --card-hover-shadow: 0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22);
            --transition-speed: 0.3s;
        }}

        /* Base Styles */
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Roboto', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            transition: background-color var(--transition-speed), color var(--transition-speed);
        }}

        /* Navigation */
        .top-nav {{
            background: var(--dark-gradient);
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 40px;
            position: fixed;
            width: 100%;
            top: 0;
            z-index: 1000;
            box-shadow: var(--card-shadow);
            box-sizing: border-box;
        }}

        .nav-left {{
            display: flex;
            align-items: center;
            gap: 30px;
        }}

        .nav-logo {{
            font-size: 1.7rem;
            color: var(--primary-color);
            font-weight: 700;
        }}

        .nav-links {{
            display: flex;
            gap: 15px;
        }}

        .nav-link {{
            color: var(--text-muted-color);
            text-decoration: none;
            font-weight: 500;
            padding: 8px 15px;
            border-radius: 8px;
            transition: all var(--transition-speed);
        }}

        .nav-link:hover {{
            background: var(--glass-bg);
            color: var(--text-color);
        }}

        .theme-toggle {{
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            color: var(--text-color);
            padding: 8px 15px;
            border-radius: 8px;
            cursor: pointer;
            transition: all var(--transition-speed);
        }}

        /* Main Content */
        .main-content {{
            margin-top: 80px;
            padding: 30px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }}

        /* Title Section */
        .title-container {{
            background: var(--dark-gradient);
            padding: 30px;
            border-radius: 16px;
            margin-bottom: 40px;
            box-shadow: var(--card-shadow);
        }}

        .dashboard-title {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-color);
            text-align: center;
            margin-bottom: 8px;
        }}

        .dashboard-subtitle {{
            font-size: 1.2rem;
            color: var(--text-muted-color);
            text-align: center;
            margin-bottom: 30px;
        }}

        /* Stats Highlights */
        .stats-highlights {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
            gap: 20px;
            padding: 20px;
            background: rgba(30, 30, 30, 0.5);
            border-radius: 10px;
            box-shadow: var(--card-shadow);
        }}
        
        .stat-item {{
            background: rgba(40, 40, 40, 0.5);
            padding: 20px 30px;
            border-radius: 8px;
            text-align: center;
            min-width: 160px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }}
        
        .stat-item:hover {{
            transform: translateY(-2px);
            box-shadow: var(--card-hover-shadow);
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--secondary-color);
            margin-bottom: 10px;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
            line-height: 1;
        }}
        
        .stat-label {{
            color: var(--text-muted-color);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }}

        /* Sections */
        section {{
            margin-bottom: 40px;
            background: var(--dark-gradient);
            padding: 30px;
            border-radius: 16px;
            box-shadow: var(--card-shadow);
        }}

        h2 {{
            color: var(--primary-color);
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.8rem;
        }}

        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .plot-container {{
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            padding: 20px;
        }}

        .plot-container h3 {{
            color: var(--secondary-color);
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 1.2rem;
        }}

        .description {{
            color: var(--text-muted-color);
            font-size: 0.9rem;
            margin-bottom: 15px;
        }}

        /* Light Theme */
        body.light-mode {{
            background-color: var(--light-bg-color);
            color: var(--light-text-color);
        }}

        body.light-mode .top-nav {{
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
        }}

        body.light-mode .title-container,
        body.light-mode section {{
            background: white;
        }}

        body.light-mode .stats-highlights {{
            background: rgba(248, 249, 250, 0.9);
        }}

        body.light-mode .stat-item {{
            background: white;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }}

        body.light-mode .plot-container {{
            background: white;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }}

        body.light-mode .nav-link {{
            color: var(--light-text-muted-color);
        }}

        body.light-mode .nav-link:hover {{
            background: rgba(0, 0, 0, 0.05);
            color: var(--light-text-color);
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            .top-nav {{
                padding: 15px 20px;
            }}

            .nav-links {{
                display: none;
            }}

            .main-content {{
                padding: 15px;
            }}

            .dashboard-title {{
                font-size: 2rem;
            }}

            .grid-container {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <nav class="top-nav">
        <div class="nav-left">
            <div class="nav-logo">PhD Dashboard</div>
            <div class="nav-links">
                <a href="#demographics" class="nav-link">Demographics</a>
                <a href="#academics" class="nav-link">Academics</a>
                <a href="#test-scores" class="nav-link">Test Scores</a>
                <a href="#relationships" class="nav-link">Relationships</a>
                <a href="#country-comparison" class="nav-link">Country Comparison</a>
                <a href="#ml-analysis" class="nav-link">ML Analysis</a>
            </div>
        </div>
        <div class="nav-right">
            <button class="theme-toggle">
                <i class="fas fa-sun"></i> Light Mode
            </button>
        </div>
    </nav>

    <main class="main-content">
        <div class="title-container">
            <div class="dashboard-title">PhD Applications Analysis</div>
            <div class="dashboard-subtitle">Interactive Exploration of Computer Science PhD Applicant Data</div>
            
            <div class="stats-highlights">
                <div class="stat-item">
                    <div class="stat-value">{len(df)}</div>
                    <div class="stat-label">Total Applications</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{df['Citizenship Country'].nunique()}</div>
                    <div class="stat-label">Countries</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{df['GPA'].mean():.2f}</div>
                    <div class="stat-label">Avg GPA</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{df['Test Type'].value_counts().index[0]}</div>
                    <div class="stat-label">Top Test</div>
                </div>
            </div>
        </div>

        <section id="demographics">
            <h2>Applicant Demographics</h2>
            <div class="grid-container">
                <div class="plot-container">
                    <h3>Distribution by Country</h3>
                    <p class="description">Number of applicants from different countries.</p>
                    <!--PLOT_COUNTRY_DISTRIBUTION-->
                </div>
                <div class="plot-container">
                    <h3>Distribution by Student Type</h3>
                    <p class="description">Breakdown of applicants by student type (e.g., New First Time, Transfer).</p>
                    <!--PLOT_STUDENT_TYPE_DISTRIBUTION-->
                </div>
                <div class="plot-container">
                    <h3>Distribution by Applicant Type</h3>
                    <p class="description">Breakdown by applicant type (e.g., Regular, Deferral).</p>
                    <!--PLOT_APPLICANT_TYPE_DISTRIBUTION-->
                </div>
            </div>
        </section>

        <section id="academics">
            <h2>Academic Qualifications</h2>
            <div class="grid-container">
                <div class="plot-container">
                    <h3>GPA Distribution</h3>
                    <p class="description">Distribution of Grade Point Averages (GPA) among applicants.</p>
                    <!--PLOT_GPA_DISTRIBUTION-->
                </div>
                <div class="plot-container">
                    <h3>Top 15 Subject Areas</h3>
                    <p class="description">Most common undergraduate subject areas.</p>
                    <!--PLOT_SUBJECT_AREA_DISTRIBUTION-->
                </div>
                <div class="plot-container">
                    <h3>Distribution of Degree Types</h3>
                    <p class="description">Types of degrees held by applicants (e.g., BSc, MSc).</p>
                    <!--PLOT_DEGREE_TYPE_DISTRIBUTION-->
                </div>
            </div>
        </section>

        <section id="test-scores">
            <h2>Test Scores</h2>
            <div class="grid-container">
                <div class="plot-container">
                    <h3>English Proficiency Test Distribution</h3>
                    <p class="description">Distribution of scores for common English tests (IELTS, TOEFL, Duolingo).</p>
                    <!--PLOT_ENGLISH_TEST_DISTRIBUTION-->
                </div>
                <div class="plot-container">
                    <h3>GRE Verbal Score Distribution</h3>
                    <p class="description">Distribution of GRE Verbal scores (where available).</p>
                    <!--PLOT_GRE_VERBAL_DISTRIBUTION-->
                </div>
                <div class="plot-container">
                    <h3>GRE Quantitative Score Distribution</h3>
                    <p class="description">Distribution of GRE Quantitative scores (where available).</p>
                    <!--PLOT_GRE_QUANT_DISTRIBUTION-->
                </div>
            </div>
        </section>

        <section id="relationships">
            <h2>Relationship Analysis</h2>
            <div class="grid-container">
                <div class="plot-container">
                    <h3>GPA vs. Test Scores</h3>
                    <p class="description">Scatter plot showing relationships between GPA and standardized test scores.</p>
                    <!--PLOT_GPA_VS_SCORES-->
                </div>
                <div class="plot-container">
                    <h3>GRE Score Relationships</h3>
                    <p class="description">Relationship between GRE Verbal and Quantitative scores.</p>
                    <!--PLOT_GRE_RELATIONSHIPS-->
                </div>
            </div>
        </section>

        <section id="country-comparison">
            <h2>Country Comparison</h2>
            <div class="grid-container">
                <div class="plot-container">
                    <h3>GPA by Country</h3>
                    <p class="description">Box plots showing GPA distribution across top countries.</p>
                    <!--PLOT_GPA_BY_COUNTRY-->
                </div>
                <div class="plot-container">
                    <h3>Test Scores by Country</h3>
                    <p class="description">Box plots showing test score distributions across top countries.</p>
                    <!--PLOT_SCORES_BY_COUNTRY-->
                </div>
            </div>
        </section>

        <section id="ml-analysis">
            <h2>Machine Learning Analysis</h2>
            <div class="grid-container">
                <div class="plot-container">
                    <h3>Applicant Clusters</h3>
                    <p class="description">PCA visualization of applicant clusters based on academic metrics.</p>
                    <!--PLOT_CLUSTERS-->
                </div>
                <div class="plot-container">
                    <h3>Cluster Summary</h3>
                    <p class="description">Summary statistics and characteristics of each identified cluster.</p>
                    <!--CLUSTER_SUMMARY-->
                </div>
            </div>
        </section>
    </main>

    <script>
        // Theme toggle functionality
        const themeToggle = document.querySelector('.theme-toggle');
        const body = document.body;
        
        themeToggle.addEventListener('click', () => {{
            body.classList.toggle('light-mode');
            const icon = themeToggle.querySelector('i');
            const text = themeToggle.textContent.trim();
            
            if (body.classList.contains('light-mode')) {{
                icon.className = 'fas fa-moon';
                themeToggle.innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
            }} else {{
                icon.className = 'fas fa-sun';
                themeToggle.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
            }}
        }});
    </script>
</body>
</html>
"""

# Start with the template structure
final_html_content = html_template_structure

# Define default plot HTML
default_plot_html = "<p>Plot not available.</p>"

# Define mapping from plot dictionary keys to HTML placeholders
placeholder_map = {
    'PLOT_COUNTRY_DISTRIBUTION': plots_html.get('country_distribution', default_plot_html),
    'PLOT_STUDENT_TYPE_DISTRIBUTION': plots_html.get('student_type_distribution', default_plot_html),
    'PLOT_APPLICANT_TYPE_DISTRIBUTION': plots_html.get('applicant_type_distribution', default_plot_html),
    'PLOT_GPA_DISTRIBUTION': plots_html.get('gpa_distribution', default_plot_html),
    'PLOT_SUBJECT_AREA_DISTRIBUTION': plots_html.get('subject_area_distribution', default_plot_html),
    'PLOT_DEGREE_TYPE_DISTRIBUTION': plots_html.get('degree_type_distribution', default_plot_html),
    'PLOT_ENGLISH_TEST_DISTRIBUTION': plots_html.get('english_test_distribution', default_plot_html),
    'PLOT_GRE_VERBAL_DISTRIBUTION': plots_html.get('gre_verbal_distribution', default_plot_html),
    'PLOT_GRE_QUANT_DISTRIBUTION': plots_html.get('gre_quant_distribution', default_plot_html),
    'PLOT_GPA_VS_SCORES': plots_html.get('gpa_vs_ielts', default_plot_html),  # Updated for relationship section
    'PLOT_GRE_RELATIONSHIPS': plots_html.get('gre_verbal_vs_quant', default_plot_html),  # Updated for relationship section
    'PLOT_GPA_BY_COUNTRY': plots_html.get('country_gpa_comparison', default_plot_html),  # Updated for country comparison
    'PLOT_SCORES_BY_COUNTRY': plots_html.get('country_ielts_comparison', default_plot_html),  # Updated for country comparison
    'PLOT_PARALLEL_COORDINATES': plots_html.get('parallel_coordinates', default_plot_html),
    'PLOT_COUNTRY_SUBJECT_HEATMAP': plots_html.get('country_subject_heatmap', default_plot_html),
    'PLOT_COUNTRY_RADAR': plots_html.get('country_radar', default_plot_html),
    'PLOT_CLUSTER_PLOT': plots_html.get('cluster_plot', default_plot_html),
    'CLUSTER_SUMMARY': plots_html.get('cluster_summary', default_plot_html)
}

# First replace the title section
final_html_content = final_html_content.replace('<!--TITLE_SECTION-->', plots_html['title_section'])

# Then replace all plot placeholders
for key, value in placeholder_map.items():
    placeholder = f'<!--{key}-->'
    final_html_content = final_html_content.replace(placeholder, value)

# Replace cluster summary placeholder
for key, placeholder in placeholder_map.items():
    plot_html = placeholder
    final_html_content = final_html_content.replace(key, plot_html)

# Replace cluster summary placeholder
final_html_content = final_html_content.replace('CLUSTER_SUMMARY', cluster_summary_text)


# Save the complete HTML dashboard
output_dashboard_file = 'phd_applications_dashboard.html'
try:
    with open(output_dashboard_file, 'w', encoding='utf-8') as f:
        f.write(final_html_content)
    logging.info(f"Interactive dashboard with plots saved to '{output_dashboard_file}'")
except Exception as e:
    logging.error(f"Error writing dashboard file: {e}")

logging.info("Analysis script finished.")

def merge_dashboard():
    """
    Merges the machine learning analysis plots into the existing dashboard.
    This function should be called after the original dashboard has been generated.
    """
    try:
        # Check if the dashboard file exists
        if not os.path.exists('phd_applications_dashboard.html'):
            logging.error("Dashboard file not found")
            return False
            
        # Create a backup of the original file
        backup_file = 'phd_applications_dashboard.html.bak'
        shutil.copy2('phd_applications_dashboard.html', backup_file)
        logging.info(f"Created backup at {backup_file}")
        
        # Check if we need to rerun the machine learning analysis
        if 'elbow_curve' not in plots_html or 'cluster_pca' not in plots_html:
            logging.info("Machine learning plots not found in plots_html dictionary, regenerating...")
            
            # Extract features for clustering
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
            
            # Apply K-means with the optimal k
            optimal_k = 3  # Based on elbow method visualization
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to the original dataframe
            df['Cluster'] = clusters
            
            # Perform PCA for visualization
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
            plots_html['elbow_curve'] = fig_elbow.to_html(full_html=False, include_plotlyjs='cdn')
            plots_html['cluster_pca'] = fig_pca.to_html(full_html=False, include_plotlyjs=False)
            plots_html['parallel_clusters'] = fig_parallel_clusters.to_html(full_html=False, include_plotlyjs=False)
            plots_html['radar_clusters'] = fig_radar_clusters.to_html(full_html=False, include_plotlyjs=False)
        
        # Read the HTML file
        with open('phd_applications_dashboard.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Check if the machine learning section already exists
        if 'id="ml-analysis"' in html_content:
            logging.info("ML section already exists in dashboard")
        else:
            # Find where to insert the ML section (before the closing </div> of the container)
            insert_idx = html_content.rfind('</div>')
            if insert_idx == -1:
                logging.error("Could not find position to insert ML section")
                return False
                
            # Create the ML section HTML
            ml_section = f"""
            <div id="ml-analysis" class="section">
                <h2>Advanced Analysis with Machine Learning</h2>
                <div class="grid">
                    <div class="plot-container">
                        <h3>Elbow Method for Optimal Clusters</h3>
                        {plots_html['elbow_curve']}
                    </div>
                    <div class="plot-container">
                        <h3>Applicant Clusters</h3>
                        {plots_html['cluster_pca']}
                    </div>
                    <div class="plot-container">
                        <h3>Parallel Coordinates by Cluster</h3>
                        {plots_html['parallel_clusters']}
                    </div>
                    <div class="plot-container">
                        <h3>Cluster Characteristics</h3>
                        {plots_html['radar_clusters']}
                    </div>
                </div>
            </div>
            """
            
            # Insert the ML section
            new_html = html_content[:insert_idx] + ml_section + html_content[insert_idx:]
            
            # Add a link to the ML section in the nav bar
            nav_links_idx = new_html.find('<div class="nav-links">')
            if nav_links_idx != -1:
                # Find where the nav links end
                nav_links_end = new_html.find('</div>', nav_links_idx)
                if nav_links_end != -1:
                    # Add the new nav link
                    ml_nav_link = '<a href="#ml-analysis" class="nav-link">Advanced Analysis</a>'
                    new_html = new_html[:nav_links_end] + ml_nav_link + new_html[nav_links_end:]
            
            # Write the updated HTML
            with open('phd_applications_dashboard.html', 'w', encoding='utf-8') as f:
                f.write(new_html)
                
            logging.info("Added machine learning section to dashboard")
            
        return True
        
    except Exception as e:
        logging.error(f"Error merging machine learning plots: {str(e)}")
        # If there was an error and we have a backup, restore it
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, 'phd_applications_dashboard.html')
            logging.info("Restored dashboard from backup")
        return False

# At the end of the script, after generating the original dashboard:
try:
    dashboard_html = html_template.format(**plots_html)
    with open('phd_applications_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    logging.info("Interactive dashboard created at 'phd_applications_dashboard.html'")
    
    # After creating the dashboard, merge in the ML analysis
    if merge_dashboard():
        logging.info("Successfully added machine learning analysis to dashboard")
    else:
        logging.error("Failed to add machine learning analysis to dashboard")
        
    print("Analysis completed! Check out the interactive dashboard.")
except Exception as e:
    logging.error(f"Error generating dashboard HTML: {str(e)}")
    print("Failed to generate dashboard. Check the logs for details.")
# --- End of script --- 