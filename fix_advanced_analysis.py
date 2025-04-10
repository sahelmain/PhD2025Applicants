import os
import logging
import shutil
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fix_advanced_analysis():
    try:
        # First check if the dashboard file exists
        dashboard_file = 'dashboard.html'
        if not os.path.exists(dashboard_file):
            logging.error(f"Dashboard file {dashboard_file} not found")
            return False
            
        # Create a backup of the original file
        backup_file = 'dashboard.html.bak'
        shutil.copy2(dashboard_file, backup_file)
        logging.info(f"Created backup at {backup_file}")
        
        # Read the CSV data
        try:
            df = pd.read_csv('cleaned_phd_applications.csv')
            logging.info(f"Successfully loaded data with {len(df)} rows")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return False
            
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
        
        # Convert plots to HTML
        elbow_curve_html = fig_elbow.to_html(full_html=False, include_plotlyjs='cdn')
        cluster_pca_html = fig_pca.to_html(full_html=False, include_plotlyjs=False)
        parallel_clusters_html = fig_parallel_clusters.to_html(full_html=False, include_plotlyjs=False)
        radar_clusters_html = fig_radar_clusters.to_html(full_html=False, include_plotlyjs=False)
        
        # Read the HTML file
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Check if the advanced analysis section already exists
        if 'id="advanced-analysis"' in html_content:
            logging.info("Advanced analysis section already exists in dashboard")
            return True
            
        # Find where to insert the advanced analysis section - right before the footer
        footer_idx = html_content.find('<div class="footer">')
        if footer_idx == -1:
            # Try to find the closing </div> of the last section instead
            footer_idx = html_content.rfind('</div>')
            
        if footer_idx == -1:
            logging.error("Could not find position to insert advanced analysis section")
            return False
            
        # Create the advanced analysis section HTML
        advanced_section = f"""
        <div id="advanced-analysis" class="section">
            <h2>Machine Learning Analysis</h2>
            <p class="description">In-depth analysis using machine learning techniques to identify patterns and clusters in PhD applicant data.</p>
            
            <div class="plot-container">
                <h3>Optimal Number of Clusters</h3>
                {elbow_curve_html}
                <p class="plot-description">The elbow method helps determine the optimal number of clusters for k-means clustering. The "elbow" point indicates where adding more clusters provides diminishing returns.</p>
            </div>
            
            <div class="plot-container">
                <h3>Applicant Clusters</h3>
                {cluster_pca_html}
                <p class="plot-description">This scatter plot shows clusters of applicants based on their academic metrics, reduced to 2 dimensions using Principal Component Analysis (PCA).</p>
            </div>
            
            <div class="plot-container">
                <h3>Multi-dimensional Analysis by Cluster</h3>
                {parallel_clusters_html}
                <p class="plot-description">Parallel coordinates plot showing how different clusters compare across multiple dimensions simultaneously.</p>
            </div>
            
            <div class="plot-container">
                <h3>Cluster Characteristics</h3>
                {radar_clusters_html}
                <p class="plot-description">Radar chart comparing the average characteristics of each applicant cluster across standardized metrics.</p>
            </div>
        </div>
        """
        
        # Insert the advanced analysis section
        new_html = html_content[:footer_idx] + advanced_section + html_content[footer_idx:]
        
        # Add a link to the advanced analysis section in the nav bar
        nav_links_idx = new_html.find('<div class="nav-links">')
        if nav_links_idx != -1:
            # Find where the nav links end
            nav_links_end = new_html.find('</div>', nav_links_idx)
            if nav_links_end != -1:
                # Add the new nav link
                ml_nav_link = '<a href="#advanced-analysis" class="nav-link">ML Analysis</a>'
                new_html = new_html[:nav_links_end] + ml_nav_link + new_html[nav_links_end:]
        
        # Write the updated HTML
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(new_html)
            
        logging.info("Added advanced analysis section to dashboard")
        return True
        
    except Exception as e:
        logging.error(f"Error adding advanced analysis: {str(e)}")
        # If there was an error and we have a backup, restore it
        if 'backup_file' in locals() and os.path.exists(backup_file):
            shutil.copy2(backup_file, dashboard_file)
            logging.info(f"Restored dashboard from backup")
        return False
        
if __name__ == "__main__":
    if fix_advanced_analysis():
        print("Successfully added advanced analysis section to the dashboard!")
        print("Open dashboard.html in your browser to see the changes.")
    else:
        print("Failed to add advanced analysis section. Check the logs for details.") 