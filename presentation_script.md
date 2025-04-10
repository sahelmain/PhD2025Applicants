# PhD Applications Data Analysis - Presentation Script

## Introduction

"Thank you for the opportunity to analyze the PhD application data. I've created a comprehensive analysis using Python and Plotly that explores the demographics, academic qualifications, and test scores of our Computer Science PhD applicants through an interactive dashboard. This analysis can help identify patterns in our applicant pool and potentially inform future recruitment and admissions strategies."

## Demonstration Flow

### 1. The Analysis Approach

"For this analysis, I used Python with pandas and Plotly libraries to clean, analyze, and generate interactive visualizations. The key output is an interactive HTML dashboard that allows us to explore various aspects of the applicant pool directly."

"The first step involved thorough data cleaning, including:
- Handling inconsistent encodings in the data file
- Standardizing country names and formatting
- Converting data to appropriate numeric types
- Documenting and handling missing values"

### 2. Interactive Dashboard Demo

"Let's explore the interactive dashboard (`phd_applications_dashboard.html`). This allows us to dive deeper into the data visually."

[OPEN phd_applications_dashboard.html in a browser]

"As you can see, we can hover over elements to get specific numbers, zoom in on areas of interest, and interact with the plots."

**(Guide through the sections of the dashboard):**

**a. Applicant Demographics**
"Here we see the applicant breakdown by country, student type, and applicant type. Notice how you can hover to see the exact count for each bar. Bangladesh, China, and Iran are the top represented countries."

**b. Academic Qualifications**
"Next, we have the academic profiles. The GPA distribution shows the mean and median, and hovering reveals counts for each bin. We can also interactively explore the degree types and top subject areas."

**c. Test Scores**
"The dashboard shows distributions for English tests and GRE scores. Again, hovering provides details, and we can zoom into specific score ranges."

**d. Relationship Analysis**
"Here, the scatter plots show relationships, like GPA vs. IELTS or GRE scores. The interactive plots include trendlines, and hovering over points reveals individual applicant data (like GPA and score)."

**e. Country Comparisons**
"Finally, these interactive box plots compare GPA and IELTS scores across the top countries. Hovering over the boxes shows quartiles, median, and outliers."

### 3. Key Insights and Recommendations

"This interactive exploration reveals several key insights:"

[Refer to key_insights.md if needed, but focus on what the dashboard shows]

"1. Our program has a strong draw from specific international regions, notably Bangladesh, China, and Iran. The interactivity allows us to easily compare their profiles.

2. While GPAs cluster around 3.3-3.5, the interactive plots let us easily see the spread and identify outliers or specific ranges.

3. The dominance of engineering backgrounds is clear, but the interactive plots make it easy to see the relative frequencies of other fields.

4. The interactive test score plots provide clear visual benchmarks (e.g., IELTS median is 7.0, TOEFL median 96.0). The scatter plots show weak-to-moderate correlations between GPA and test scores.

5. Country comparison plots highlight variations in academic profiles, which could inform targeted recruitment or support strategies."

### 4. Technical Approach (Briefly)

"Behind this dashboard is a well-documented Python script (`analyze_phd_applications.py`) that handles the data processing and generates these interactive Plotly visualizations, embedding them into a single HTML file for easy sharing and viewing."

### 5. Future Analysis Directions

"This interactive dashboard provides a great foundation. Future steps could involve:

1. Integrating acceptance data to add another layer of interactivity, perhaps coloring points by admission status.
2. Adding filters to the dashboard (e.g., filter by country or term).
3. Extending the analysis to include application timing or other variables."

## Conclusion

"This interactive analysis provides a dynamic way to understand our PhD applicant pool and can directly inform strategic decisions. The entire process is automated in a Python script, making it repeatable and extensible for future application cycles.

The key deliverables are:
- The interactive HTML dashboard (`phd_applications_dashboard.html`).
- The underlying Python analysis script.
- Supporting documentation like the cleaned data and summary statistics.

I'm confident this approach demonstrates the ability to not only analyze data but also to present it effectively using modern, interactive tools. I'd be happy to discuss any aspect further." 