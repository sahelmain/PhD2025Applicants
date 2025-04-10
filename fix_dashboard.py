import os
import logging
import shutil
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fix_dashboard():
    try:
        # First, run the analysis script to generate the base dashboard
        logging.info("Running analysis script...")
        result = subprocess.run(['python', 'analyze_phd_applications_fixed.py'], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Analysis script failed: {result.stderr}")
            return False

        # Create a backup of the generated dashboard
        if os.path.exists('phd_applications_dashboard.html'):
            backup_file = 'phd_applications_dashboard.backup.html'
            shutil.copy2('phd_applications_dashboard.html', backup_file)
            logging.info(f"Created backup at {backup_file}")

        # Create the enhanced content with a complete structure
        enhanced_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhD Applications Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.0/papaparse.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #e0e0e0;
            background: #1a1a1a;
            margin: 0;
            padding: 20px;
        }

        .section {
            margin-bottom: 40px;
            background: rgba(255,255,255,0.03);
            padding: 20px;
            border-radius: 8px;
        }

        h1, h2 {
            color: #ffffff;
            margin-top: 0;
        }

        .description {
            margin-bottom: 20px;
            color: #b0b0b0;
        }

        .action-button {
            background: #4a90e2;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }

        .action-button.secondary {
            background: rgba(255,255,255,0.1);
        }

        .action-button:hover {
            opacity: 0.9;
        }

        /* Data table styles */
        .table-container {
            margin: 20px 0;
            overflow-x: auto;
        }

        .table-controls {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }

        .data-table th,
        .data-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .data-table th {
            background: rgba(255,255,255,0.1);
            font-weight: bold;
        }

        .data-table tr:hover {
            background: rgba(255,255,255,0.1);
        }

        /* Compare section styles */
        .compare-widget {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }

        .compare-form {
            flex: 1;
            min-width: 300px;
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 8px;
        }

        .form-group {
            margin-bottom: 15px;
        }

        .form-input,
        .form-select {
            width: 100%;
            padding: 8px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 4px;
            color: #e0e0e0;
        }

        .comparison-charts {
            flex: 2;
            min-width: 400px;
        }

        .comparison-chart {
            margin-bottom: 20px;
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 8px;
        }

        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .plot-container {
            background: rgba(255,255,255,0.03);
            padding: 15px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <h1>PhD Applications Analysis Dashboard</h1>
    
    <div id="demographics" class="section">
        <h2>1. Demographics</h2>
        <div class="description">
            <p>Distribution of applicants by country and subject area.</p>
        </div>
        <div class="grid-container">
            <div class="plot-container" id="country-dist"></div>
            <div class="plot-container" id="subject-dist"></div>
        </div>
    </div>

    <div id="academics" class="section">
        <h2>2. Academic Qualifications</h2>
        <div class="description">
            <p>Distribution of GPAs and degree types among applicants.</p>
        </div>
        <div class="grid-container">
            <div class="plot-container" id="gpa-dist"></div>
            <div class="plot-container" id="degree-dist"></div>
        </div>
    </div>

    <div id="test-scores" class="section">
        <h2>3. Test Scores</h2>
        <div class="description">
            <p>Analysis of GRE and English test scores.</p>
        </div>
        <div class="grid-container">
            <div class="plot-container" id="gre-scores"></div>
            <div class="plot-container" id="english-scores"></div>
        </div>
    </div>

    <div id="correlations" class="section">
        <h2>4. Score Correlations</h2>
        <div class="description">
            <p>Relationships between different test scores and academic performance.</p>
        </div>
        <div class="grid-container">
            <div class="plot-container" id="gpa-gre-correlation"></div>
            <div class="plot-container" id="score-matrix"></div>
        </div>
    </div>

    <div id="trends" class="section">
        <h2>5. Application Trends</h2>
        <div class="description">
            <p>Temporal patterns in applications and score distributions.</p>
        </div>
        <div class="grid-container">
            <div class="plot-container" id="time-series"></div>
            <div class="plot-container" id="seasonal-patterns"></div>
        </div>
    </div>

    <div id="clustering" class="section">
        <h2>6. Applicant Clusters</h2>
        <div class="description">
            <p>Machine learning-based clustering of applicant profiles.</p>
        </div>
        <div class="grid-container">
            <div class="plot-container" id="cluster-plot"></div>
            <div class="plot-container" id="cluster-characteristics"></div>
        </div>
    </div>

    <div id="parallel" class="section">
        <h2>7. Parallel Coordinates</h2>
        <div class="description">
            <p>Multi-dimensional visualization of applicant characteristics.</p>
        </div>
        <div class="plot-container" id="parallel-coords"></div>
    </div>

    <!-- Data Table Section -->
    <div id="data-table" class="section">
        <h2>8. Data Table</h2>
        <div class="description">
            <p>Interactive table of all applicants with search, sort, and filter capabilities.</p>
        </div>
        <div class="table-container">
            <div class="table-controls">
                <input type="text" id="table-search" placeholder="Search applicants..." class="form-input">
                <button id="export-csv" class="action-button secondary">Export CSV</button>
            </div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Country</th>
                        <th>GPA</th>
                        <th>Degree</th>
                        <th>Subject</th>
                        <th>Test Type</th>
                        <th>Score</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="applicants-table-body">
                </tbody>
            </table>
            <div class="pagination">
                <button id="prev-page" class="action-button secondary" disabled>Previous</button>
                <span id="page-info">Page 1 of 1</span>
                <button id="next-page" class="action-button secondary">Next</button>
            </div>
        </div>
    </div>

    <!-- Compare Section -->
    <div id="compare" class="section">
        <h2>9. Compare Your Profile</h2>
        <div class="description">
            <p>Compare your academic profile with the current applicant pool.</p>
        </div>
        <div class="compare-widget">
            <div class="compare-form">
                <div class="form-group">
                    <label for="compare-gpa">GPA (4.0 scale)</label>
                    <input type="number" id="compare-gpa" min="0" max="4" step="0.01" class="form-input">
                </div>
                <div class="form-group">
                    <label for="compare-verbal">GRE Verbal</label>
                    <input type="number" id="compare-verbal" min="130" max="170" class="form-input">
                </div>
                <div class="form-group">
                    <label for="compare-quant">GRE Quantitative</label>
                    <input type="number" id="compare-quant" min="130" max="170" class="form-input">
                </div>
                <div class="form-group">
                    <label for="compare-test-type">English Test</label>
                    <select id="compare-test-type" class="form-select">
                        <option value="">Select Test</option>
                        <option value="TOEFL">TOEFL</option>
                        <option value="IELTS">IELTS</option>
                        <option value="Duolingo">Duolingo</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="compare-test-score">Test Score</label>
                    <input type="number" id="compare-test-score" class="form-input">
                </div>
                <button id="compare-submit" class="action-button">Compare</button>
                <button id="compare-reset" class="action-button secondary">Reset</button>
            </div>
            <div class="comparison-charts">
                <div id="gpa-comparison"></div>
                <div id="gre-comparison"></div>
                <div id="english-comparison"></div>
            </div>
        </div>
    </div>

    <script>
        // Load and process data
        function loadData() {
            return fetch('cleaned_phd_applications.csv')
                .then(response => response.text())
                .then(data => Papa.parse(data, { header: true }))
                .then(result => result.data);
        }

        // Create visualizations
        function createVisualizations(data) {
            // Demographics plots
            const countryData = data.reduce((acc, row) => {
                acc[row.Country] = (acc[row.Country] || 0) + 1;
                return acc;
            }, {});

            Plotly.newPlot('country-dist', [{
                values: Object.values(countryData),
                labels: Object.keys(countryData),
                type: 'pie',
                textinfo: 'label+percent',
                hoverinfo: 'label+value',
                hole: 0.4
            }], {
                title: 'Applicants by Country',
                height: 400,
                showlegend: true,
                legend: { orientation: 'h' }
            });

            const subjectData = data.reduce((acc, row) => {
                acc[row.Subject] = (acc[row.Subject] || 0) + 1;
                return acc;
            }, {});

            Plotly.newPlot('subject-dist', [{
                x: Object.keys(subjectData),
                y: Object.values(subjectData),
                type: 'bar',
                marker: {
                    color: 'rgba(74, 144, 226, 0.8)'
                }
            }], {
                title: 'Distribution by Subject Area',
                height: 400,
                xaxis: { tickangle: 45 },
                yaxis: { title: 'Number of Applicants' }
            });

            // Academics plots
            const gpaValues = data.map(row => parseFloat(row.GPA)).filter(gpa => !isNaN(gpa));
            
            Plotly.newPlot('gpa-dist', [{
                x: gpaValues,
                type: 'histogram',
                nbinsx: 20,
                marker: {
                    color: 'rgba(74, 144, 226, 0.8)'
                }
            }], {
                title: 'GPA Distribution',
                height: 400,
                xaxis: { title: 'GPA' },
                yaxis: { title: 'Frequency' }
            });

            const degreeData = data.reduce((acc, row) => {
                acc[row.Degree] = (acc[row.Degree] || 0) + 1;
                return acc;
            }, {});

            Plotly.newPlot('degree-dist', [{
                values: Object.values(degreeData),
                labels: Object.keys(degreeData),
                type: 'pie',
                textinfo: 'label+percent',
                hoverinfo: 'label+value'
            }], {
                title: 'Degree Types',
                height: 400
            });

            // Test scores plots
            const greScores = data.filter(row => 
                !isNaN(parseFloat(row.Verbal)) && !isNaN(parseFloat(row.Quantitative))
            ).map(row => ({
                verbal: parseFloat(row.Verbal),
                quant: parseFloat(row.Quantitative)
            }));

            Plotly.newPlot('gre-scores', [{
                x: greScores.map(score => score.verbal),
                y: greScores.map(score => score.quant),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    color: 'rgba(74, 144, 226, 0.6)',
                    size: 8
                }
            }], {
                title: 'GRE Score Distribution',
                height: 400,
                xaxis: { title: 'Verbal Score' },
                yaxis: { title: 'Quantitative Score' }
            });

            // Initialize data table
            const tableBody = document.getElementById('applicants-table-body');
            data.slice(0, 10).forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.ID || ''}</td>
                    <td>${row.Country || ''}</td>
                    <td>${row.GPA || ''}</td>
                    <td>${row.Degree || ''}</td>
                    <td>${row.Subject || ''}</td>
                    <td>${row['Test Type'] || ''}</td>
                    <td>${row.Score || ''}</td>
                    <td><button class="action-button">View</button></td>
                `;
                tableBody.appendChild(tr);
            });
        }

        // Initialize comparison functionality
        function initializeComparison() {
            const compareSubmit = document.getElementById('compare-submit');
            const compareReset = document.getElementById('compare-reset');

            if (compareSubmit) {
                compareSubmit.addEventListener('click', function() {
                    const gpa = document.getElementById('compare-gpa').value;
                    const verbal = document.getElementById('compare-verbal').value;
                    const quant = document.getElementById('compare-quant').value;
                    const testType = document.getElementById('compare-test-type').value;
                    const testScore = document.getElementById('compare-test-score').value;
                    
                    loadData().then(data => {
                        // GPA comparison
                        if (gpa) {
                            const gpaValues = data.map(row => parseFloat(row.GPA)).filter(val => !isNaN(val));
                            Plotly.newPlot('gpa-comparison', [{
                                type: 'violin',
                                y: gpaValues,
                                name: 'All Applicants',
                                box: { visible: true },
                                meanline: { visible: true }
                            }, {
                                type: 'scatter',
                                y: [gpa],
                                name: 'Your GPA',
                                mode: 'markers',
                                marker: { size: 12, color: 'red' }
                            }], {
                                title: 'GPA Comparison',
                                height: 300
                            });
                        }
                        
                        // GRE comparison
                        if (verbal && quant) {
                            const greScores = data.filter(row => 
                                !isNaN(parseFloat(row.Verbal)) && !isNaN(parseFloat(row.Quantitative))
                            ).map(row => ({
                                verbal: parseFloat(row.Verbal),
                                quant: parseFloat(row.Quantitative)
                            }));

                            Plotly.newPlot('gre-comparison', [{
                                x: greScores.map(score => score.verbal),
                                y: greScores.map(score => score.quant),
                                mode: 'markers',
                                name: 'All Applicants',
                                marker: { size: 8, color: 'rgba(74, 144, 226, 0.6)' }
                            }, {
                                x: [verbal],
                                y: [quant],
                                mode: 'markers',
                                name: 'Your Scores',
                                marker: { size: 12, color: 'red' }
                            }], {
                                title: 'GRE Score Comparison',
                                height: 300,
                                xaxis: { title: 'Verbal' },
                                yaxis: { title: 'Quantitative' }
                            });
                        }
                        
                        // English test score comparison
                        if (testScore && testType) {
                            const testScores = data
                                .filter(row => row['Test Type'] === testType)
                                .map(row => parseFloat(row.Score))
                                .filter(score => !isNaN(score));

                            Plotly.newPlot('english-comparison', [{
                                type: 'violin',
                                y: testScores,
                                name: `All ${testType} Scores`,
                                box: { visible: true },
                                meanline: { visible: true }
                            }, {
                                type: 'scatter',
                                y: [testScore],
                                name: 'Your Score',
                                mode: 'markers',
                                marker: { size: 12, color: 'red' }
                            }], {
                                title: `${testType} Score Comparison`,
                                height: 300
                            });
                        }
                    });
                });
            }

            if (compareReset) {
                compareReset.addEventListener('click', function() {
                    document.getElementById('compare-gpa').value = '';
                    document.getElementById('compare-verbal').value = '';
                    document.getElementById('compare-quant').value = '';
                    document.getElementById('compare-test-type').value = '';
                    document.getElementById('compare-test-score').value = '';
                    
                    ['gpa-comparison', 'gre-comparison', 'english-comparison'].forEach(id => {
                        const element = document.getElementById(id);
                        if (element) element.innerHTML = '';
                    });
                });
            }
        }

        // Initialize everything when the page loads
        document.addEventListener('DOMContentLoaded', function() {
            loadData().then(data => {
                createVisualizations(data);
                initializeComparison();
            }).catch(error => {
                console.error('Error loading data:', error);
            });
        });
    </script>
</body>
</html>"""

        # Write the enhanced content to the file
        with open('phd_applications_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(enhanced_content)

        logging.info("Successfully fixed and enhanced dashboard")
        return True

    except Exception as e:
        logging.error(f"Error fixing dashboard: {e}")
        # Restore from backup if something went wrong
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, 'phd_applications_dashboard.html')
            logging.info("Restored from backup due to error")
        return False

if __name__ == "__main__":
    if fix_dashboard():
        print("Successfully fixed and enhanced the dashboard!")
        print("Open phd_applications_dashboard.html in your browser to see the changes.")
    else:
        print("Failed to fix dashboard. Check the logs for details.") 