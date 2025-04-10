import os
import logging
import shutil
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def merge_dashboard():
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
            
            # Read the original dashboard content
            with open('phd_applications_dashboard.html', 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Find the insertion point - right before the closing body tag
            body_close_index = original_content.rfind('</body>')
            if body_close_index == -1:
                logging.error("Could not find closing body tag in HTML file")
                return False
            
            # Split the content
            start_content = original_content[:body_close_index].strip()
            end_content = original_content[body_close_index:]
            
            # Create the enhanced content with proper string formatting
            enhanced_content = start_content + """
<style>
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
</style>

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
/* Data table functionality */
document.addEventListener('DOMContentLoaded', function() {
    // Load data from CSV
    fetch('cleaned_phd_applications.csv')
        .then(response => response.text())
        .then(data => {
            const rows = data.split('\\n');
            const headers = rows[0].split(',');
            const tableBody = document.getElementById('applicants-table-body');
            
            // Display first 10 rows
            for (let i = 1; i <= 10 && i < rows.length; i++) {
                const values = rows[i].split(',');
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${values[0] || ''}</td>
                    <td>${values[5] || ''}</td>
                    <td>${values[9] || ''}</td>
                    <td>${values[10] || ''}</td>
                    <td>${values[11] || ''}</td>
                    <td>${values[19] || ''}</td>
                    <td>${values[20] || ''}</td>
                    <td><button class="action-button">View</button></td>
                `;
                tableBody.appendChild(row);
            }
        });

    // Compare functionality
    const compareSubmit = document.getElementById('compare-submit');
    if (compareSubmit) {
        compareSubmit.addEventListener('click', function() {
            const gpa = document.getElementById('compare-gpa').value;
            const verbal = document.getElementById('compare-verbal').value;
            const quant = document.getElementById('compare-quant').value;
            const testType = document.getElementById('compare-test-type').value;
            const testScore = document.getElementById('compare-test-score').value;
            
            // Create comparison visualizations using Plotly
            if (gpa) {
                Plotly.newPlot('gpa-comparison', [{
                    type: 'violin',
                    y: [gpa],
                    name: 'Your GPA',
                    box: { visible: true },
                    meanline: { visible: true }
                }], {
                    title: 'GPA Comparison',
                    height: 300
                });
            }
            
            if (verbal && quant) {
                Plotly.newPlot('gre-comparison', [{
                    type: 'scatter',
                    x: [verbal],
                    y: [quant],
                    mode: 'markers',
                    name: 'Your Scores',
                    marker: { size: 12 }
                }], {
                    title: 'GRE Score Comparison',
                    xaxis: { title: 'Verbal' },
                    yaxis: { title: 'Quantitative' },
                    height: 300
                });
            }
            
            if (testScore) {
                Plotly.newPlot('english-comparison', [{
                    type: 'indicator',
                    mode: 'gauge+number',
                    value: testScore,
                    title: { text: `${testType} Score` },
                    gauge: { axis: { range: [0, 120] } }
                }], {
                    height: 300
                });
            }
        });
    }
});
</script>
""" + end_content
            
            # Write the enhanced content back to the file
            with open('phd_applications_dashboard.html', 'w', encoding='utf-8') as f:
                f.write(enhanced_content)
            
            logging.info("Successfully merged and enhanced dashboard")
            return True
            
    except Exception as e:
        logging.error(f"Error merging dashboard: {e}")
        # Restore from backup if something went wrong
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, 'phd_applications_dashboard.html')
            logging.info("Restored from backup due to error")
        return False

if __name__ == "__main__":
    if merge_dashboard():
        print("Successfully merged and enhanced the dashboard!")
        print("Open phd_applications_dashboard.html in your browser to see the changes.")
    else:
        print("Failed to merge dashboard. Check the logs for details.") 