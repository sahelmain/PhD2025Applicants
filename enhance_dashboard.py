import os
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Data table section HTML
data_table_section = """
<div id="data-table" class="section">
    <div class="section-header">
        <h2>8. Applicant Data Table</h2>
        <i class="fas fa-chevron-down toggle-icon"></i>
    </div>
    <div class="section-content">
        <p class="description">Interactive table of all applicants with search, sort, and filter capabilities. Click on rows to view detailed information.</p>
        
        <div class="plot-container">
            <h3>Applicant Data Table</h3>
            <div class="table-controls">
                <div class="search-container">
                    <input type="text" id="table-search" placeholder="Search applicants..." class="search-input">
                    <i class="fas fa-search search-icon"></i>
                </div>
                <button id="export-csv" class="action-button secondary">
                    <i class="fas fa-download"></i> Export CSV
                </button>
            </div>
            
            <div class="table-container">
                <table id="applicants-table" class="data-table">
                    <thead>
                        <tr>
                            <th data-sort="id">ID <i class="fas fa-sort"></i></th>
                            <th data-sort="country">Country <i class="fas fa-sort"></i></th>
                            <th data-sort="gpa">GPA <i class="fas fa-sort"></i></th>
                            <th data-sort="degree">Degree <i class="fas fa-sort"></i></th>
                            <th data-sort="subject">Subject Area <i class="fas fa-sort"></i></th>
                            <th data-sort="test">English Test <i class="fas fa-sort"></i></th>
                            <th data-sort="score">Test Score <i class="fas fa-sort"></i></th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Rows will be populated by JavaScript -->
                    </tbody>
                </table>
            </div>
            
            <div class="pagination">
                <button id="prev-page" class="pagination-btn" disabled><i class="fas fa-chevron-left"></i></button>
                <span id="page-info">Page 1 of 1</span>
                <button id="next-page" class="pagination-btn"><i class="fas fa-chevron-right"></i></button>
            </div>
        </div>
    </div>
</div>
"""

# Compare yourself section HTML
compare_section = """
<div id="compare" class="section">
    <div class="section-header">
        <h2>9. Compare Yourself</h2>
        <i class="fas fa-chevron-down toggle-icon"></i>
    </div>
    <div class="section-content">
        <p class="description">Enter your academic information to see how you compare with the applicant pool.</p>
        
        <div class="plot-container">
            <h3>Compare Your Profile</h3>
            <div class="compare-widget">
                <div class="compare-form">
                    <h3>Your Information</h3>
                    <div class="form-group">
                        <label for="compare-gpa">GPA (4.0 scale)</label>
                        <input type="number" id="compare-gpa" min="0" max="4" step="0.01" class="form-input" placeholder="e.g., 3.5">
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="compare-verbal">GRE Verbal</label>
                            <input type="number" id="compare-verbal" min="130" max="170" class="form-input" placeholder="130-170">
                        </div>
                        <div class="form-group">
                            <label for="compare-quant">GRE Quant</label>
                            <input type="number" id="compare-quant" min="130" max="170" class="form-input" placeholder="130-170">
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="compare-test-type">English Test</label>
                        <select id="compare-test-type" class="form-select">
                            <option value="">Select Test</option>
                            <option value="TOEFL">TOEFL</option>
                            <option value="IELTS">IELTS</option>
                            <option value="Duolingo">Duolingo</option>
                            <option value="PTE">PTE Academic</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="compare-test-score">Test Score</label>
                        <input type="number" id="compare-test-score" class="form-input" placeholder="Enter score">
                    </div>
                    
                    <button id="compare-submit" class="action-button">Compare</button>
                    <button id="compare-reset" class="action-button secondary">Reset</button>
                </div>
                
                <div class="compare-results">
                    <h3>Comparison Results</h3>
                    <div id="results-placeholder" class="results-message">
                        <i class="fas fa-chart-line"></i>
                        <p>Enter your information and click "Compare" to see how you compare with current applicants.</p>
                    </div>
                    <div id="comparison-charts" class="comparison-charts" style="display: none;">
                        <div class="comparison-chart" id="gpa-comparison-chart"></div>
                        <div class="comparison-chart" id="gre-comparison-chart"></div>
                        <div class="comparison-chart" id="english-comparison-chart"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
"""

# Additional CSS for the new sections
additional_css = """
<style>
/* Data Table Styling */
.search-container {
    position: relative;
    flex: 1;
    max-width: 400px;
}

.search-input {
    width: 100%;
    padding: 12px 40px 12px 15px;
    border-radius: 8px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: var(--text-color);
    font-size: 1rem;
    transition: all 0.3s;
}

body.light-theme .search-input {
    background: rgba(0,0,0,0.05);
    color: var(--light-text-color);
    border-color: rgba(0,0,0,0.1);
}

.table-controls {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
    flex-wrap: wrap;
    gap: 15px;
}

.search-icon {
    position: absolute;
    right: 15px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-muted-color);
}

.table-container {
    overflow-x: auto;
    background: rgba(0,0,0,0.1);
    border-radius: 8px;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}

.data-table th {
    background: rgba(0,0,0,0.2);
    color: var(--text-color);
    font-weight: 600;
    text-align: left;
    padding: 15px;
    position: sticky;
    top: 0;
    z-index: 10;
}

.data-table th i {
    margin-left: 5px;
    font-size: 0.8rem;
    opacity: 0.7;
}

.data-table tr:nth-child(even) {
    background: rgba(255,255,255,0.03);
}

.data-table td {
    padding: 12px 15px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 20px;
    gap: 15px;
}

.pagination-btn {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: var(--text-color);
    border-radius: 8px;
    padding: 8px 15px;
    cursor: pointer;
    transition: all 0.3s;
}

.pagination-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Compare Widget Styling */
.compare-widget {
    display: flex;
    gap: 30px;
    flex-wrap: wrap;
}

.compare-form {
    flex: 1;
    min-width: 300px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 25px;
}

.compare-form h3 {
    margin-top: 0;
    margin-bottom: 20px;
    text-align: left;
    font-size: 1.3rem;
}

.compare-results {
    flex: 2;
    min-width: 400px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 25px;
}

.compare-results h3 {
    margin-top: 0;
    margin-bottom: 20px;
    text-align: left;
    font-size: 1.3rem;
}

.form-group {
    margin-bottom: 20px;
}

.form-row {
    display: flex;
    gap: 15px;
    margin-bottom: 20px;
}

.form-row .form-group {
    flex: 1;
    margin-bottom: 0;
}

.form-input, .form-select {
    width: 100%;
    padding: 12px 15px;
    border-radius: 8px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: var(--text-color);
    font-size: 1rem;
    transition: all 0.3s;
}

.results-message {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 30px;
    color: var(--text-muted-color);
    text-align: center;
    gap: 15px;
}

.results-message i {
    font-size: 3rem;
    opacity: 0.7;
}

.comparison-charts {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.comparison-chart {
    height: 200px;
    width: 100%;
}

.action-button {
    padding: 10px 18px;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s;
    background: linear-gradient(135deg, #4E79A7 0%, #3a5e85 100%);
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.action-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}

.action-button.secondary {
    background: rgba(255,255,255,0.1);
    color: var(--text-color);
}
</style>
"""

# Additional JavaScript for the new functionality
additional_js = """
<script>
// Add new nav links
document.addEventListener('DOMContentLoaded', function() {
    // Add Data Table & Compare sections to navigation
    const navLinks = document.querySelector('.nav-links');
    if (navLinks) {
        const mlLink = document.querySelector('a[href="#ml-analysis"]');
        if (mlLink) {
            const dataTableLink = document.createElement('a');
            dataTableLink.href = '#data-table';
            dataTableLink.className = 'nav-link';
            dataTableLink.textContent = 'Data Table';
            
            const compareLink = document.createElement('a');
            compareLink.href = '#compare';
            compareLink.className = 'nav-link';
            compareLink.textContent = 'Compare';
            
            navLinks.insertBefore(dataTableLink, mlLink.nextSibling);
            navLinks.insertBefore(compareLink, dataTableLink.nextSibling);
        }
    }
    
    // Data table functionality
    const tableBody = document.querySelector('#applicants-table tbody');
    if (tableBody) {
        // Fetch and load data
        fetch('cleaned_phd_applications.csv')
            .then(response => response.text())
            .then(data => {
                // Parse CSV
                const rows = data.split('\\n');
                const headers = rows[0].split(',');
                
                // Extract data
                const applicants = [];
                for (let i = 1; i < rows.length; i++) {
                    if (!rows[i]) continue;
                    
                    const values = rows[i].split(',');
                    const applicant = {
                        id: values[0],
                        country: values[5] || 'Unknown',
                        gpa: parseFloat(values[9]) || 'N/A',
                        degree: values[10] || 'N/A',
                        subject: values[11] || 'N/A',
                        testType: values[19] || 'N/A',
                        testScore: values[20] || 'N/A'
                    };
                    
                    applicants.push(applicant);
                }
                
                // Populate table with first page
                const rowsPerPage = 10;
                const firstPage = applicants.slice(0, rowsPerPage);
                
                firstPage.forEach(applicant => {
                    const row = document.createElement('tr');
                    
                    row.innerHTML = `
                        <td>${applicant.id}</td>
                        <td>${applicant.country}</td>
                        <td>${typeof applicant.gpa === 'number' ? applicant.gpa.toFixed(2) : applicant.gpa}</td>
                        <td>${applicant.degree}</td>
                        <td>${applicant.subject}</td>
                        <td>${applicant.testType}</td>
                        <td>${applicant.testScore}</td>
                        <td>
                            <button class="table-action-btn view-btn" data-id="${applicant.id}">
                                <i class="fas fa-eye"></i>
                            </button>
                        </td>
                    `;
                    
                    tableBody.appendChild(row);
                });
                
                // Set up pagination
                const pageInfo = document.getElementById('page-info');
                if (pageInfo) {
                    const maxPage = Math.ceil(applicants.length / rowsPerPage);
                    pageInfo.textContent = `Page 1 of ${maxPage}`;
                }
                
                // Set up search
                const searchInput = document.getElementById('table-search');
                if (searchInput) {
                    searchInput.addEventListener('input', function() {
                        const searchTerm = this.value.toLowerCase();
                        tableBody.innerHTML = '';
                        
                        const filtered = applicants.filter(app => 
                            Object.values(app).some(val => 
                                val && val.toString().toLowerCase().includes(searchTerm)
                            )
                        );
                        
                        const firstPageFiltered = filtered.slice(0, rowsPerPage);
                        
                        if (firstPageFiltered.length === 0) {
                            tableBody.innerHTML = '<tr><td colspan="8">No matching applicants found</td></tr>';
                        } else {
                            firstPageFiltered.forEach(applicant => {
                                const row = document.createElement('tr');
                                
                                row.innerHTML = `
                                    <td>${applicant.id}</td>
                                    <td>${applicant.country}</td>
                                    <td>${typeof applicant.gpa === 'number' ? applicant.gpa.toFixed(2) : applicant.gpa}</td>
                                    <td>${applicant.degree}</td>
                                    <td>${applicant.subject}</td>
                                    <td>${applicant.testType}</td>
                                    <td>${applicant.testScore}</td>
                                    <td>
                                        <button class="table-action-btn view-btn" data-id="${applicant.id}">
                                            <i class="fas fa-eye"></i>
                                        </button>
                                    </td>
                                `;
                                
                                tableBody.appendChild(row);
                            });
                        }
                    });
                }
                
                // Set up export
                const exportBtn = document.getElementById('export-csv');
                if (exportBtn) {
                    exportBtn.addEventListener('click', function() {
                        const csv = [headers.join(',')];
                        applicants.forEach(app => {
                            const values = [
                                app.id,
                                app.country,
                                app.gpa,
                                app.degree,
                                app.subject,
                                app.testType,
                                app.testScore
                            ];
                            csv.push(values.join(','));
                        });
                        
                        const blob = new Blob([csv.join('\\n')], { type: 'text/csv' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'phd_applicants.csv';
                        a.click();
                        URL.revokeObjectURL(url);
                    });
                }
            })
            .catch(error => {
                console.error('Error loading CSV:', error);
                if (tableBody) {
                    tableBody.innerHTML = '<tr><td colspan="8">Error loading data</td></tr>';
                }
            });
    }
    
    // Compare functionality
    const compareSubmit = document.getElementById('compare-submit');
    const compareReset = document.getElementById('compare-reset');
    
    if (compareSubmit && compareReset) {
        const compareGPA = document.getElementById('compare-gpa');
        const compareVerbal = document.getElementById('compare-verbal');
        const compareQuant = document.getElementById('compare-quant');
        const compareTestType = document.getElementById('compare-test-type');
        const compareTestScore = document.getElementById('compare-test-score');
        const resultsPlaceholder = document.getElementById('results-placeholder');
        const comparisonCharts = document.getElementById('comparison-charts');
        
        // Stats for comparison
        const stats = {
            gpa: { mean: 3.30, min: 1.59, max: 3.98 },
            gre: { 
                verbal: { mean: 152.29, min: 135, max: 170 },
                quant: { mean: 155.13, min: 130, max: 170 }
            },
            english: {
                ielts: { mean: 6.94, min: 6, max: 8.5 },
                toefl: { mean: 95.39, min: 83, max: 110 },
                duolingo: { mean: 117.73, min: 105, max: 140 }
            }
        };
        
        compareSubmit.addEventListener('click', function() {
            const gpa = parseFloat(compareGPA.value);
            const verbal = parseInt(compareVerbal.value);
            const quant = parseInt(compareQuant.value);
            const testType = compareTestType.value.toLowerCase();
            const testScore = parseFloat(compareTestScore.value);
            
            // Hide placeholder and show charts
            if (resultsPlaceholder && comparisonCharts) {
                resultsPlaceholder.style.display = 'none';
                comparisonCharts.style.display = 'block';
            }
            
            // Create GPA comparison chart
            if (gpa && !isNaN(gpa)) {
                const gpaChart = document.getElementById('gpa-comparison-chart');
                if (gpaChart) {
                    Plotly.newPlot(gpaChart, [
                        {
                            type: 'indicator',
                            mode: 'gauge+number+delta',
                            value: gpa,
                            title: { text: 'Your GPA vs Average', font: { size: 16 } },
                            delta: { reference: stats.gpa.mean, valueformat: '.2f' },
                            gauge: {
                                axis: { range: [stats.gpa.min, stats.gpa.max], tickwidth: 1 },
                                bar: { color: '#4E79A7' },
                                bgcolor: 'white',
                                borderwidth: 2,
                                bordercolor: 'gray',
                                threshold: {
                                    line: { color: 'red', width: 2 },
                                    thickness: 0.75,
                                    value: stats.gpa.mean
                                }
                            }
                        }
                    ], {
                        height: 200,
                        margin: { t: 25, b: 25, l: 25, r: 25 }
                    });
                }
            }
            
            // Create GRE comparison chart
            if ((verbal && !isNaN(verbal)) || (quant && !isNaN(quant))) {
                const greChart = document.getElementById('gre-comparison-chart');
                if (greChart) {
                    Plotly.newPlot(greChart, [
                        {
                            x: ['Verbal', 'Quantitative'],
                            y: [verbal || 0, quant || 0],
                            type: 'bar',
                            name: 'Your Score',
                            marker: { color: ['#4E79A7', '#F28E2B'] }
                        },
                        {
                            x: ['Verbal', 'Quantitative'],
                            y: [stats.gre.verbal.mean, stats.gre.quant.mean],
                            type: 'bar',
                            name: 'Average Score',
                            marker: { color: ['rgba(78,121,167,0.5)', 'rgba(242,142,43,0.5)'] }
                        }
                    ], {
                        title: 'Your GRE Scores vs Average',
                        height: 200,
                        margin: { t: 30, b: 30, l: 30, r: 30 },
                        barmode: 'group'
                    });
                }
            }
            
            // Create English test comparison chart
            if (testType && testScore && !isNaN(testScore)) {
                const englishChart = document.getElementById('english-comparison-chart');
                if (englishChart) {
                    let testStats = stats.english.toefl; // Default
                    if (testType === 'ielts') testStats = stats.english.ielts;
                    if (testType === 'duolingo') testStats = stats.english.duolingo;
                    
                    Plotly.newPlot(englishChart, [
                        {
                            type: 'indicator',
                            mode: 'gauge+number+delta',
                            value: testScore,
                            title: { text: `Your ${testType.toUpperCase()} Score`, font: { size: 16 } },
                            delta: { reference: testStats.mean, valueformat: '.1f' },
                            gauge: {
                                axis: { range: [testStats.min, testStats.max], tickwidth: 1 },
                                bar: { color: '#59A14F' },
                                bgcolor: 'white',
                                borderwidth: 2,
                                bordercolor: 'gray',
                                threshold: {
                                    line: { color: 'red', width: 2 },
                                    thickness: 0.75,
                                    value: testStats.mean
                                }
                            }
                        }
                    ], {
                        height: 200,
                        margin: { t: 25, b: 25, l: 25, r: 25 }
                    });
                }
            }
        });
        
        compareReset.addEventListener('click', function() {
            // Reset form
            compareGPA.value = '';
            compareVerbal.value = '';
            compareQuant.value = '';
            compareTestType.value = '';
            compareTestScore.value = '';
            
            // Show placeholder and hide charts
            if (resultsPlaceholder && comparisonCharts) {
                resultsPlaceholder.style.display = 'flex';
                comparisonCharts.style.display = 'none';
            }
        });
    }
    
    // Set up section collapsing for the new sections
    const sectionHeaders = document.querySelectorAll('.section-header');
    sectionHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const content = header.nextElementSibling;
            if (content && content.classList.contains('section-content')) {
                content.classList.toggle('collapsed');
                const icon = header.querySelector('.toggle-icon');
                if (icon) {
                    icon.style.transform = content.classList.contains('collapsed') 
                        ? 'rotate(180deg)' 
                        : 'rotate(0)';
                }
            }
        });
    });
});
</script>
"""

def enhance_dashboard():
    try:
        # Check if the dashboard file exists
        if not os.path.exists('phd_applications_dashboard.html'):
            logging.error("Dashboard file not found")
            return False

        # First, run the analysis script to ensure we have fresh data
        logging.info("Running analysis script to regenerate dashboard...")
        os.system('python analyze_phd_applications_fixed.py')

        # Create a backup of the original file
        backup_file = 'phd_applications_dashboard.backup.html'
        shutil.copy2('phd_applications_dashboard.html', backup_file)
        logging.info(f"Created backup at {backup_file}")

        # Read the dashboard HTML
        with open('phd_applications_dashboard.html', 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Check if the content already has our enhancements
        if 'Compare Yourself' in html_content:
            logging.info("Dashboard already contains enhancements. No changes needed.")
            return True

        # Find the insertion point - right before the closing body tag
        body_close_index = html_content.rfind('</body>')
        if body_close_index == -1:
            logging.error("Could not find closing body tag in HTML file")
            # Restore from backup
            shutil.copy2(backup_file, 'phd_applications_dashboard.html')
            return False

        # Split the content
        start_content = html_content[:body_close_index]
        end_content = html_content[body_close_index:]

        # Add our new sections and scripts
        enhanced_content = start_content

        # Add the new sections only if they don't already exist
        if '<div id="data-table"' not in html_content:
            enhanced_content += data_table_section

        if '<div id="compare"' not in html_content:
            enhanced_content += compare_section

        # Add the final closing tags
        enhanced_content += end_content

        # Write the enhanced content back to the file
        with open('phd_applications_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(enhanced_content)

        logging.info("Successfully enhanced dashboard with new features")
        return True

    except Exception as e:
        logging.error(f"Error enhancing dashboard: {e}")
        # Restore from backup if something went wrong
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, 'phd_applications_dashboard.html')
            logging.info("Restored from backup due to error")
        return False

    finally:
        # Clean up backup file if everything went well
        if os.path.exists(backup_file) and os.path.exists('phd_applications_dashboard.html'):
            os.remove(backup_file)
            logging.info("Cleaned up backup file")

if __name__ == "__main__":
    if enhance_dashboard():
        print("Successfully enhanced the dashboard!")
        print("Open phd_applications_dashboard.html in your browser to see the changes.")
    else:
        print("Failed to enhance dashboard. Check the logs for details.") 