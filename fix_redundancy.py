import os
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fix_redundancy():
    """
    Remove the redundant Machine Learning Analysis section from the dashboard.
    Keep the one that's part of the Advanced Analysis section.
    """
    try:
        # First check if the dashboard file exists
        dashboard_file = 'dashboard.html'
        if not os.path.exists(dashboard_file):
            logging.error(f"Dashboard file {dashboard_file} not found")
            return False
            
        # Create a backup of the original file
        backup_file = 'dashboard.html.bak.redundancy'
        with open(dashboard_file, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        logging.info(f"Created backup at {backup_file}")
        
        # Read the HTML content
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for the redundant sections
        advanced_analysis_count = html_content.count('id="advanced-analysis"')
        ml_analysis_count = html_content.count('Machine Learning Analysis')
        
        if advanced_analysis_count <= 1 and ml_analysis_count <= 2:
            logging.info("No redundancy detected.")
            return True
        
        # Find and remove the standalone ML Analysis section at the end
        # The pattern is a div with id="advanced-analysis" all the way to its closing div
        pattern = r'<div id="advanced-analysis" class="section">[\s\S]*?<h2>Machine Learning Analysis</h2>[\s\S]*?</div>\s*</div>'
        
        # Find all matches
        matches = list(re.finditer(pattern, html_content))
        
        if len(matches) == 0:
            logging.info("No redundant ML Analysis section found.")
            return True
        
        # Get the last match, which is most likely the redundant one
        last_match = matches[-1]
        redundant_section = last_match.group(0)
        
        # Remove only if this is definitely a redundant section (check if it's near the end)
        footer_pos = html_content.find('<footer>')
        match_pos = last_match.start()
        
        if footer_pos > 0 and match_pos > 0 and match_pos < footer_pos and match_pos > len(html_content) // 2:
            # This is likely the redundant section close to the footer
            new_html = html_content.replace(redundant_section, '')
            
            # Also check for and remove the redundant nav link
            nav_pattern = r'<a href="#advanced-analysis" class="nav-link">ML Analysis</a>'
            new_html = new_html.replace(nav_pattern, '')
            
            # Write the updated content
            with open(dashboard_file, 'w', encoding='utf-8') as f:
                f.write(new_html)
                
            logging.info("Successfully removed redundant Machine Learning Analysis section.")
            return True
        else:
            logging.warning("Redundant section found but position is unexpected. Manual fix recommended.")
            return False
            
    except Exception as e:
        logging.error(f"Error fixing redundancy: {str(e)}")
        return False

if __name__ == "__main__":
    if fix_redundancy():
        print("Successfully fixed redundancy in the dashboard!")
        print("The duplicated Machine Learning Analysis section has been removed.")
    else:
        print("Failed to fix redundancy. Check the logs for details.") 