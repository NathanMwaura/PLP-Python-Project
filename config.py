"""
Configuration file for CORD-19 analysis project
"""

import os

# File paths
DATA_PATH = 'metadata.csv'
OUTPUT_DIR = 'outputs'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Analysis parameters
DEFAULT_TOP_JOURNALS = 15
DEFAULT_TOP_WORDS = 20
DEFAULT_YEAR_RANGE = (2019, 2023)

# Visualization settings
FIGURE_SIZE_LARGE = (14, 10)
FIGURE_SIZE_MEDIUM = (12, 8)
FIGURE_SIZE_SMALL = (10, 6)

# Color palettes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#17becf'
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': "CORD-19 Data Explorer",
    'page_icon': "🦠",
    'layout': "wide"
}

# Data cleaning parameters
MIN_TITLE_LENGTH = 5
MIN_ABSTRACT_LENGTH = 10
YEAR_FILTER_RANGE = (2000, 2025)

# Word analysis settings
STOP_WORDS = {
    'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'been',
    'have', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about',
    'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just',
    'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only',
    'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here',
    'through', 'when', 'where', 'much', 'some', 'has', 'had', 'did', 'get',
    'may', 'him', 'old', 'see', 'two', 'way', 'who', 'its', 'make', 'most',
    'her', 'use', 'day', 'even', 'new', 'want', 'because', 'any', 'these',
    'give', 'man', 'our', 'under', 'never', 'well', 'back', 'could', 'good',
    'take', 'come', 'state', 'used', 'part', 'between', 'high', 'right',
    'during', 'both', 'without', 'many', 'might', 'home', 'help', 'small',
    'world', 'another', 'does', 'three', 'every', 'must', 'while', 'same'
}

# WordCloud settings
WORDCLOUD_CONFIG = {
    'width': 800,
    'height': 400,
    'background_color': 'white',
    'max_words': 100,
    'colormap': 'viridis'
}

def create_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"✅ Output directories created: {OUTPUT_DIR}, {FIGURES_DIR}")

def get_data_path():
    """Get the path to the data file"""
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    else:
        print(f"❌ Data file not found: {DATA_PATH}")
        print("Please download metadata.csv from CORD-19 dataset")
        return None

if __name__ == "__main__":
    create_output_dirs()
    print("Configuration loaded successfully!")
