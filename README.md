# CORD-19 Analysis Project Setup Guide

## 🚀 Quick Start

### 1. Create GitHub Repository
```bash
# Create a new repository named 'Frameworks_Assignment'
git clone https://github.com/yourusername/Frameworks_Assignment.git
cd Frameworks_Assignment
```

### 2. Set Up Python Environment
```bash
# Create virtual environment (recommended)
python -m venv cord19_env

# Activate virtual environment
# On Windows:
cord19_env\Scripts\activate
# On macOS/Linux:
source cord19_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
1. Visit [CORD-19 on Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
2. Download the `metadata.csv` file
3. Place it in the project root directory

### 5. Run the Project
```bash
# Check everything is set up correctly
python run_analysis.py --check

# Run basic analysis
python run_analysis.py --analysis

# Launch interactive Streamlit app
python run_analysis.py --app

# Or run everything at once
python run_analysis.py --all
```

## 📁 Project Structure

After setup, your project should look like this:

```
Frameworks_Assignment/
├── cord19_analysis.py          # Main analysis script
├── streamlit_app.py           # Interactive web application
├── cord19_exploration.ipynb   # Jupyter notebook for exploration
├── run_analysis.py           # Project runner script
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── metadata.csv             # CORD-19 dataset (download separately)
├── outputs/                 # Generated results (created automatically)
│   └── figures/            # Saved visualizations
└── .gitignore              # Git ignore file
```

## 🔧 Detailed Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Git (for version control)
- Internet connection (for downloading dependencies and dataset)

### Step-by-Step Setup

1. **Create Project Directory**
   ```bash
   mkdir Frameworks_Assignment
   cd Frameworks_Assignment
   ```

2. **Initialize Git Repository**
   ```bash
   git init
   git remote add origin https://github.com/yourusername/Frameworks_Assignment.git
   ```

3. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source venv/bin/activate
   ```

4. **Create Project Files**
   Copy all the provided code files into your project directory:
   - `cord19_analysis.py`
   - `streamlit_app.py`
   - `requirements.txt`
   - `config.py`
   - `run_analysis.py`
   - `README.md`

5. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. **Download Dataset**
   - Go to [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
   - Download `metadata.csv` (approximately 100-200 MB)
   - Save it in your project root directory

7. **Set Up Project Structure**
   ```bash
   python run_analysis.py --setup
   ```

8. **Verify Installation**
   ```bash
   python run_analysis.py --check
   ```

## 🏃‍♂️ Running the Analysis

### Option 1: Run Everything
```bash
python run_analysis.py --all
```

### Option 2: Run Components Separately

**Basic Analysis:**
```bash
python cord19_analysis.py
```

**Interactive Web App:**
```bash
streamlit run streamlit_app.py
```

**Jupyter Notebook:**
```bash
jupyter notebook cord19_exploration.ipynb
```

## 🐛 Troubleshooting

### Common Issues and Solutions

**1. "metadata.csv not found" Error**
- Make sure you've downloaded the file from Kaggle
- Check that the file is in the project root directory
- Verify the filename is exactly `metadata.csv`

**2. Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install individually
pip install pandas matplotlib seaborn streamlit numpy plotly wordcloud
```

**3. Streamlit App Not Loading**
```bash
# Check if Streamlit is installed
streamlit --version

# Try running with explicit Python path
python -m streamlit run streamlit_app.py

# Check if port 8501 is available
```

**4. Memory Issues with Large Dataset**
- The full CORD-19 dataset can be very large
- Consider using a smaller sample for initial testing
- Monitor memory usage during analysis

**5. Jupyter Notebook Issues**
```bash
# Install Jupyter if not available
pip install jupyter

# Launch from command line
jupyter notebook

# Or use Jupyter Lab
pip install jupyterlab
jupyter lab
```

### Performance Tips

**For Large Datasets:**
- Use data sampling for initial exploration
- Implement chunked processing for memory efficiency
- Consider using Dask for larger-than-memory datasets

**For Streamlit App:**
- Use `@st.cache_data` for expensive computations
- Limit initial data loading
- Implement progressive loading for large visualizations

## 🎯 Assignment Submission

### Required Deliverables

1. **GitHub Repository**: `Frameworks_Assignment`
2. **Core Files**:
   - `cord19_analysis.py` - Main analysis script
   - `streamlit_app.py` - Interactive web application
   - `requirements.txt` - Dependencies list
   - `README.md` - Project documentation

3. **Optional Enhancements**:
   - Jupyter notebook with exploratory analysis
   - Configuration files
   - Additional visualizations
   - Extended documentation

### Submission Checklist

- [ ] GitHub repository created and configured
- [ ] All code files uploaded and functional
- [ ] README.md with clear instructions
- [ ] Requirements.txt with all dependencies
- [ ] Streamlit app runs without errors
- [ ] Analysis produces meaningful visualizations
- [ ] Code is well-commented and organized
- [ ] Repository URL submitted to instructor

### Repository URL Format
```
https://github.com/yourusername/Frameworks_Assignment
```

## 📊 Expected Outcomes

After completing this setup, you should have:

1. **Functional Analysis Pipeline**: Scripts that load, clean, and analyze CORD-19 data
2. **Interactive Web Application**: Streamlit app for data exploration
3. **Comprehensive Visualizations**: Charts showing publication trends, journal analysis, word patterns
4. **Well-Documented Code**: Clear comments and documentation
5. **Professional Repository**: Organized GitHub repo ready for submission

## 🆘 Getting Help

If you encounter issues:

1. **Check the README.md** for detailed usage instructions
2. **Review error messages** carefully for specific issues
3. **Verify file paths** and data availability
4. **Check Python and package versions** for compatibility
5. **Consult course resources** or instructor for guidance

## 🎉 Success Indicators

Your setup is successful when:
- ✅ All dependencies install without errors
- ✅ Dataset loads and basic info displays
- ✅ Streamlit app launches and shows data
- ✅ Visualizations generate correctly
- ✅ No critical errors in analysis pipeline
- ✅ GitHub repository is properly organized

Good luck with your CORD-19 analysis project! 🦠📊
