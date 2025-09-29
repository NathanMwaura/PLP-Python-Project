#!/usr/bin/env python3
"""
CORD-19 Analysis Project Runner
Main script to run different components of the CORD-19 analysis project
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

def print_header():
    """Print a nice header for the application"""
    print("\n" + "="*60)
    print("🦠 CORD-19 Analysis Project Runner".center(60))
    print("="*60 + "\n")

def print_section(title):
    """Print a section header"""
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def check_requirements():
    """Check if required packages are installed"""
    print_section("📦 Checking Required Packages")
    
    required_packages = {
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'streamlit': 'streamlit',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'wordcloud': 'wordcloud'
    }
    
    missing_packages = []
    installed_packages = []
    
    for display_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            installed_packages.append(display_name)
            print(f"  ✅ {display_name:<15} - Installed")
        except ImportError:
            missing_packages.append(display_name)
            print(f"  ❌ {display_name:<15} - Missing")
    
    print()
    if missing_packages:
        print(f"❌ Missing {len(missing_packages)} package(s): {', '.join(missing_packages)}")
        print("\n💡 To install missing packages, run:")
        print("   pip install -r requirements.txt")
        print("\n   Or install individually:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print(f"✅ All {len(required_packages)} required packages are installed!")
    return True

def check_data_file():
    """Check if the data file exists"""
    print_section("📊 Checking Data File")
    
    data_file = Path('metadata.csv')
    
    if not data_file.exists():
        print("  ❌ metadata.csv not found in current directory")
        print("\n💡 To download the dataset:")
        print("   1. Visit: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge")
        print("   2. Download the 'metadata.csv' file")
        print("   3. Place it in the project root directory")
        return False
    
    # Check file size
    file_size = data_file.stat().st_size / (1024 * 1024)  # Convert to MB
    print(f"  ✅ metadata.csv found ({file_size:.2f} MB)")
    
    # Try to read first few lines to verify it's valid
    try:
        import pandas as pd
        df_test = pd.read_csv(data_file, nrows=5)
        print(f"  ✅ File is readable (detected {len(df_test.columns)} columns)")
        return True
    except Exception as e:
        print(f"  ⚠️  Warning: File may be corrupted - {str(e)}")
        return False

def check_project_files():
    """Check if required project files exist"""
    print_section("📁 Checking Project Files")
    
    required_files = {
        'cord19_analysis.py': 'Main analysis script',
        'streamlit_app.py': 'Streamlit web application',
        'requirements.txt': 'Dependencies list',
        'README.md': 'Project documentation'
    }
    
    missing_files = []
    
    for filename, description in required_files.items():
        if Path(filename).exists():
            print(f"  ✅ {filename:<25} - {description}")
        else:
            print(f"  ❌ {filename:<25} - {description} (Missing)")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} file(s) missing: {', '.join(missing_files)}")
        print("   Some functionality may not be available.")
        return False
    
    print(f"\n✅ All {len(required_files)} project files are present!")
    return True

def run_basic_analysis():
    """Run the basic analysis script"""
    print_section("🔬 Running Basic Analysis")
    
    if not Path('cord19_analysis.py').exists():
        print("  ❌ cord19_analysis.py not found")
        return False
    
    if not Path('metadata.csv').exists():
        print("  ❌ metadata.csv not found")
        print("  Please download the dataset first")
        return False
    
    print("  🔄 Starting analysis... This may take a few minutes.")
    print("  (You can interrupt with Ctrl+C if needed)\n")
    
    try:
        # Run the analysis script
        result = subprocess.run(
            [sys.executable, 'cord19_analysis.py'],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n  ✅ Analysis completed successfully!")
            return True
        else:
            print(f"\n  ❌ Analysis failed with return code {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n  ⚠️  Analysis interrupted by user")
        return False
    except Exception as e:
        print(f"\n  ❌ Error running analysis: {str(e)}")
        return False

def run_streamlit_app():
    """Launch the Streamlit application"""
    print_section("🚀 Launching Streamlit Application")
    
    if not Path('streamlit_app.py').exists():
        print("  ❌ streamlit_app.py not found")
        return False
    
    print("  🔄 Starting Streamlit server...")
    print("  📱 The app will open in your default browser")
    print("  ⏹️  Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'])
        return True
    except KeyboardInterrupt:
        print("\n  ⏹️  Streamlit server stopped")
        return True
    except FileNotFoundError:
        print("  ❌ Streamlit not found. Install it with: pip install streamlit")
        return False
    except Exception as e:
        print(f"  ❌ Error launching Streamlit: {str(e)}")
        return False

def run_jupyter_notebook():
    """Launch Jupyter notebook"""
    print_section("📓 Launching Jupyter Notebook")
    
    notebook_file = 'cord19_exploration.ipynb'
    
    if Path(notebook_file).exists():
        print(f"  📖 Opening {notebook_file}...")
    else:
        print(f"  ⚠️  {notebook_file} not found. Opening Jupyter anyway...")
    
    print("  ⏹️  Press Ctrl+C in terminal to stop Jupyter server\n")
    
    try:
        if Path(notebook_file).exists():
            subprocess.run([sys.executable, '-m', 'jupyter', 'notebook', notebook_file])
        else:
            subprocess.run([sys.executable, '-m', 'jupyter', 'notebook'])
        return True
    except KeyboardInterrupt:
        print("\n  ⏹️  Jupyter server stopped")
        return True
    except FileNotFoundError:
        print("  ❌ Jupyter not installed. Install with: pip install jupyter")
        return False
    except Exception as e:
        print(f"  ❌ Error launching Jupyter: {str(e)}")
        return False

def setup_project():
    """Set up the project structure"""
    print_section("🔧 Setting Up Project Structure")
    
    # Create directories
    directories = {
        'outputs': 'Analysis results',
        'outputs/figures': 'Generated visualizations',
        'data': 'Data files'
    }
    
    print("  Creating project directories...")
    for directory, description in directories.items():
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"    ✅ Created: {directory:<20} - {description}")
        else:
            print(f"    ℹ️  Exists:  {directory:<20} - {description}")
    
    # Create .gitignore if it doesn't exist
    gitignore_content = """# Data files
*.csv
data/
metadata.csv

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
cord19_env/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Outputs
outputs/
*.png
*.jpg
*.jpeg
*.pdf

# Distribution / packaging
dist/
build/
*.egg-info/
"""
    
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content.strip())
        print(f"\n  ✅ Created .gitignore")
    else:
        print(f"\n  ℹ️  .gitignore already exists")
    
    print("\n✅ Project structure setup complete!")
    return True

def create_sample_config():
    """Create a sample configuration file"""
    print_section("⚙️  Creating Sample Configuration")
    
    config_path = Path('config.py')
    
    if config_path.exists():
        print("  ℹ️  config.py already exists")
        return True
    
    config_content = '''"""
Configuration file for CORD-19 analysis project
"""

# File paths
DATA_PATH = 'metadata.csv'
OUTPUT_DIR = 'outputs'

# Analysis parameters
DEFAULT_TOP_JOURNALS = 15
DEFAULT_TOP_WORDS = 20
DEFAULT_YEAR_RANGE = (2019, 2023)

# Visualization settings
FIGURE_SIZE = (12, 8)

print("✅ Configuration loaded")
'''
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print("  ✅ config.py created")
    return True

def show_status():
    """Show overall project status"""
    print_section("📊 Project Status Summary")
    
    status_items = {
        'Python Environment': sys.executable,
        'Python Version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'Working Directory': os.getcwd(),
        'Data File': '✅ Present' if Path('metadata.csv').exists() else '❌ Missing',
        'Analysis Script': '✅ Present' if Path('cord19_analysis.py').exists() else '❌ Missing',
        'Streamlit App': '✅ Present' if Path('streamlit_app.py').exists() else '❌ Missing',
        'Output Directory': '✅ Present' if Path('outputs').exists() else '❌ Missing'
    }
    
    for item, value in status_items.items():
        print(f"  {item:<20}: {value}")
    
    print()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='CORD-19 Analysis Project Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --check          # Check all requirements
  python run_analysis.py --setup          # Set up project structure
  python run_analysis.py --analysis       # Run basic analysis
  python run_analysis.py --app            # Launch Streamlit app
  python run_analysis.py --all            # Check everything and run analysis
  python run_analysis.py --status         # Show project status

For first-time setup, run:
  python run_analysis.py --setup --check
        """
    )
    
    parser.add_argument('--setup', action='store_true', 
                       help='Set up project structure (create directories, .gitignore)')
    parser.add_argument('--check', action='store_true', 
                       help='Check requirements, data file, and project files')
    parser.add_argument('--analysis', action='store_true', 
                       help='Run the basic analysis script')
    parser.add_argument('--app', action='store_true', 
                       help='Launch the Streamlit web application')
    parser.add_argument('--notebook', action='store_true', 
                       help='Launch Jupyter notebook')
    parser.add_argument('--all', action='store_true', 
                       help='Run full check and basic analysis')
    parser.add_argument('--status', action='store_true', 
                       help='Show project status summary')
    parser.add_argument('--config', action='store_true', 
                       help='Create sample configuration file')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        print("\n💡 Tip: Start with 'python run_analysis.py --check' to verify your setup")
        return
    
    # Track overall success
    all_success = True
    
    # Status command
    if args.status:
        show_status()
    
    # Setup command
    if args.setup:
        success = setup_project()
        all_success &= success
    
    # Config command
    if args.config:
        success = create_sample_config()
        all_success &= success
    
    # Check commands
    if args.check or args.all:
        req_success = check_requirements()
        file_success = check_project_files()
        data_success = check_data_file()
        
        all_success &= req_success and file_success and data_success
        
        if not all_success:
            print("\n⚠️  Some checks failed. Please fix the issues above before proceeding.")
            if not args.all:
                return
    
    # Analysis command
    if args.analysis or args.all:
        if all_success or not args.all:
            success = run_basic_analysis()
            all_success &= success
        else:
            print("\n⚠️  Skipping analysis due to failed checks")
    
    # App command
    if args.app:
        if all_success or not args.all:
            run_streamlit_app()
        else:
            print("\n⚠️  Fix the issues above before launching the app")
    
    # Notebook command
    if args.notebook:
        run_jupyter_notebook()
    
    # Final summary
    print("\n" + "="*60)
    if all_success:
        print("🎉 All operations completed successfully!".center(60))
    else:
        print("⚠️  Some operations failed. Check messages above.".center(60))
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
