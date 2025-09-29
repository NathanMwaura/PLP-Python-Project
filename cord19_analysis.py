import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import re
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class CORD19Analyzer:
    """
    A class to analyze CORD-19 research dataset
    """
    
    def __init__(self, file_path):
        """
        Initialize the analyzer with the dataset
        
        Args:
            file_path (str): Path to the metadata.csv file
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
            Exception: For other file reading errors
        """
        import os
        
        print("Loading CORD-19 dataset...")

        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, 'metadata.csv')
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}\nPlease download metadata.csv from CORD-19 dataset.")
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"File size: {file_size:.2f} MB")
        
        try:
            # Load the dataset with progress indication for large files
            if file_size > 100:
                print("Large file detected. This may take a moment...")
            
            self.df = pd.read_csv(
                file_path,
                low_memory=False,
                usecols=['title', 'abstract', 'journal', 'publish_time', 'source_x'],  # adjust as needed
                nrows=100000  # adjust or remove for full dataset
            )
            print(f"✅ Dataset loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
            
            # Store file path for reference
            self.file_path = file_path
            
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"The file {file_path} is empty")
        
        
    def basic_exploration(self):
        """
        Perform basic data exploration
        """
        print("\n=== BASIC DATA EXPLORATION ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumn names:")
        for i, col in enumerate(self.df.columns):
            print(f"{i+1:2d}. {col}")
        
        print(f"\nData types:")
        print(self.df.dtypes)
        
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
        print(f"\nBasic statistics:")
        print(self.df.describe())
        
        return self.df.info()
    
    def check_missing_values(self):
        """
        Check for missing values in the dataset
        """
        print("\n=== MISSING VALUES ANALYSIS ===")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        })
        
        # Sort by missing percentage
        missing_df = missing_df.sort_values('Missing Percentage', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        
        return missing_df
    
    def clean_data(self):
        """
        Clean the data for analysis
        """
        print("\n=== DATA CLEANING ===")
        original_shape = self.df.shape
        
        # Create a copy for cleaning
        self.df_clean = self.df.copy()
        
        # Convert publish_time to datetime
        if 'publish_time' in self.df_clean.columns:
            self.df_clean['publish_time'] = pd.to_datetime(self.df_clean['publish_time'], errors='coerce')
            self.df_clean['year'] = self.df_clean['publish_time'].dt.year
            self.df_clean['month'] = self.df_clean['publish_time'].dt.month
        
        # Remove rows without title or abstract
        if 'title' in self.df_clean.columns:
            self.df_clean = self.df_clean.dropna(subset=['title'])
        
        # Fill missing journal names
        if 'journal' in self.df_clean.columns:
            self.df_clean['journal'] = self.df_clean['journal'].fillna('Unknown Journal')
        
        # Create word counts for abstracts and titles
        if 'abstract' in self.df_clean.columns:
            self.df_clean['abstract_word_count'] = self.df_clean['abstract'].fillna('').apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
        
        if 'title' in self.df_clean.columns:
            self.df_clean['title_word_count'] = self.df_clean['title'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
        
        print(f"Original shape: {original_shape}")
        print(f"Cleaned shape: {self.df_clean.shape}")
        print(f"Rows removed: {original_shape[0] - self.df_clean.shape[0]}")
        
        return self.df_clean
    
    def analyze_publications_by_year(self):
        """
        Analyze publication trends by year
        """
        if 'year' not in self.df_clean.columns:
            print("Year data not available")
            return None
            
        year_counts = self.df_clean['year'].value_counts().sort_index()
        
        # Filter reasonable years (2000-2023)
        year_counts = year_counts[(year_counts.index >= 2000) & (year_counts.index <= 2023)]
        
        plt.figure(figsize=(12, 6))
        plt.bar(year_counts.index, year_counts.values, color='skyblue', alpha=0.7)
        plt.title('COVID-19 Research Publications by Year', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Publications', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(year_counts.values):
            plt.text(year_counts.index[i], v + 50, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return year_counts
    
    def analyze_top_journals(self, top_n=15):
        """
        Analyze top journals publishing COVID-19 research
        
        Args:
            top_n (int): Number of top journals to display
        """
        if 'journal' not in self.df_clean.columns:
            print("Journal data not available")
            return None
            
        journal_counts = self.df_clean['journal'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(journal_counts)), journal_counts.values, color='lightcoral', alpha=0.7)
        plt.title(f'Top {top_n} Journals Publishing COVID-19 Research', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Publications', fontsize=12)
        plt.ylabel('Journals', fontsize=12)
        plt.yticks(range(len(journal_counts)), journal_counts.index)
        
        # Add value labels on bars
        for i, v in enumerate(journal_counts.values):
            plt.text(v + 10, i, str(v), va='center')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return journal_counts
    
    def analyze_title_words(self, top_n=20):
        """
        Analyze most frequent words in paper titles
        
        Args:
            top_n (int): Number of top words to display
        """
        if 'title' not in self.df_clean.columns:
            print("Title data not available")
            return None
        
        # Combine all titles
        all_titles = ' '.join(self.df_clean['title'].fillna('').astype(str))
        
        # Clean and split words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'been', 'have', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'some', 'has', 'had', 'did', 'get', 'may', 'him', 'old', 'see', 'two', 'way', 'who', 'its', 'said', 'make', 'most', 'her', 'him', 'use', 'day', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'man', 'our', 'under', 'stop', 'never'}
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequency
        word_freq = Counter(filtered_words)
        top_words = word_freq.most_common(top_n)
        
        # Create visualization
        words, counts = zip(*top_words)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(words)), counts, color='lightgreen', alpha=0.7)
        plt.title(f'Top {top_n} Words in Paper Titles', fontsize=16, fontweight='bold')
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Words', fontsize=12)
        plt.yticks(range(len(words)), words)
        
        # Add value labels
        for i, v in enumerate(counts):
            plt.text(v + 10, i, str(v), va='center')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return top_words
    
    def create_title_wordcloud(self):
        """
        Create a word cloud from paper titles
        """
        if 'title' not in self.df_clean.columns:
            print("Title data not available")
            return None
        
        # Combine all titles
        all_titles = ' '.join(self.df_clean['title'].fillna('').astype(str))
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            max_words=100,
                            colormap='viridis').generate(all_titles)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Word Cloud of Paper Titles', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return wordcloud
    
    def analyze_by_source(self):
        """
        Analyze papers by source
        """
        if 'source_x' in self.df_clean.columns:
            source_col = 'source_x'
        elif 'source' in self.df_clean.columns:
            source_col = 'source'
        else:
            print("Source data not available")
            return None
        
        source_counts = self.df_clean[source_col].value_counts()
        
        plt.figure(figsize=(10, 6))
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Papers by Source', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        return source_counts
    
    def generate_summary_report(self):
        """
        Generate a summary report of the analysis
        """
        print("\n" + "="*50)
        print("CORD-19 DATASET ANALYSIS SUMMARY")
        print("="*50)
        
        print(f"\nDataset Overview:")
        print(f"- Total papers: {len(self.df_clean):,}")
        print(f"- Date range: {self.df_clean['year'].min() if 'year' in self.df_clean.columns else 'N/A'} - {self.df_clean['year'].max() if 'year' in self.df_clean.columns else 'N/A'}")
        
        if 'journal' in self.df_clean.columns:
            print(f"- Unique journals: {self.df_clean['journal'].nunique():,}")
            print(f"- Top journal: {self.df_clean['journal'].value_counts().index[0]} ({self.df_clean['journal'].value_counts().iloc[0]} papers)")
        
        if 'abstract_word_count' in self.df_clean.columns:
            avg_abstract_length = self.df_clean['abstract_word_count'].mean()
            print(f"- Average abstract length: {avg_abstract_length:.1f} words")
        
        if 'year' in self.df_clean.columns:
            peak_year = self.df_clean['year'].value_counts().index[0]
            peak_count = self.df_clean['year'].value_counts().iloc[0]
            print(f"- Peak publication year: {peak_year} ({peak_count:,} papers)")
        
        print("\nKey Insights:")
        print("- The dataset represents a comprehensive collection of COVID-19 research")
        print("- Publications show clear temporal patterns related to the pandemic")
        print("- Medical and scientific journals dominate the publication landscape")
        
def main():
    """
    Main function to run the analysis
    """
    # Note: Replace 'metadata.csv' with the actual path to your file
    file_path = 'metadata.csv'
    
    try:
        # Initialize analyzer
        analyzer = CORD19Analyzer(file_path)
        
        # Perform basic exploration
        analyzer.basic_exploration()
        
        # Check missing values
        analyzer.check_missing_values()
        
        # Clean data
        analyzer.clean_data()
        
        # Perform analyses
        print("\n=== ANALYSIS RESULTS ===")
        analyzer.analyze_publications_by_year()
        analyzer.analyze_top_journals()
        analyzer.analyze_title_words()
        analyzer.create_title_wordcloud()
        analyzer.analyze_by_source()
        
        # Generate summary
        analyzer.generate_summary_report()
        
        print("\nAnalysis completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please download the metadata.csv file from the CORD-19 dataset and update the file_path variable.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
