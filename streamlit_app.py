import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import re
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    """Load and cache the CORD-19 dataset"""
    try:
        df = pd.read_csv(file_path)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def clean_data(df):
    """Clean and preprocess the data"""
    df_clean = df.copy()
    
    # Convert publish_time to datetime
    if 'publish_time' in df_clean.columns:
        df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
        df_clean['year'] = df_clean['publish_time'].dt.year
        df_clean['month'] = df_clean['publish_time'].dt.month
    
    # Remove rows without title
    if 'title' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['title'])
    
    # Fill missing journal names
    if 'journal' in df_clean.columns:
        df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    
    # Create word counts
    if 'abstract' in df_clean.columns:
        df_clean['abstract_word_count'] = df_clean['abstract'].fillna('').apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
    
    if 'title' in df_clean.columns:
        df_clean['title_word_count'] = df_clean['title'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
    
    return df_clean

def create_year_analysis(df, year_range):
    """Create publication analysis by year"""
    if 'year' not in df.columns:
        return None
    
    # Filter by year range
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    year_counts = filtered_df['year'].value_counts().sort_index()
    
    # Create interactive plot
    fig = px.bar(x=year_counts.index, y=year_counts.values,
                 labels={'x': 'Year', 'y': 'Number of Publications'},
                 title='COVID-19 Research Publications by Year')
    fig.update_layout(showlegend=False)
    
    return fig, year_counts

def create_journal_analysis(df, top_n):
    """Create top journals analysis"""
    if 'journal' not in df.columns:
        return None
    
    journal_counts = df['journal'].value_counts().head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(x=journal_counts.values, y=journal_counts.index,
                 orientation='h',
                 labels={'x': 'Number of Publications', 'y': 'Journals'},
                 title=f'Top {top_n} Journals Publishing COVID-19 Research')
    fig.update_layout(height=600)
    
    return fig, journal_counts

def create_word_analysis(df, top_n):
    """Analyze most frequent words in titles"""
    if 'title' not in df.columns:
        return None, None
    
    # Combine all titles
    all_titles = ' '.join(df['title'].fillna('').astype(str))
    
    # Clean and split words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
    
    # Remove stop words
    stop_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'been', 'have', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'some', 'has', 'had', 'did', 'get', 'may', 'him', 'old', 'see', 'two', 'way', 'who', 'its', 'said', 'make', 'most', 'her', 'him', 'use', 'day', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'man', 'our', 'under', 'stop', 'never'}
    
    filtered_words = [word for word in words if word not in stop_words]
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(top_n)
    
    if top_words:
        words, counts = zip(*top_words)
        fig = px.bar(x=counts, y=words, orientation='h',
                     labels={'x': 'Frequency', 'y': 'Words'},
                     title=f'Top {top_n} Words in Paper Titles')
        fig.update_layout(height=600)
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            max_words=100,
                            colormap='viridis').generate(all_titles)
        
        return fig, wordcloud
    
    return None, None

def create_source_analysis(df):
    """Analyze papers by source"""
    source_col = None
    for col in ['source_x', 'source']:
        if col in df.columns:
            source_col = col
            break
    
    if source_col is None:
        return None
    
    source_counts = df[source_col].value_counts()
    
    fig = px.pie(values=source_counts.values, names=source_counts.index,
                 title='Distribution of Papers by Source')
    
    return fig, source_counts

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🦠 CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### Simple exploration of COVID-19 research papers")
    
    # Sidebar
    st.sidebar.header("Dataset Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CORD-19 metadata.csv file",
        type=['csv'],
        help="Download from: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge"
    )
    
    if uploaded_file is not None:
        # Load data
        with st.spinner('Loading dataset...'):
            df, error = load_data(uploaded_file)
        
        if error:
            st.error(f"Error loading file: {error}")
            return
        
        if df is None:
            st.error("Failed to load the dataset")
            return
        
        # Clean data
        with st.spinner('Cleaning data...'):
            df_clean = clean_data(df)
        
        # Display basic info
        st.sidebar.success(f"✅ Dataset loaded successfully!")
        st.sidebar.info(f"📊 **{len(df_clean):,}** papers loaded")
        
        # Sidebar controls
        st.sidebar.header("Analysis Parameters")
        
        # Year range selector
        if 'year' in df_clean.columns:
            year_min = int(df_clean['year'].min()) if not pd.isna(df_clean['year'].min()) else 2000
            year_max = int(df_clean['year'].max()) if not pd.isna(df_clean['year'].max()) else 2023
            year_range = st.sidebar.slider(
                "Select year range",
                min_value=year_min,
                max_value=year_max,
                value=(max(2019, year_min), year_max),
                help="Filter papers by publication year"
            )
        else:
            year_range = (2019, 2023)
        
        # Top N selector for various analyses
        top_journals = st.sidebar.slider("Number of top journals to show", 5, 25, 15)
        top_words = st.sidebar.slider("Number of top words to show", 10, 30, 20)
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", "📅 Time Analysis", "📰 Journal Analysis", 
            "💬 Word Analysis", "📊 Source Analysis"
        ])
        
        with tab1:
            st.header("Dataset Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Papers", f"{len(df_clean):,}")
            
            with col2:
                if 'journal' in df_clean.columns:
                    st.metric("Unique Journals", f"{df_clean['journal'].nunique():,}")
            
            with col3:
                if 'year' in df_clean.columns:
                    st.metric("Year Range", f"{df_clean['year'].min():.0f} - {df_clean['year'].max():.0f}")
            
            with col4:
                if 'abstract_word_count' in df_clean.columns:
                    avg_length = df_clean['abstract_word_count'].mean()
                    st.metric("Avg Abstract Length", f"{avg_length:.0f} words")
            
            # Dataset sample
            st.subheader("Sample Data")
            display_cols = ['title', 'journal', 'publish_time', 'authors']
            available_cols = [col for col in display_cols if col in df_clean.columns]
            if available_cols:
                st.dataframe(df_clean[available_cols].head(10))
            
            # Missing data analysis
            st.subheader("Data Quality")
            missing_data = df_clean.isnull().sum()
            missing_percent = (missing_data / len(df_clean)) * 100
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': missing_percent.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
            
            if not missing_df.empty:
                st.bar_chart(missing_df.set_index('Column')['Missing %'])
        
        with tab2:
            st.header("Publication Trends Over Time")
            
            if 'year' in df_clean.columns:
                fig, year_counts = create_year_analysis(df_clean, year_range)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Key insights
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.write("**Key Insights:**")
                    peak_year = year_counts.idxmax()
                    peak_count = year_counts.max()
                    st.write(f"• Peak publication year: **{peak_year}** with **{peak_count:,}** papers")
                    st.write(f"• Total papers in selected range: **{year_counts.sum():,}**")
                    if len(year_counts) > 1:
                        growth = ((year_counts.iloc[-1] - year_counts.iloc[0]) / year_counts.iloc[0]) * 100
                        st.write(f"• Growth from {year_counts.index[0]} to {year_counts.index[-1]}: **{growth:+.1f}%**")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Year data not available in the dataset")
        
        with tab3:
            st.header("Top Publishing Journals")
            
            if 'journal' in df_clean.columns:
                fig, journal_counts = create_journal_analysis(df_clean, top_journals)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.write("**Journal Statistics:**")
                    st.write(f"• Total unique journals: **{df_clean['journal'].nunique():,}**")
                    st.write(f"• Top journal: **{journal_counts.index[0]}** ({journal_counts.iloc[0]:,} papers)")
                    st.write(f"• Top {top_journals} journals account for **{journal_counts.sum():,}** papers ({(journal_counts.sum()/len(df_clean)*100):.1f}%)")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Journal data not available in the dataset")
        
        with tab4:
            st.header("Word Analysis from Paper Titles")
            
            if 'title' in df_clean.columns:
                fig, wordcloud = create_word_analysis(df_clean, top_words)
                
                if fig:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Most Frequent Words")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Word Cloud")
                        if wordcloud:
                            fig_wc, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig_wc)
                            plt.close()
                
                # Insights
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.write("**Word Analysis Insights:**")
                st.write("• The word cloud and frequency chart show the most common terms in research titles")
                st.write("• Medical and scientific terms dominate, reflecting the nature of COVID-19 research")
                st.write("• Terms like 'covid', 'sars', 'coronavirus' are expectedly prominent")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Title data not available in the dataset")
        
        with tab5:
            st.header("Papers by Source")
            
            source_analysis = create_source_analysis(df_clean)
            if source_analysis and source_analysis[0]:
                fig, source_counts = source_analysis
                st.plotly_chart(fig, use_container_width=True)
                
                # Source statistics
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.write("**Source Distribution:**")
                for source, count in source_counts.head(5).items():
                    percentage = (count / len(df_clean)) * 100
                    st.write(f"• **{source}**: {count:,} papers ({percentage:.1f}%)")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Source data not available in the dataset")
        
        # Footer with additional information
        st.markdown("---")
        st.markdown("### About this Analysis")
        st.info("""
        **Dataset**: CORD-19 (COVID-19 Open Research Dataset)
        
        **Purpose**: This tool provides an interactive exploration of COVID-19 research papers, 
        allowing you to analyze publication trends, journal distributions, and content patterns.
        
        **Features**:
        - Time-based publication analysis
        - Journal ranking and statistics  
        - Word frequency analysis and word clouds
        - Source distribution visualization
        - Interactive filtering and parameter adjustment
        
        **Usage**: Upload the metadata.csv file from the CORD-19 dataset and explore the various tabs 
        to gain insights into the research landscape around COVID-19.
        """)
        
    else:
        # Welcome screen when no file is uploaded
        st.markdown("""
        ## Welcome to the CORD-19 Data Explorer! 🦠
        
        This interactive application helps you explore and analyze the CORD-19 dataset, 
        which contains metadata about COVID-19 research papers.
        
        ### Getting Started:
        
        1. **Download the Dataset**:
           - Visit: [CORD-19 on Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
           - Download the `metadata.csv` file
        
        2. **Upload the File**:
           - Use the file uploader in the sidebar to upload your `metadata.csv` file
        
        3. **Explore the Data**:
           - Navigate through different tabs to analyze various aspects of the research
           - Adjust parameters using the sidebar controls
        
        ### What You'll Discover:
        
        - 📈 **Publication trends** over time
        - 📰 **Top journals** publishing COVID-19 research  
        - 💬 **Word patterns** in research titles
        - 📊 **Source distributions** of the papers
        - 🔍 **Interactive filtering** and exploration
        
        ### Requirements:
        - The metadata.csv file from CORD-19 dataset
        - Stable internet connection for optimal performance
        
        Ready to start exploring? Upload your dataset file using the sidebar! 
        """)
        
        # Sample visualization to show what's possible
        st.markdown("### Preview: Sample Analysis")
        
        # Create sample data for demonstration
        sample_years = list(range(2019, 2024))
        sample_counts = [100, 500, 2000, 1500, 800]
        
        fig_sample = px.bar(x=sample_years, y=sample_counts,
                           title="Sample: COVID-19 Publications by Year",
                           labels={'x': 'Year', 'y': 'Number of Publications'})
        st.plotly_chart(fig_sample, use_container_width=True)
        
        st.caption("*This is sample data for demonstration. Upload your dataset to see real analysis.*")

if __name__ == "__main__":
    main()
