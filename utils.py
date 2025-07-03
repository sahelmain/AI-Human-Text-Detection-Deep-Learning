import pandas as pd
import numpy as np
import re
import nltk
import textstat
from collections import Counter
import PyPDF2
import docx
from fpdf import FPDF
import io
import base64
import pickle
from datetime import datetime

# NLTK Setup and Error Handling
def setup_nltk_safely():
    """Setup NLTK data with error handling"""
    try:
        # Try to download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return True
    except Exception as e:
        print(f"NLTK setup failed: {e}")
        return False

# Safe tokenization functions with fallbacks
def safe_sentence_tokenize(text):
    """Safely tokenize sentences with fallback"""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        # Fallback: simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Ultimate fallback
        return text.split('. ')

def safe_word_tokenize(text):
    """Safely tokenize words with fallback"""
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        # Fallback: simple word splitting
        return text.split()
    except Exception:
        return text.split()

# Initialize NLTK
setup_nltk_safely()

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(uploaded_file):
    """Extract text from Word document"""
    try:
        doc = docx.Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading Word document: {str(e)}"

def extract_text_statistics(text):
    """Extract comprehensive text statistics"""
    
    stats = {}
    
    # Basic counts
    stats['character_count'] = len(text)
    stats['word_count'] = len(text.split())
    stats['sentence_count'] = len(nltk.sent_tokenize(text))
    stats['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
    
    # Average lengths
    words = text.split()
    sentences = nltk.sent_tokenize(text)
    
    stats['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    stats['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
    
    # Readability scores
    if len(text.strip()) > 0:
        stats['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        stats['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        stats['automated_readability_index'] = textstat.automated_readability_index(text)
        stats['coleman_liau_index'] = textstat.coleman_liau_index(text)
        stats['gunning_fog'] = textstat.gunning_fog(text)
        stats['smog_index'] = textstat.smog_index(text)
    else:
        for key in ['flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability_index', 
                   'coleman_liau_index', 'gunning_fog', 'smog_index']:
            stats[key] = 0
    
    # Lexical diversity
    unique_words = len(set(words))
    stats['lexical_diversity'] = unique_words / len(words) if words else 0
    
    # Most common words
    word_freq = Counter([word.lower() for word in words if word.isalpha()])
    stats['most_common_words'] = word_freq.most_common(10)
    
    # Punctuation analysis
    punctuation_marks = ".,!?;:"
    stats['punctuation_count'] = sum(text.count(p) for p in punctuation_marks)
    stats['punctuation_ratio'] = stats['punctuation_count'] / len(text) if len(text) > 0 else 0
    
    return stats

def analyze_text_features(text, vectorizer=None):
    """Analyze text features for machine learning interpretation"""
    
    features = {}
    
    # Basic linguistic features using safe tokenization
    words = safe_word_tokenize(text)
    sentences = safe_sentence_tokenize(text)
    
    # Length features
    features['char_count'] = len(text)
    features['word_count'] = len(words)
    features['sentence_count'] = len(sentences)
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
    
    # Readability features
    if len(text.strip()) > 0:
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        features['automated_readability_index'] = textstat.automated_readability_index(text)
    
    # Stylistic features
    features['punctuation_ratio'] = sum(text.count(p) for p in ".,!?;:") / len(text) if len(text) > 0 else 0
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0
    
    # Lexical diversity
    unique_words = len(set([word.lower() for word in words if word.isalpha()]))
    features['lexical_diversity'] = unique_words / len(words) if words else 0
    
    # Function words ratio (common AI indicators)
    function_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
    function_word_count = sum(1 for word in words if word.lower() in function_words)
    features['function_word_ratio'] = function_word_count / len(words) if words else 0
    
    # If vectorizer is available, get TF-IDF feature importance
    if vectorizer:
        try:
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            # Transform the text
            tfidf_matrix = vectorizer.transform([text])
            # Get non-zero features and their scores
            feature_scores = tfidf_matrix.toarray()[0]
            
            # Get top features
            top_indices = np.argsort(feature_scores)[-20:][::-1]  # Top 20 features
            top_features = [(feature_names[i], feature_scores[i]) for i in top_indices if feature_scores[i] > 0]
            
            features['top_tfidf_features'] = top_features
        except:
            features['top_tfidf_features'] = []
    
    return features

def generate_analysis_report(text, prediction_results, text_stats, model_comparison=None):
    """Generate a comprehensive PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    
    # Title
    pdf.cell(0, 10, "AI vs Human Text Detection Report", 0, 1, 'C')
    pdf.ln(10)
    
    # Timestamp
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)
    
    # Text Preview
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Text Sample (First 200 characters):", 0, 1)
    pdf.set_font("Arial", size=10)
    preview_text = text[:200] + "..." if len(text) > 200 else text
    # Clean text for PDF - remove problematic characters
    clean_preview = preview_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, clean_preview)
    pdf.ln(5)
    
    # Prediction Results
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Prediction Results:", 0, 1)
    pdf.set_font("Arial", size=10)
    
    prediction_text = "AI-Generated" if prediction_results['prediction'] == 1 else "Human-Written"
    pdf.cell(0, 5, f"Prediction: {prediction_text}", 0, 1)
    pdf.cell(0, 5, f"Confidence: {prediction_results['confidence']:.2%}", 0, 1)
    pdf.cell(0, 5, f"Human Probability: {prediction_results['probabilities'][0]:.2%}", 0, 1)
    pdf.cell(0, 5, f"AI Probability: {prediction_results['probabilities'][1]:.2%}", 0, 1)
    pdf.ln(10)
    
    # Text Statistics
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Text Statistics:", 0, 1)
    pdf.set_font("Arial", size=10)
    
    pdf.cell(0, 5, f"Character Count: {text_stats['character_count']}", 0, 1)
    pdf.cell(0, 5, f"Word Count: {text_stats['word_count']}", 0, 1)
    pdf.cell(0, 5, f"Sentence Count: {text_stats['sentence_count']}", 0, 1)
    pdf.cell(0, 5, f"Average Word Length: {text_stats['avg_word_length']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Average Sentence Length: {text_stats['avg_sentence_length']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Lexical Diversity: {text_stats['lexical_diversity']:.3f}", 0, 1)
    pdf.ln(5)
    
    # Readability Scores
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Readability Scores:", 0, 1)
    pdf.set_font("Arial", size=10)
    
    pdf.cell(0, 5, f"Flesch Reading Ease: {text_stats['flesch_reading_ease']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Flesch-Kincaid Grade: {text_stats['flesch_kincaid_grade']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Automated Readability Index: {text_stats['automated_readability_index']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Coleman-Liau Index: {text_stats['coleman_liau_index']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Gunning Fog: {text_stats['gunning_fog']:.2f}", 0, 1)
    pdf.cell(0, 5, f"SMOG Index: {text_stats['smog_index']:.2f}", 0, 1)
    pdf.ln(10)
    
    # Model Comparison (if provided)
    if model_comparison:
        pdf.set_font("Arial", "B", size=12)
        pdf.cell(0, 10, "Model Comparison Results:", 0, 1)
        pdf.set_font("Arial", size=10)
        
        for result in model_comparison:
            pdf.cell(0, 5, f"{result['Model']}: {result['Prediction']} ({result['Confidence']})", 0, 1)
    
    # Convert to bytes - fpdf2 returns bytes directly
    try:
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            # Older version returns string
            pdf_bytes = pdf_output.encode('latin-1')
        elif isinstance(pdf_output, bytearray):
            # Convert bytearray to bytes for Streamlit compatibility
            pdf_bytes = bytes(pdf_output)
        else:
            # Newer version returns bytes directly
            pdf_bytes = pdf_output
        return pdf_bytes
    except Exception as e:
        # Fallback - create a simple bytes response
        return b"PDF generation error occurred"

def create_downloadable_excel_report(text, prediction_results, text_stats, model_comparison=None):
    """Create downloadable Excel report with multiple sheets"""
    
    # Create Excel writer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # Summary sheet
        summary_data = {
            'Metric': ['Prediction', 'Confidence', 'Human Probability', 'AI Probability'],
            'Value': [
                'AI-Generated' if prediction_results['prediction'] == 1 else 'Human-Written',
                f"{prediction_results['confidence']:.2%}",
                f"{prediction_results['probabilities'][0]:.2%}",
                f"{prediction_results['probabilities'][1]:.2%}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Text statistics sheet
        stats_data = {
            'Statistic': list(text_stats.keys()),
            'Value': [str(v) if not isinstance(v, list) else str(v[:5]) for v in text_stats.values()]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Text_Statistics', index=False)
        
        # Model comparison sheet (if available)
        if model_comparison:
            comparison_df = pd.DataFrame(model_comparison)
            comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
    
    return output.getvalue() 