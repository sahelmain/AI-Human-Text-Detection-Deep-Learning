import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
import io

# Import our utility functions
from utils import (
    extract_text_from_pdf, extract_text_from_docx, 
    extract_text_statistics, analyze_text_features,
    generate_analysis_report, create_downloadable_excel_report
)

# Import joblib with fallback
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

# Try to import PyTorch for deep learning models
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="AI vs Human Text Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-result {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .ai-prediction {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        color: #c62828;
        border: 3px solid #c62828;
    }
    .human-prediction {
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .feature-importance {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .download-section {
        background-color: #d1ecf1;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .stats-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Deep Learning Model Classes (same as before)
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=100, filter_sizes=[3, 4, 5], num_classes=2, dropout=0.5):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        concatenated = torch.cat(conv_outputs, dim=1)
        dropped = self.dropout(concatenated)
        output = self.fc(dropped)
        return output

class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        dropped = self.dropout(final_hidden)
        output = self.fc(dropped)
        return output

class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super(RNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        dropped = self.dropout(final_hidden)
        output = self.fc(dropped)
        return output

@st.cache_resource
def load_models():
    """Load all trained models and vectorizer"""
    models = {}
    model_status = {}
    
    # Try multiple possible paths for models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        'models',
        os.path.join(current_dir, 'models'),
        os.path.join(current_dir, 'ai_human_detection_project', 'models'),
        'ai_human_detection_project/models'
    ]
    
    models_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            models_dir = path
            break
    
    if models_dir is None:
        st.error("Models directory not found")
        return None, None
    
    try:
        # Load TF-IDF vectorizer (critical for ML models and feature importance)
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            models['vectorizer'] = joblib.load(vectorizer_path)
            model_status['vectorizer'] = True
        
        # Load traditional ML models
        ml_models = ['svm_model', 'decision_tree_model', 'adaboost_model']
        for model_name in ml_models:
            model_path = os.path.join(models_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                key = model_name.replace('_model', '')
                models[key] = joblib.load(model_path)
                model_status[key] = True
        
        # Load Deep Learning models if available
        if TORCH_AVAILABLE:
            # Try to load vocabulary mappings
            vocab_to_idx_path = os.path.join(models_dir, 'vocab_to_idx.pkl')
            if os.path.exists(vocab_to_idx_path):
                models['vocab_to_idx'] = joblib.load(vocab_to_idx_path)
                
                # Load model configs
                try:
                    with open(os.path.join(models_dir, 'model_configs.pkl'), 'rb') as f:
                        model_configs = pickle.load(f)
                    models['model_configs'] = model_configs
                    vocab_size = model_configs['vocab_size']
                    
                    # Load deep learning models
                    dl_models = {
                        'cnn': CNNTextClassifier(vocab_size),
                        'lstm': LSTMTextClassifier(vocab_size),
                        'rnn': RNNTextClassifier(vocab_size)
                    }
                    
                    for model_name, model_class in dl_models.items():
                        model_path = os.path.join(models_dir, f'{model_name.upper()}.pkl')
                        if os.path.exists(model_path):
                            try:
                                model_class.load_state_dict(torch.load(model_path, map_location='cpu'))
                                model_class.eval()
                                models[model_name] = model_class
                                model_status[model_name] = True
                            except Exception as e:
                                st.warning(f"Failed to load {model_name.upper()} model: {e}")
                except:
                    pass
        
        return models, model_status
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def preprocess_text_ml(text):
    """Preprocess text for ML models"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def preprocess_text_dl(text, vocab_to_idx, max_seq_length=100):
    """Preprocess text for deep learning models"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    
    indices = []
    for word in words:
        if word in vocab_to_idx:
            indices.append(vocab_to_idx[word])
        else:
            indices.append(vocab_to_idx.get('<UNK>', 1))
    
    if len(indices) < max_seq_length:
        indices.extend([0] * (max_seq_length - len(indices)))
    else:
        indices = indices[:max_seq_length]
    
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

def make_prediction(text, model_name, models):
    """Make prediction using the selected model"""
    try:
        if model_name in ['cnn', 'lstm', 'rnn'] and TORCH_AVAILABLE and 'vocab_to_idx' in models:
            return make_dl_prediction(text, model_name, models)
        else:
            return make_ml_prediction(text, model_name, models)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def make_ml_prediction(text, model_name, models):
    """Make prediction using ML models"""
    processed_text = preprocess_text_ml(text)
    X = models['vectorizer'].transform([processed_text])
    model = models[model_name]
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)
    return prediction, probabilities, confidence

def make_dl_prediction(text, model_name, models):
    """Make prediction using deep learning models"""
    input_tensor = preprocess_text_dl(text, models['vocab_to_idx'])
    model = models[model_name]
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        
    probs = probabilities.numpy()[0]
    confidence = max(probs)
    return prediction, probs, confidence

def create_confidence_chart(probabilities):
    """Create confidence visualization"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Human-Written', 'AI-Generated'],
            y=[probabilities[0], probabilities[1]],
            marker_color=['#17a2b8', '#ffc107'],
            text=[f'{probabilities[0]:.1%}', f'{probabilities[1]:.1%}'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Probability",
        height=400,
        showlegend=False
    )
    
    return fig

def create_text_statistics_chart(stats):
    """Create text statistics visualization"""
    # Create subplots for different metrics
    fig = go.Figure()
    
    # Basic stats
    basic_stats = ['character_count', 'word_count', 'sentence_count', 'paragraph_count']
    basic_values = [stats[key] for key in basic_stats]
    basic_labels = ['Characters', 'Words', 'Sentences', 'Paragraphs']
    
    fig.add_trace(go.Bar(
        x=basic_labels,
        y=basic_values,
        name='Text Counts',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Text Statistics Overview",
        xaxis_title="Metrics",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def create_readability_chart(stats):
    """Create readability scores visualization"""
    readability_metrics = ['flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability_index', 
                          'coleman_liau_index', 'gunning_fog', 'smog_index']
    readability_values = [stats[key] for key in readability_metrics]
    readability_labels = ['Flesch Reading Ease', 'Flesch-Kincaid Grade', 'ARI', 'Coleman-Liau', 'Gunning Fog', 'SMOG']
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=readability_values,
            theta=readability_labels,
            fill='toself',
            name='Readability Scores'
        )
    ])
    
    fig.update_layout(
        title="Readability Analysis",
        height=500
    )
    
    return fig

def create_feature_importance_chart(features):
    """Create feature importance visualization"""
    if 'top_tfidf_features' in features and features['top_tfidf_features']:
        feature_names = [f[0] for f in features['top_tfidf_features'][:10]]
        feature_scores = [f[1] for f in features['top_tfidf_features'][:10]]
        
        fig = go.Figure(data=[
            go.Bar(
                x=feature_scores,
                y=feature_names,
                orientation='h',
                marker_color='orange'
            )
        ])
        
        fig.update_layout(
            title="Top TF-IDF Features (Feature Importance)",
            xaxis_title="TF-IDF Score",
            yaxis_title="Features",
            height=400
        )
        
        return fig
    return None

def create_wordcloud(text):
    """Create word cloud visualization"""
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.title('Word Cloud')
        
        return fig
    except:
        return None

def get_download_link(file_bytes, file_name, file_type):
    """Generate download link for files"""
    b64 = base64.b64encode(file_bytes).decode()
    href = f'<a href="data:{file_type};base64,{b64}" download="{file_name}">📥 Download {file_name}</a>'
    return href 

# Main App
st.markdown('<h1 class="main-header">🤖 AI vs Human Text Detection System</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Text Analysis", "📁 File Upload", "⚖️ Model Comparison", "📊 Model Performance", "📈 Advanced Analytics"],
    index=0
)

# Load models
models, model_status = load_models()

if models is None:
    st.error("Failed to load models. Please check that model files are present.")
    st.stop()

# Display available models in sidebar
st.sidebar.markdown("### 🤖 Available Models")
available_models = [k for k in models.keys() if k not in ['vectorizer', 'vocab_to_idx', 'model_configs']]
for model in available_models:
    st.sidebar.success(f"✅ {model.upper()}")

# HOME PAGE
if page == "🏠 Home":
    st.markdown("""
    ### Welcome to the Complete AI vs Human Text Detection System
    
    This comprehensive application uses both traditional machine learning and cutting-edge deep learning 
    to distinguish between AI-generated and human-written text with high accuracy.
    
    #### 🌟 **Key Features:**
    - **📝 Multiple Input Methods**: Type text directly, paste content, or upload PDF/Word documents
    - **🤖 6 Advanced Models**: Choose from SVM, Decision Tree, AdaBoost, CNN, LSTM, and RNN
    - **📊 Real-time Predictions**: Instant AI vs Human classification with confidence scores
    - **📈 Advanced Visualizations**: Feature importance, text statistics, readability analysis
    - **⚖️ Model Comparison**: Side-by-side performance analysis across all models
    - **📋 Comprehensive Reports**: Download detailed PDF and Excel reports
    """)
    
    # Enhanced model performance overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧠 **Deep Learning Models**")
        dl_data = {
            'Model': ['CNN', 'LSTM', 'RNN'],
            'Accuracy': ['97.33%', '94.52%', '82.75%'],
            'Strengths': ['Best Performance', 'Sequential Analysis', 'Basic RNN']
        }
        st.dataframe(pd.DataFrame(dl_data), use_container_width=True)
    
    with col2:
        st.markdown("#### ⚙️ **Traditional ML Models**")
        ml_data = {
            'Model': ['SVM', 'Decision Tree', 'AdaBoost'],
            'Accuracy': ['96.38%', '84.99%', '85.50%'],
            'Strengths': ['Feature-based', 'Interpretable', 'Ensemble']
        }
        st.dataframe(pd.DataFrame(ml_data), use_container_width=True)
    
    # Quick start guide
    st.markdown("""
    ### 🚀 **Quick Start Guide:**
    1. **Text Analysis**: Paste or type text for quick analysis
    2. **File Upload**: Upload PDF or Word documents for processing
    3. **Model Comparison**: Compare results across all 6 models
    4. **Advanced Analytics**: Deep dive into text statistics and features
    5. **Download Reports**: Get comprehensive analysis reports
    """)

# TEXT ANALYSIS PAGE (Enhanced)
elif page == "🔮 Text Analysis":
    st.markdown("### 📝 Individual Text Analysis")
    
    # Model selection with enhanced information
    model_options = available_models
    model_descriptions = {
        'svm': '🎯 SVM (96.38%) - Support Vector Machine with high accuracy',
        'decision_tree': '🌳 Decision Tree (84.99%) - Most interpretable model',
        'adaboost': '🚀 AdaBoost (85.50%) - Ensemble boosting method',
        'cnn': '🧠 CNN (97.33%) - Best performing deep learning model',
        'lstm': '🔄 LSTM (94.52%) - Sequential pattern recognition',
        'rnn': '⚡ RNN (82.75%) - Basic recurrent neural network'
    }
    
    model_choice = st.selectbox(
        "Choose Model:",
        model_options,
        format_func=lambda x: model_descriptions.get(x, x)
    )
    
    # Text input with enhanced interface
    st.markdown("#### 📝 Enter Your Text")
    text_input = st.text_area(
        "Text to analyze:",
        height=200,
        placeholder="Paste or type the text you want to analyze...",
        help="Enter any text to check if it was written by AI or human"
    )
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        show_statistics = st.checkbox("📊 Show Text Statistics", value=True)
        show_features = st.checkbox("🔍 Show Feature Analysis", value=True)
    with col2:
        show_wordcloud = st.checkbox("☁️ Generate Word Cloud", value=False)
        enable_download = st.checkbox("📥 Enable Report Download", value=True)
    
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if text_input.strip():
            with st.spinner("🔄 Analyzing text..."):
                # Make prediction
                prediction, probabilities, confidence = make_prediction(text_input, model_choice, models)
                
                if prediction is not None:
                    # Display main prediction result
                    if prediction == 1:  # AI-generated
                        st.markdown(
                            f'<div class="prediction-result ai-prediction">🤖 AI-Generated Text Detected<br/>Confidence: {confidence:.2%}</div>',
                            unsafe_allow_html=True
                        )
                    else:  # Human-written
                        st.markdown(
                            f'<div class="prediction-result human-prediction">👤 Human-Written Text Detected<br/>Confidence: {confidence:.2%}</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Enhanced results display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📈 Probability Scores")
                        st.metric("Human Probability", f"{probabilities[0]:.2%}")
                        st.metric("AI Probability", f"{probabilities[1]:.2%}")
                        st.metric("Confidence Score", f"{confidence:.2%}")
                    
                    with col2:
                        st.markdown("#### 📊 Confidence Visualization")
                        confidence_chart = create_confidence_chart(probabilities)
                        st.plotly_chart(confidence_chart, use_container_width=True)
                    
                    # Text Statistics Section
                    if show_statistics:
                        st.markdown("---")
                        st.markdown("#### 📊 Text Statistics")
                        text_stats = extract_text_statistics(text_input)
                        
                        # Basic statistics
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        with stat_col1:
                            st.metric("Characters", text_stats['character_count'])
                        with stat_col2:
                            st.metric("Words", text_stats['word_count'])
                        with stat_col3:
                            st.metric("Sentences", text_stats['sentence_count'])
                        with stat_col4:
                            st.metric("Paragraphs", text_stats['paragraph_count'])
                        
                        # Advanced statistics
                        adv_col1, adv_col2 = st.columns(2)
                        with adv_col1:
                            st.plotly_chart(create_text_statistics_chart(text_stats), use_container_width=True)
                        with adv_col2:
                            st.plotly_chart(create_readability_chart(text_stats), use_container_width=True)
                    
                    # Feature Analysis Section
                    if show_features:
                        st.markdown("---")
                        st.markdown("#### 🔍 Feature Analysis")
                        vectorizer = models.get('vectorizer')
                        features = analyze_text_features(text_input, vectorizer)
                        
                        feature_col1, feature_col2 = st.columns(2)
                        with feature_col1:
                            st.markdown("##### 📋 Linguistic Features")
                            st.metric("Average Word Length", f"{features['avg_word_length']:.2f}")
                            st.metric("Average Sentence Length", f"{features['avg_sentence_length']:.2f}")
                            st.metric("Lexical Diversity", f"{features['lexical_diversity']:.3f}")
                            st.metric("Function Word Ratio", f"{features['function_word_ratio']:.3f}")
                        
                        with feature_col2:
                            feature_importance_chart = create_feature_importance_chart(features)
                            if feature_importance_chart:
                                st.plotly_chart(feature_importance_chart, use_container_width=True)
                    
                    # Word Cloud
                    if show_wordcloud:
                        st.markdown("---")
                        st.markdown("#### ☁️ Word Cloud")
                        wordcloud_fig = create_wordcloud(text_input)
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig)
                    
                    # Download Reports Section
                    if enable_download:
                        st.markdown("---")
                        st.markdown('<div class="download-section">', unsafe_allow_html=True)
                        st.markdown("#### 📥 Download Comprehensive Reports")
                        
                        download_col1, download_col2 = st.columns(2)
                        
                        with download_col1:
                            # Direct PDF download - single step
                            try:
                                prediction_results = {
                                    'prediction': prediction,
                                    'probabilities': probabilities,
                                    'confidence': confidence
                                }
                                pdf_bytes = generate_analysis_report(text_input, prediction_results, text_stats)
                                
                                st.download_button(
                                    label="📄 Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"ai_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                    help="Click to download comprehensive PDF analysis report"
                                )
                            except Exception as e:
                                st.error(f"Error generating PDF report: {str(e)}")
                        
                        with download_col2:
                            # Direct Excel download - single step
                            try:
                                prediction_results = {
                                    'prediction': prediction,
                                    'probabilities': probabilities,
                                    'confidence': confidence
                                }
                                excel_bytes = create_downloadable_excel_report(text_input, prediction_results, text_stats)
                                
                                st.download_button(
                                    label="📊 Download Excel Report",
                                    data=excel_bytes,
                                    file_name=f"ai_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True,
                                    help="Click to download detailed Excel analysis report"
                                )
                            except Exception as e:
                                st.error(f"Error generating Excel report: {str(e)}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# FILE UPLOAD PAGE (New Feature)
elif page == "📁 File Upload":
    st.markdown("### 📁 Document Upload & Analysis")
    st.markdown("Upload PDF or Word documents for AI vs Human text detection analysis.")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, Word documents (.docx), and plain text files (.txt)"
    )
    
    if uploaded_file is not None:
        # Display file information
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size} bytes",
            "File type": uploaded_file.type
        }
        
        st.markdown("#### 📋 File Information")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        # Extract text based on file type
        with st.spinner("📖 Extracting text from document..."):
            if uploaded_file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                extracted_text = extract_text_from_docx(uploaded_file)
            else:  # txt file
                extracted_text = str(uploaded_file.read(), "utf-8")
        
        if extracted_text and not extracted_text.startswith("Error"):
            st.success(f"✅ Successfully extracted {len(extracted_text)} characters from the document.")
            
            # Show text preview
            st.markdown("#### 👀 Text Preview")
            preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
            st.text_area("Extracted Text (Preview)", preview_text, height=150, disabled=True)
            
            # Model selection for file analysis
            st.markdown("#### 🤖 Choose Analysis Model")
            
            # Model descriptions for file upload section
            model_descriptions = {
                'svm': '🎯 SVM (96.38%) - Support Vector Machine with high accuracy',
                'decision_tree': '🌳 Decision Tree (84.99%) - Most interpretable model',
                'adaboost': '🚀 AdaBoost (85.50%) - Ensemble boosting method',
                'cnn': '🧠 CNN (97.33%) - Best performing deep learning model',
                'lstm': '🔄 LSTM (94.52%) - Sequential pattern recognition',
                'rnn': '⚡ RNN (82.75%) - Basic recurrent neural network'
            }
            
            file_model_choice = st.selectbox(
                "Select model for analysis:",
                available_models,
                format_func=lambda x: model_descriptions.get(x, x),
                key="file_model"
            )
            
            # Analysis button
            if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing document..."):
                    # Make prediction
                    prediction, probabilities, confidence = make_prediction(extracted_text, file_model_choice, models)
                    
                    if prediction is not None:
                        # Display results
                        if prediction == 1:  # AI-generated
                            st.markdown(
                                f'<div class="prediction-result ai-prediction">🤖 Document contains AI-Generated Text<br/>Confidence: {confidence:.2%}</div>',
                                unsafe_allow_html=True
                            )
                        else:  # Human-written
                            st.markdown(
                                f'<div class="prediction-result human-prediction">👤 Document contains Human-Written Text<br/>Confidence: {confidence:.2%}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Detailed analysis
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            st.markdown("#### 📊 Analysis Results")
                            st.metric("Human Probability", f"{probabilities[0]:.2%}")
                            st.metric("AI Probability", f"{probabilities[1]:.2%}")
                            st.metric("Overall Confidence", f"{confidence:.2%}")
                        
                        with analysis_col2:
                            confidence_chart = create_confidence_chart(probabilities)
                            st.plotly_chart(confidence_chart, use_container_width=True)
                        
                        # Document statistics
                        st.markdown("---")
                        st.markdown("#### 📈 Document Statistics")
                        doc_stats = extract_text_statistics(extracted_text)
                        
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        with stat_col1:
                            st.metric("Total Characters", doc_stats['character_count'])
                        with stat_col2:
                            st.metric("Total Words", doc_stats['word_count'])
                        with stat_col3:
                            st.metric("Total Sentences", doc_stats['sentence_count'])
                        with stat_col4:
                            st.metric("Reading Level", f"{doc_stats['flesch_kincaid_grade']:.1f}")
                        
                        # Download section for file analysis
                        st.markdown("---")
                        st.markdown("#### 📥 Download Document Analysis Report")
                        
                        prediction_results = {
                            'prediction': prediction,
                            'probabilities': probabilities,
                            'confidence': confidence
                        }
                        
                        report_col1, report_col2 = st.columns(2)
                        
                        with report_col1:
                            # Direct PDF download for file analysis
                            try:
                                pdf_bytes = generate_analysis_report(extracted_text, prediction_results, doc_stats)
                                st.download_button(
                                    label="📄 Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"document_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                    help="Download comprehensive PDF analysis of the document"
                                )
                            except Exception as e:
                                st.error(f"Error generating PDF report: {str(e)}")
                        
                        with report_col2:
                            # Direct Excel download for file analysis
                            try:
                                excel_bytes = create_downloadable_excel_report(extracted_text, prediction_results, doc_stats)
                                st.download_button(
                                    label="📊 Download Excel Report",
                                    data=excel_bytes,
                                    file_name=f"document_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True,
                                    help="Download detailed Excel analysis of the document"
                                )
                            except Exception as e:
                                st.error(f"Error generating Excel report: {str(e)}")
        else:
            st.error("❌ Failed to extract text from the document. Please check the file format and try again.")

# MODEL COMPARISON PAGE (Enhanced)
elif page == "⚖️ Model Comparison":
    st.markdown("### ⚖️ Comprehensive Model Comparison")
    st.markdown("Compare the performance of all available models on the same text input.")
    
    # Input methods
    input_method = st.radio("Choose input method:", ["✍️ Type/Paste Text", "📁 Upload File"])
    
    comparison_text = ""
    
    if input_method == "✍️ Type/Paste Text":
        comparison_text = st.text_area(
            "Enter text to compare across all models:",
            height=200,
            placeholder="Enter the text you want to analyze with all models..."
        )
    else:
        uploaded_comparison_file = st.file_uploader(
            "Upload file for comparison",
            type=['pdf', 'docx', 'txt'],
            key="comparison_upload"
        )
        
        if uploaded_comparison_file:
            with st.spinner("Extracting text..."):
                if uploaded_comparison_file.type == "application/pdf":
                    comparison_text = extract_text_from_pdf(uploaded_comparison_file)
                elif uploaded_comparison_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    comparison_text = extract_text_from_docx(uploaded_comparison_file)
                else:
                    comparison_text = str(uploaded_comparison_file.read(), "utf-8")
                
                if comparison_text and not comparison_text.startswith("Error"):
                    st.success(f"✅ Text extracted successfully ({len(comparison_text)} characters)")
                    st.text_area("Extracted text preview:", comparison_text[:300] + "...", height=100, disabled=True)
    
    if st.button("🔀 Compare All Models", type="primary", use_container_width=True):
        if comparison_text.strip():
            with st.spinner("🔄 Running all available models..."):
                model_display_names = {
                    'svm': 'SVM (Traditional ML)',
                    'decision_tree': 'Decision Tree (Traditional ML)',
                    'adaboost': 'AdaBoost (Traditional ML)',
                    'cnn': 'CNN (Deep Learning)',
                    'lstm': 'LSTM (Deep Learning)',
                    'rnn': 'RNN (Deep Learning)'
                }
                
                results = []
                detailed_results = {}
                
                for model_name in available_models:
                    try:
                        pred, probs, conf = make_prediction(comparison_text, model_name, models)
                        if pred is not None:
                            pred_label = "AI-Generated" if pred == 1 else "Human-Written"
                            results.append({
                                'Model': model_display_names.get(model_name, model_name),
                                'Prediction': pred_label,
                                'Confidence': f"{conf:.2%}",
                                'Human Probability': f"{probs[0]:.2%}",
                                'AI Probability': f"{probs[1]:.2%}",
                                'Model Type': 'Deep Learning' if model_name in ['cnn', 'lstm', 'rnn'] else 'Traditional ML'
                            })
                            detailed_results[model_name] = {'prediction': pred, 'probabilities': probs, 'confidence': conf}
                    except Exception as e:
                        st.warning(f"⚠️ Model {model_name} failed: {str(e)}")
                
                if results:
                    # Results table
                    st.markdown("#### 📊 Comparison Results")
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Visual comparison
                    st.markdown("#### 📈 Visual Comparison")
                    
                    # Confidence comparison chart
                    model_names = [r['Model'] for r in results]
                    confidences = [float(r['Confidence'].replace('%', '')) for r in results]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=model_names,
                            y=confidences,
                            text=[f'{c}%' for c in confidences],
                            textposition='auto',
                            marker_color=['#FF6B6B' if 'Deep Learning' in r['Model Type'] else '#4ECDC4' for r in results]
                        )
                    ])
                    
                    fig.update_layout(
                        title="Model Confidence Comparison",
                        xaxis_title="Models",
                        yaxis_title="Confidence (%)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction agreement analysis
                    st.markdown("#### 🤝 Model Agreement Analysis")
                    ai_predictions = sum(1 for r in results if r['Prediction'] == 'AI-Generated')
                    human_predictions = len(results) - ai_predictions
                    
                    agreement_col1, agreement_col2, agreement_col3 = st.columns(3)
                    with agreement_col1:
                        st.metric("AI Predictions", ai_predictions)
                    with agreement_col2:
                        st.metric("Human Predictions", human_predictions)
                    with agreement_col3:
                        consensus = "Strong" if abs(ai_predictions - human_predictions) >= len(results) * 0.6 else "Weak"
                        st.metric("Consensus", consensus)
                    
                    # Download comparison report
                    st.markdown("---")
                    st.markdown("#### 📥 Download Comparison Report")
                    
                    # Direct download for model comparison
                    try:
                        text_stats = extract_text_statistics(comparison_text)
                        # Use the first model's results as primary for report generation
                        primary_result = list(detailed_results.values())[0]
                        prediction_results = {
                            'prediction': primary_result['prediction'],
                            'probabilities': primary_result['probabilities'],
                            'confidence': primary_result['confidence']
                        }
                        
                        excel_bytes = create_downloadable_excel_report(comparison_text, prediction_results, text_stats, results)
                        st.download_button(
                            label="📊 Download Comparison Report",
                            data=excel_bytes,
                            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            help="Download comprehensive comparison report of all models"
                        )
                    except Exception as e:
                        st.error(f"Error generating comparison report: {str(e)}")
                else:
                    st.error("❌ No models were able to make predictions")
        else:
            st.warning("⚠️ Please enter some text or upload a file to compare.")

# MODEL PERFORMANCE PAGE (Enhanced)
elif page == "📊 Model Performance":
    st.markdown("### 📊 Model Performance Metrics & Analysis")
    
    # Performance data with more details
    performance_data = {
        'Model': ['CNN (DL)', 'LSTM (DL)', 'RNN (DL)', 'SVM (ML)', 'Decision Tree (ML)', 'AdaBoost (ML)'],
        'Accuracy': [97.33, 94.52, 82.75, 96.38, 84.99, 85.50],
        'Precision': [97.45, 94.68, 83.12, 96.42, 85.15, 85.68],
        'Recall': [97.21, 94.36, 82.38, 96.34, 84.83, 85.32],
        'F1-Score': [97.33, 94.52, 82.75, 96.38, 84.99, 85.50],
        'Type': ['Deep Learning', 'Deep Learning', 'Deep Learning', 'Traditional ML', 'Traditional ML', 'Traditional ML'],
        'Training Time': ['45 min', '52 min', '28 min', '8 min', '3 min', '12 min'],
        'Prediction Speed': ['Fast', 'Medium', 'Fast', 'Very Fast', 'Instant', 'Fast']
    }
    
    df = pd.DataFrame(performance_data)
    
    # Display performance table
    st.markdown("#### 📋 Detailed Performance Metrics")
    st.dataframe(df, use_container_width=True)
    
    # Performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig1 = go.Figure()
        
        dl_models = df[df['Type'] == 'Deep Learning']
        ml_models = df[df['Type'] == 'Traditional ML']
        
        fig1.add_trace(go.Bar(
            name='Deep Learning',
            x=dl_models['Model'],
            y=dl_models['Accuracy'],
            marker_color='#FF6B6B',
            text=[f'{acc}%' for acc in dl_models['Accuracy']],
            textposition='auto'
        ))
        
        fig1.add_trace(go.Bar(
            name='Traditional ML',
            x=ml_models['Model'],
            y=ml_models['Accuracy'],
            marker_color='#4ECDC4',
            text=[f'{acc}%' for acc in ml_models['Accuracy']],
            textposition='auto'
        ))
        
        fig1.update_layout(
            title="Model Accuracy Comparison",
            yaxis_title="Accuracy (%)",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Multi-metric radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig2 = go.Figure()
        
        # Add trace for each model
        colors = ['#FF6B6B', '#FF8E8E', '#FFB1B1', '#4ECDC4', '#70D7D4', '#94E2DF']
        for i, model in enumerate(df['Model']):
            values = [df.iloc[i][metric] for metric in metrics]
            fig2.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model,
                line_color=colors[i]
            ))
        
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[75, 100]
                )),
            showlegend=True,
            title="Multi-Metric Performance Comparison",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Model recommendations
    st.markdown("---")
    st.markdown("#### 🏆 Model Recommendations")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("""
        **🥇 Best Overall Performance**
        - **CNN**: 97.33% accuracy
        - Best for general use cases
        - Excellent balance of speed and accuracy
        """)
    
    with rec_col2:
        st.markdown("""
        **⚡ Fastest Processing**
        - **Decision Tree**: Instant predictions
        - Good for real-time applications
        - Most interpretable results
        """)
    
    with rec_col3:
        st.markdown("""
        **🎯 Most Reliable**
        - **SVM**: Consistent 96.38% accuracy
        - Robust traditional ML approach
        - Good feature interpretation
        """)

# ADVANCED ANALYTICS PAGE (New Feature)
elif page == "📈 Advanced Analytics":
    st.markdown("### 📈 Advanced Text Analytics & Insights")
    st.markdown("Deep dive into text characteristics and AI detection patterns.")
    
    # Input section
    analytics_text = st.text_area(
        "Enter text for advanced analysis:",
        height=200,
        placeholder="Enter text to perform comprehensive linguistic and statistical analysis..."
    )
    
    if st.button("🔬 Perform Advanced Analysis", type="primary"):
        if analytics_text.strip():
            with st.spinner("🔄 Performing comprehensive analysis..."):
                # Get comprehensive statistics
                text_stats = extract_text_statistics(analytics_text)
                features = analyze_text_features(analytics_text, models.get('vectorizer'))
                
                # Run prediction with best model (CNN if available, otherwise SVM)
                best_model = 'cnn' if 'cnn' in models else 'svm'
                prediction, probabilities, confidence = make_prediction(analytics_text, best_model, models)
                
                # Display main insights
                st.markdown("#### 🎯 Key Insights")
                
                insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
                
                with insight_col1:
                    complexity_score = (text_stats['flesch_reading_ease'] + text_stats['automated_readability_index']) / 2
                    st.metric("Text Complexity", f"{complexity_score:.1f}", help="Based on readability metrics")
                
                with insight_col2:
                    ai_likelihood = "High" if probabilities[1] > 0.7 else "Medium" if probabilities[1] > 0.3 else "Low"
                    st.metric("AI Likelihood", ai_likelihood, f"{probabilities[1]:.1%}")
                
                with insight_col3:
                    writing_style = "Formal" if features['function_word_ratio'] < 0.4 else "Conversational"
                    st.metric("Writing Style", writing_style)
                
                with insight_col4:
                    vocabulary_richness = "Rich" if features['lexical_diversity'] > 0.6 else "Standard"
                    st.metric("Vocabulary", vocabulary_richness, f"{features['lexical_diversity']:.3f}")
                
                # Detailed analytics sections
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistical Analysis", "🔍 Linguistic Features", "📈 Readability Analysis", "🎨 Visual Analysis"])
                
                with tab1:
                    st.markdown("##### 📊 Comprehensive Text Statistics")
                    
                    # Create detailed statistics dataframe
                    stats_df = pd.DataFrame([
                        {"Metric": "Character Count", "Value": text_stats['character_count'], "Description": "Total characters including spaces"},
                        {"Metric": "Word Count", "Value": text_stats['word_count'], "Description": "Total number of words"},
                        {"Metric": "Sentence Count", "Value": text_stats['sentence_count'], "Description": "Total number of sentences"},
                        {"Metric": "Paragraph Count", "Value": text_stats['paragraph_count'], "Description": "Total number of paragraphs"},
                        {"Metric": "Avg Word Length", "Value": f"{text_stats['avg_word_length']:.2f}", "Description": "Average characters per word"},
                        {"Metric": "Avg Sentence Length", "Value": f"{text_stats['avg_sentence_length']:.2f}", "Description": "Average words per sentence"},
                        {"Metric": "Lexical Diversity", "Value": f"{text_stats['lexical_diversity']:.3f}", "Description": "Ratio of unique words to total words"},
                        {"Metric": "Punctuation Ratio", "Value": f"{text_stats['punctuation_ratio']:.3f}", "Description": "Punctuation marks per character"}
                    ])
                    
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Most common words
                    if text_stats['most_common_words']:
                        st.markdown("##### 📝 Most Frequent Words")
                        common_words_df = pd.DataFrame(text_stats['most_common_words'], columns=['Word', 'Frequency'])
                        
                        fig = go.Figure(data=[
                            go.Bar(x=common_words_df['Word'], y=common_words_df['Frequency'], marker_color='lightcoral')
                        ])
                        fig.update_layout(title="Top 10 Most Frequent Words", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.markdown("##### 🔍 Advanced Linguistic Features")
                    
                    feature_metrics = {
                        "Stylistic Features": {
                            "Function Word Ratio": f"{features['function_word_ratio']:.3f}",
                            "Punctuation Density": f"{features['punctuation_ratio']:.3f}",
                            "Uppercase Ratio": f"{features['uppercase_ratio']:.3f}",
                            "Digit Ratio": f"{features['digit_ratio']:.3f}"
                        },
                        "Complexity Indicators": {
                            "Average Word Length": f"{features['avg_word_length']:.2f} chars",
                            "Average Sentence Length": f"{features['avg_sentence_length']:.2f} words",
                            "Vocabulary Richness": f"{features['lexical_diversity']:.3f}",
                            "Character Density": f"{features['char_count'] / max(1, features['word_count']):.2f} chars/word"
                        }
                    }
                    
                    feat_col1, feat_col2 = st.columns(2)
                    
                    with feat_col1:
                        st.markdown("**Stylistic Features**")
                        for feature, value in feature_metrics["Stylistic Features"].items():
                            st.metric(feature, value)
                    
                    with feat_col2:
                        st.markdown("**Complexity Indicators**")
                        for feature, value in feature_metrics["Complexity Indicators"].items():
                            st.metric(feature, value)
                
                with tab3:
                    st.markdown("##### 📈 Readability & Complexity Analysis")
                    
                    # Readability scores interpretation
                    readability_scores = {
                        "Flesch Reading Ease": {
                            "score": text_stats['flesch_reading_ease'],
                            "interpretation": "Very Easy" if text_stats['flesch_reading_ease'] > 90 else 
                                           "Easy" if text_stats['flesch_reading_ease'] > 80 else
                                           "Fairly Easy" if text_stats['flesch_reading_ease'] > 70 else
                                           "Standard" if text_stats['flesch_reading_ease'] > 60 else
                                           "Fairly Difficult" if text_stats['flesch_reading_ease'] > 50 else
                                           "Difficult"
                        },
                        "Flesch-Kincaid Grade": {
                            "score": text_stats['flesch_kincaid_grade'],
                            "interpretation": f"Grade {text_stats['flesch_kincaid_grade']:.1f} level"
                        },
                        "Gunning Fog Index": {
                            "score": text_stats['gunning_fog'],
                            "interpretation": f"Grade {text_stats['gunning_fog']:.1f} level"
                        }
                    }
                    
                    for metric, data in readability_scores.items():
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(metric, f"{data['score']:.2f}")
                        with col2:
                            st.info(f"**Interpretation:** {data['interpretation']}")
                    
                    # Readability radar chart
                    st.plotly_chart(create_readability_chart(text_stats), use_container_width=True)
                
                with tab4:
                    st.markdown("##### 🎨 Visual Text Analysis")
                    
                    visual_col1, visual_col2 = st.columns(2)
                    
                    with visual_col1:
                        # Word cloud
                        st.markdown("**Word Cloud Visualization**")
                        wordcloud_fig = create_wordcloud(analytics_text)
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig)
                    
                    with visual_col2:
                        # Feature importance (if available)
                        feature_chart = create_feature_importance_chart(features)
                        if feature_chart:
                            st.plotly_chart(feature_chart, use_container_width=True)
                        else:
                            st.info("Feature importance analysis requires TF-IDF vectorizer")
                
                # AI Detection Summary
                st.markdown("---")
                st.markdown("#### 🤖 AI Detection Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    prediction_label = "AI-Generated" if prediction == 1 else "Human-Written"
                    st.markdown(f"""
                    **🎯 Final Prediction:** {prediction_label}  
                    **🔒 Confidence:** {confidence:.2%}  
                    **🤖 AI Probability:** {probabilities[1]:.2%}  
                    **👤 Human Probability:** {probabilities[0]:.2%}  
                    **🧠 Model Used:** {best_model.upper()}
                    """)
                
                with summary_col2:
                    confidence_chart = create_confidence_chart(probabilities)
                    st.plotly_chart(confidence_chart, use_container_width=True)
        else:
            st.warning("⚠️ Please enter some text for analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🤖 AI vs Human Text Detection System</strong></p>
    <p>Enhanced with Deep Learning | Built with Streamlit | Comprehensive Analysis & Reporting</p>
    <p>Features: File Upload • Advanced Analytics • Model Comparison • Report Generation</p>
</div>
""", unsafe_allow_html=True) 