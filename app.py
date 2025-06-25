import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import joblib

# Suppress warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="AI vs Human Text Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .ai-prediction {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .human-prediction {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models and vectorizer"""
    models = {}
    model_status = {}
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    
    # Debug: Show what directory we're looking in
    st.write(f"Looking for models in: {models_dir}")
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Files in models directory: {os.listdir(models_dir) if os.path.exists(models_dir) else 'Directory not found'}")
    
    try:
        # Load TF-IDF vectorizer
        vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        models["vectorizer"] = joblib.load(vectorizer_path)
        model_status["vectorizer"] = True
        
        # Load SVM model
        svm_path = os.path.join(models_dir, "svm_model.pkl")
        models["svm"] = joblib.load(svm_path)
        model_status["svm"] = True
        
        # Load Decision Tree model
        dt_path = os.path.join(models_dir, "decision_tree_model.pkl")
        models["decision_tree"] = joblib.load(dt_path)
        model_status["decision_tree"] = True
        
        # Load AdaBoost model
        ada_path = os.path.join(models_dir, "adaboost_model.pkl")
        models["adaboost"] = joblib.load(ada_path)
        model_status["adaboost"] = True
        
        return models, model_status
        
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure all model files are in the 'models/' directory")
        st.error(f"Looking in: {models_dir}")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error loading models: {e}")
        return None, None

def preprocess_text(text):
    """Preprocess text similar to training pipeline"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text

def make_prediction(text, model_name, models):
    """Make prediction using the selected model"""
    if models is None:
        return None, None, None
    
    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Transform text using TF-IDF vectorizer
        X = models["vectorizer"].transform([processed_text])
        
        # Make prediction with selected model
        model = models[model_name]
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Get confidence score (probability of predicted class)
        confidence = max(probabilities)
        
        return prediction, probabilities, confidence
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def create_confidence_chart(probabilities):
    """Create a confidence visualization"""
    fig = go.Figure(data=[
        go.Bar(
            x=["Human-Written", "AI-Generated"],
            y=[probabilities[0], probabilities[1]],
            marker_color=["#17a2b8", "#ffc107"],
            text=[f"{probabilities[0]:.1%}", f"{probabilities[1]:.1%}"],
            textposition="auto",
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Probability",
        height=400,
        showlegend=False
    )
    
    return fig

# Main App
st.markdown("<h1 class="main-header">🤖 AI vs Human Text Detection</h1>", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Text Analysis", "⚖️ Model Comparison", "📊 Model Performance"],
    index=0
)

# Load models
models, model_status = load_models()

if models is None:
    st.error("Failed to load models. Please check that all model files are present in the 'models/' directory.")
    st.stop()

# HOME PAGE
if page == "🏠 Home":
    st.markdown("""
    ### Welcome to AI vs Human Text Detection System
    
    This application uses machine learning to distinguish between AI-generated and human-written text 
    with high accuracy. Our system employs three different algorithms to provide comprehensive analysis.
    """)
    
    # Model performance overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("SVM Model", "96.38%", "Accuracy")
        st.info("Support Vector Machine - Best overall performance")
    
    with col2:
        st.metric("Decision Tree", "84.99%", "Accuracy")
        st.info("Most interpretable model")
    
    with col3:
        st.metric("AdaBoost", "85.50%", "Accuracy")
        st.info("Ensemble method")

# SINGLE TEXT ANALYSIS
elif page == "🔮 Single Text Analysis":
    st.markdown("### Analyze Individual Text")
    
    # Model selection
    model_choice = st.selectbox(
        "Choose Model:",
        ["svm", "decision_tree", "adaboost"],
        format_func=lambda x: {
            "svm": "🎯 SVM (Recommended)",
            "decision_tree": "🌳 Decision Tree",
            "adaboost": "🚀 AdaBoost"
        }[x]
    )
    
    # Text input
    text_input = st.text_area(
        "Enter your text here:",
        height=200,
        placeholder="Paste or type the text you want to analyze..."
    )
    
    if st.button("🔍 Analyze Text", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing text..."):
                prediction, probabilities, confidence = make_prediction(text_input, model_choice, models)
                
                if prediction is not None:
                    # Display prediction
                    if prediction == 1:  # AI-generated
                        st.markdown(
                            f"<div class="prediction-result ai-prediction">🤖 AI-Generated Text Detected<br/>Confidence: {confidence:.2%}</div>",
                            unsafe_allow_html=True
                        )
                    else:  # Human-written
                        st.markdown(
                            f"<div class="prediction-result human-prediction">👤 Human-Written Text Detected<br/>Confidence: {confidence:.2%}</div>",
                            unsafe_allow_html=True
                        )
                    
                    # Detailed results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Human Probability", f"{probabilities[0]:.2%}")
                        st.metric("AI Probability", f"{probabilities[1]:.2%}")
                    
                    with col2:
                        confidence_chart = create_confidence_chart(probabilities)
                        st.plotly_chart(confidence_chart, use_container_width=True)
        else:
            st.warning("Please enter some text to analyze.")

# MODEL COMPARISON
elif page == "⚖️ Model Comparison":
    st.markdown("### Compare All Models")
    
    comparison_text = st.text_area(
        "Enter text to compare across models:",
        height=200,
        placeholder="Enter the text you want to analyze with all three models..."
    )
    
    if st.button("🔀 Compare Models", type="primary"):
        if comparison_text.strip():
            with st.spinner("Running all models..."):
                model_names = ["svm", "decision_tree", "adaboost"]
                model_labels = ["SVM", "Decision Tree", "AdaBoost"]
                
                results = []
                for i, model_name in enumerate(model_names):
                    pred, probs, conf = make_prediction(comparison_text, model_name, models)
                    if pred is not None:
                        pred_label = "AI-Generated" if pred == 1 else "Human-Written"
                        results.append({
                            "Model": model_labels[i],
                            "Prediction": pred_label,
                            "Confidence": f"{conf:.2%}",
                            "Human Prob": f"{probs[0]:.2%}",
                            "AI Prob": f"{probs[1]:.2%}"
                        })
                
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
        else:
            st.warning("Please enter some text to compare.")

# MODEL PERFORMANCE
elif page == "�� Model Performance":
    st.markdown("### Model Performance Metrics")
    
    performance_data = {
        "Model": ["SVM", "Decision Tree", "AdaBoost"],
        "Accuracy": [96.38, 84.99, 85.50],
        "Precision": [96.38, 85.02, 85.45],
        "Recall": [96.38, 84.99, 85.50],
        "F1-Score": [96.38, 84.98, 85.47]
    }
    
    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True)
    
    # Create accuracy chart
    fig = go.Figure(data=[
        go.Bar(
            x=performance_data["Model"],
            y=performance_data["Accuracy"],
            text=[f"{acc}%" for acc in performance_data["Accuracy"]],
            textposition="auto",
            marker_color=["#4ecdc4", "#ffc107", "#ff6b6b"]
        )
    ])
    
    fig.update_layout(
        title="Model Accuracy Comparison",
        yaxis_title="Accuracy (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
