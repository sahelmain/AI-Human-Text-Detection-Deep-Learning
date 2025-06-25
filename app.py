# AI vs Human Text Detection App
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re

# Page Configuration
st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🤖📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    :root {
        --bg-color-dark: #f8fafc;
        --bg-color-light: #e2e8f0;
        --primary-color: #3b82f6;
        --secondary-color: #6366f1;
        --accent-color: #ec4899;
        --text-color: #1e293b;
        --card-bg-color: rgba(255, 255, 255, 0.8);
        --card-border-color: rgba(148, 163, 184, 0.3);
        --font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-color-dark) 0%, var(--bg-color-light) 100%);
        font-family: var(--font-family);
        color: var(--text-color);
    }
    
    .main-header {
        font-size: 3rem;
        color: #1e293b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
    }
    
    .prediction-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border-color);
        animation: slideInUp 0.5s ease-out;
    }

    .ai-prediction {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(236, 72, 153, 0.2) 100%);
        border-left: 5px solid var(--accent-color);
        color: #831843;
    }
    
    .human-prediction {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(99, 102, 241, 0.2) 100%);
        border-left: 5px solid var(--secondary-color);
        color: #312e81;
    }
    
    .model-card {
        background: var(--card-bg-color);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        border: 1px solid var(--card-border-color);
        margin: 1rem 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        height: 100%;
    }
    
    .model-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .metric-card {
        background: var(--card-bg-color);
        color: var(--text-color);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
        border: 1px solid var(--card-border-color);
        transition: all 0.3s ease;
        text-align: center;
        height: 100%;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.95);
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.15);
    }
    
    .confidence-high { color: #4ade80; font-weight: 600; }
    .confidence-medium { color: #facc15; font-weight: 600; }
    .confidence-low { color: #f87171; font-weight: 600; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
        border-right: 1px solid var(--card-border-color);
    }
    
    .stSelectbox, .stTextArea {
        background-color: transparent !important;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, var(--secondary-color), #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }

    .performance-chart {
        background: var(--card-bg-color);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border-color);
    }
    
    .footer {
        background: transparent;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        color: var(--text-color);
        opacity: 0.7;
    }

    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hide Streamlit default header */
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and vectorizer"""
    models = {}
    model_status = {}
    
    try:
        # Load TF-IDF vectorizer
            models['vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
        model_status['vectorizer'] = True
        
        # Load SVM model
        models['svm'] = joblib.load('models/svm_model.pkl')
        model_status['svm'] = True
        
        # Load Decision Tree model
        models['decision_tree'] = joblib.load('models/decision_tree_model.pkl')
        model_status['decision_tree'] = True
        
        # Load AdaBoost model
        models['adaboost'] = joblib.load('models/adaboost_model.pkl')
        model_status['adaboost'] = True
        
        return models, model_status
        
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure all model files are in the 'models/' directory")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error loading models: {e}")
        return None, None

# ============================================================================
# TEXT PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_text(text):
    """Preprocess text similar to training pipeline"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def make_prediction(text, model_name, models):
    """Make prediction using the selected model"""
    if models is None:
        return None, None, None
    
    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Transform text using TF-IDF vectorizer
        X = models['vectorizer'].transform([processed_text])
        
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

def get_prediction_explanation(prediction, confidence):
    """Generate explanation for the prediction"""
    class_name = "AI-Generated" if prediction == 1 else "Human-Written"
    
    if confidence >= 0.8:
        confidence_level = "High"
        confidence_class = "confidence-high"
    elif confidence >= 0.6:
        confidence_level = "Medium"
        confidence_class = "confidence-medium"
    else:
        confidence_level = "Low"
        confidence_class = "confidence-low"
    
    explanation = f"""
    **Prediction:** {class_name}
    **Confidence:** <span class="{confidence_class}">{confidence:.2%} ({confidence_level})</span>
    
    This text appears to be **{class_name.lower()}** based on the trained model's analysis.
    """
    
    return explanation, class_name, confidence_level

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_confidence_chart(probabilities):
    """Create a confidence visualization"""
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

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

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

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">AI vs Human Text Detector</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; color: var(--text-color);">
        <p style="font-size: 1.25rem; max-width: 650px; margin: auto; opacity: 0.9;">
            Leveraging state-of-the-art machine learning to discern the subtle differences between human and AI-generated text.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model status cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="model-card">
            <div style="text-align: center;">
                <h3 style="color: var(--secondary-color); font-weight: 700;">SVM Model</h3>
                <p style="color: var(--text-color); opacity: 0.8;">Support Vector Machine</p>
                <div style="font-size: 2.5rem; font-weight: 700; margin: 1.5rem 0; color: var(--secondary-color);">96.38%</div>
                <p style="font-size: 0.9rem; color: var(--text-color); opacity: 0.7;">Highest accuracy for complex pattern recognition.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="model-card">
            <div style="text-align: center;">
                <h3 style="color: var(--secondary-color); font-weight: 700;">Decision Tree</h3>
                <p style="color: var(--text-color); opacity: 0.8;">Tree-Based Classifier</p>
                <div style="font-size: 2.5rem; font-weight: 700; margin: 1.5rem 0; color: var(--primary-color);">84.99%</div>
                <p style="font-size: 0.9rem; color: var(--text-color); opacity: 0.7;">Highly interpretable for understanding decision paths.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="model-card">
            <div style="text-align: center;">
                <h3 style="color: var(--secondary-color); font-weight: 700;">AdaBoost</h3>
                <p style="color: var(--text-color); opacity: 0.8;">Ensemble Algorithm</p>
                <div style="font-size: 2.5rem; font-weight: 700; margin: 1.5rem 0; color: var(--accent-color);">85.50%</div>
                <p style="font-size: 0.9rem; color: var(--text-color); opacity: 0.7;">Combines learners to reduce bias and improve robustness.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation Guide
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <p style="color: var(--text-color); font-size: 1.1rem;">
            👆 Use the sidebar to navigate to <strong>"🔮 Single Text Analysis"</strong> to start analyzing text!
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SINGLE TEXT ANALYSIS PAGE
# ============================================================================

elif page == "🔮 Single Text Analysis":
    st.markdown('<h1 class="main-header">🔮 Single Text Analysis</h1>', unsafe_allow_html=True)
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Text to Analyze")
        
        # Text input
        text_input = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Paste or type the text you want to analyze..."
        )
        
    with col2:
        st.markdown("### Model Selection")
        
            model_choice = st.selectbox(
            "Choose model:",
            ["svm", "decision_tree", "adaboost"],
            format_func=lambda x: {
                "svm": "🎯 SVM (Recommended)",
                "decision_tree": "🌳 Decision Tree",
                "adaboost": "🚀 AdaBoost"
            }[x]
        )
        
        st.markdown("### Model Info")
        if model_choice == "svm":
            st.info("**SVM**: Highest accuracy (96.38%), excellent for complex text patterns.")
        elif model_choice == "decision_tree":
            st.info("**Decision Tree**: Most interpretable (84.99%), shows clear decision rules.")
        else:
            st.info("**AdaBoost**: Ensemble method (85.50%), combines multiple weak learners.")
            
            # Example texts
            with st.expander("📝 Try these example texts"):
                col1, col2 = st.columns(2)
                            
                            with col1:
                st.markdown("**Human-Written Example:**")
                human_example = "The importance of education in shaping our future cannot be overstated. Throughout history, societies that have invested in education have flourished, while those that neglected it have struggled. Education empowers individuals to think critically, solve problems, and contribute meaningfully to their communities."
                st.text_area("Copy this example:", value=human_example, height=100, key="human_example", disabled=True)
                            
                            with col2:
                st.markdown("**AI-Generated Example:**")
                ai_example = "Education is fundamentally crucial for societal development and individual growth. It provides essential skills and knowledge that enable people to navigate complex challenges. Moreover, educational institutions serve as catalysts for innovation and progress across various sectors."
                st.text_area("Copy this example:", value=ai_example, height=100, key="ai_example", disabled=True)
    
    # Input validation and analysis
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        elif len(text_input.strip().split()) < 10:
            st.warning("⚠️ Text is very short. For better accuracy, please provide at least 10-15 words.")
        elif len(text_input.strip()) > 10000:
            st.warning("⚠️ Text is very long. Processing first 10,000 characters for optimal performance.")
            text_input = text_input[:10000]
        
        if text_input.strip() and len(text_input.strip().split()) >= 5:  # Minimum threshold
            with st.spinner("🔍 Analyzing text patterns..."):
                try:
                    prediction, probabilities, confidence = make_prediction(text_input, model_choice, models)
                                    
                    if prediction is not None:
                        # Display prediction
                        explanation, class_name, confidence_level = get_prediction_explanation(prediction, confidence)
                        
                        if prediction == 1:  # AI-generated
                            st.markdown(
                                f'<div class="ai-prediction">🤖 <strong>AI-Generated Text Detected</strong><br/>{explanation}</div>',
                                unsafe_allow_html=True
                            )
                        else:  # Human-written
                            st.markdown(
                                f'<div class="human-prediction">👤 <strong>Human-Written Text Detected</strong><br/>{explanation}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Detailed results
                        col1, col2 = st.columns(2)
                                
                                with col1:
                            st.markdown("### 📊 Detailed Results")
                            st.metric("Human Probability", f"{probabilities[0]:.2%}")
                            st.metric("AI Probability", f"{probabilities[1]:.2%}")
                            st.metric("Confidence Level", confidence_level)
                            
                                with col2:
                            st.markdown("### 📈 Confidence Visualization")
                            confidence_chart = create_confidence_chart(probabilities)
                            st.plotly_chart(confidence_chart, use_container_width=True)
                                
                        # Text statistics
                        with st.expander("📋 Text Statistics"):
                            word_count = len(text_input.split())
                            char_count = len(text_input)
                            sentence_count = len([s for s in text_input.split('.') if s.strip()])
                            
                            stat_col1, stat_col2, stat_col3 = st.columns(3)
                            stat_col1.metric("Word Count", word_count)
                            stat_col2.metric("Character Count", char_count)
                            stat_col3.metric("Sentences", sentence_count)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.info("Please try again with different text or contact support if the issue persists.")

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "⚖️ Model Comparison":
    st.markdown('<h1 class="main-header">⚖️ Model Comparison</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Compare predictions across all three models for the same text to see how different algorithms
    analyze the content.
    """)
    
            # Text input for comparison
            comparison_text = st.text_area(
        "Enter text to compare across models:",
        height=200,
        placeholder="Enter the text you want to analyze with all three models..."
            )
            
    if st.button("🔀 Compare Models", type="primary", use_container_width=True):
        if comparison_text.strip():
            with st.spinner("Running all models..."):
                # Get predictions from all models
                model_names = ['svm', 'decision_tree', 'adaboost']
                model_labels = ['SVM', 'Decision Tree', 'AdaBoost']
                
                predictions = []
                confidences = []
                comparison_data = []
                
                for i, model_name in enumerate(model_names):
                    pred, probs, conf = make_prediction(comparison_text, model_name, models)
                    if pred is not None:
                        pred_label = "AI-Generated" if pred == 1 else "Human-Written"
                        predictions.append(pred_label)
                        confidences.append(conf)
                        comparison_data.append({
                            'Model': model_labels[i],
                            'Prediction': pred_label,
                            'Confidence': f"{conf:.2%}",
                            'Human Probability': f"{probs[0]:.2%}",
                            'AI Probability': f"{probs[1]:.2%}"
                        })
                    else:
                        predictions.append("Error")
                        confidences.append(0)
                        comparison_data.append({
                            'Model': model_labels[i],
                            'Prediction': "Error",
                            'Confidence': "N/A",
                            'Human Probability': "N/A",
                            'AI Probability': "N/A"
                        })
                
                # Display comparison chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=model_labels,
                        y=confidences,
                        text=[f'{c:.1%}' if c > 0 else 'Error' for c in confidences],
                        textposition='auto',
                        marker_color=['#007bff', '#28a745', '#ffc107']
                    )
                ])
                
                fig.update_layout(
                    title="Model Comparison - Confidence Scores",
                    yaxis_title="Confidence",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed comparison table
                st.markdown("### 📊 Detailed Comparison")
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                    
                    # Agreement analysis
                st.markdown("### 🤝 Model Agreement Analysis")
                
                valid_predictions = [p for p in predictions if p != "Error"]
                if len(set(valid_predictions)) == 1 and len(valid_predictions) > 0:
                    st.success("🎯 **All models agree!** All three models made the same prediction.")
                elif len(valid_predictions) > 1:
                    st.warning("⚠️ **Models disagree.** Different models made different predictions.")
                else:
                    st.error("❌ **Error in predictions.** Some models failed to make predictions.")
                
                # Show individual predictions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if predictions[0] != "Error":
                        pred_class = "ai-prediction" if predictions[0] == "AI-Generated" else "human-prediction"
                        st.markdown(f'<div class="{pred_class}"><strong>SVM:</strong> {predictions[0]}<br/>Confidence: {confidences[0]:.1%}</div>', unsafe_allow_html=True)
                    else:
                        st.error("SVM: Error")
                
                with col2:
                    if predictions[1] != "Error":
                        pred_class = "ai-prediction" if predictions[1] == "AI-Generated" else "human-prediction"
                        st.markdown(f'<div class="{pred_class}"><strong>Decision Tree:</strong> {predictions[1]}<br/>Confidence: {confidences[1]:.1%}</div>', unsafe_allow_html=True)
                    else:
                        st.error("Decision Tree: Error")
                
                with col3:
                    if predictions[2] != "Error":
                        pred_class = "ai-prediction" if predictions[2] == "AI-Generated" else "human-prediction"
                        st.markdown(f'<div class="{pred_class}"><strong>AdaBoost:</strong> {predictions[2]}<br/>Confidence: {confidences[2]:.1%}</div>', unsafe_allow_html=True)
                else:
                        st.error("AdaBoost: Error")
            
        else:
            st.warning("Please enter some text to compare.")

# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================

elif page == "📊 Model Performance":
    st.markdown('<h1 class="main-header">📊 Model Performance & Information</h1>', unsafe_allow_html=True)
    
    # Model performance metrics
    st.markdown("### 🎯 Model Accuracy Comparison")
        
    # Performance data from your assignment results
    performance_data = {
        'Model': ['SVM', 'Decision Tree', 'AdaBoost'],
        'Accuracy': [96.38, 84.99, 85.50],
        'Precision': [96.38, 85.02, 85.45],
        'Recall': [96.38, 84.99, 85.50],
        'F1-Score': [96.38, 84.98, 85.47]
    }
    
    performance_df = pd.DataFrame(performance_data)
    
    # Performance visualization with new dark theme
    fig = px.bar(
        performance_df, 
        x='Model', 
        y='Accuracy',
        color='Model',
        title="Model Accuracy Comparison",
        color_discrete_map={'SVM': '#5372f0', 'Decision Tree': '#4ade80', 'AdaBoost': '#facc15'}
    )
    fig.update_layout(
        showlegend=False, 
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_color='#ffffff',
        title_x=0.5,
        font=dict(family="Inter, sans-serif", size=12, color='#dcdcdc'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
    )
    fig.update_traces(
        texttemplate='%{y:.1f}%', 
        textposition='outside'
    )
    
    st.markdown('<div class="performance-chart">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance table
    st.dataframe(performance_df, use_container_width=True)
    
    # Model information
    st.markdown("### 🤖 Model Information")
    
    tab1, tab2, tab3 = st.tabs(["SVM", "Decision Tree", "AdaBoost"])
    
    with tab1:
        st.markdown("""
        #### Support Vector Machine (SVM)
        
        **Description:** SVM finds the optimal boundary (hyperplane) that separates AI-generated and human-written texts with maximum margin.
        
        **Strengths:**
        - Excellent performance on high-dimensional data (like text)
        - Robust against overfitting
        - Works well with sparse data
        
        **Best for:** Complex text patterns and high accuracy requirements
        
        **Training Results:**
        - Cross-validation accuracy: 97.28% ± 1.09%
        - Test accuracy: 96.38%
        - Optimized kernel and parameters via Grid Search
        """)
    
    with tab2:
        st.markdown("""
        #### Decision Tree
        
        **Description:** Creates a tree-like model of decisions based on text features to classify AI vs Human content.
        
        **Strengths:**
        - Highly interpretable results
        - Handles both numerical and categorical features
        - No assumptions about data distribution
        
        **Best for:** Understanding decision rules and feature importance
        
        **Training Results:**
        - Cross-validation accuracy: 85.32% ± 2.15%
        - Test accuracy: 84.99%
        - Optimized depth and splitting criteria
        """)
    
    with tab3:
        st.markdown("""
        #### AdaBoost (Adaptive Boosting)
        
        **Description:** Combines multiple weak learners (typically decision stumps) to create a strong classifier.
        
        **Strengths:**
        - Reduces bias and variance
        - Automatically focuses on difficult cases
        - Good generalization performance
        
        **Best for:** Ensemble learning and improved robustness
        
        **Training Results:**
        - Cross-validation accuracy: 85.68% ± 1.98%
        - Test accuracy: 85.50%
        - 100 estimators with optimized learning rate
        """)
    
    # Feature importance
    st.markdown("### 📈 Feature Processing")
    
    st.markdown("""
    **Text Preprocessing Pipeline:**
    1. **Lowercasing:** Convert all text to lowercase
    2. **Special Character Removal:** Remove non-alphabetic characters
    3. **Whitespace Normalization:** Remove extra spaces
    4. **TF-IDF Vectorization:** Convert text to numerical features
    
    **TF-IDF Features:**
    - Captures word importance relative to document and corpus
    - Reduces impact of common words
    - Creates sparse feature vectors for efficient processing
    - 5,000 most important features selected
    """)

# ============================================================================
# FOOTER AND TECHNICAL INFORMATION
# ============================================================================

st.markdown("---")

# Technical specifications footer
st.markdown("### 🔬 Technical Specifications")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    **📊 Dataset**
    - 3,728 labeled text samples
    - Binary classification (Human vs AI)
    - Essay-style text format
    - Balanced class distribution
    """)

with col2:
    st.markdown("""
    **⚙️ Preprocessing**
    - Text normalization & cleaning
    - TF-IDF vectorization (5K features)
    - Stop word removal
    - Special character handling
    """)

with col3:
    st.markdown("""
    **🎯 Model Performance**
    - SVM: 96.38% accuracy
    - Decision Tree: 84.99% accuracy
    - AdaBoost: 85.50% accuracy
    - 5-fold cross-validation
    """)

with col4:
    st.markdown("""
    **🛠️ Technology Stack**
    - Python 3.8+ & Scikit-learn
    - Streamlit web framework
    - NLTK for text processing
    - Plotly for visualizations
""")

st.markdown("---")

# Project information
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <h4>📚 Project 1: AI vs Human Text Detection</h4>
    <p><em>Advanced Text Classification with Machine Learning Pipeline</em></p>
    <p><strong>Built by Sahel Azzam | Course: Intro to Large Language Models and AI Agents</strong></p>
    <p style="color: #ec4899; font-weight: bold;">🏆 Achieving 96.38% accuracy with state-of-the-art ML techniques</p>
</div>
""", unsafe_allow_html=True)
