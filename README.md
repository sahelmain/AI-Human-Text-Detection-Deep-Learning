# AI vs Human Text Detection - Deep Learning App

🤖📝 **An advanced web application for detecting AI-generated vs. human-written text using deep learning and traditional ML models. Achieves up to 97.33% accuracy with interactive features like AI agents for explanations.**

[![Deployed App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)](https://ai-human-text-detection-deep-learning.streamlit.app)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview
This project builds a comprehensive text classification system to distinguish between AI-generated and human-written text. It integrates:
- **Deep Learning Models**: CNN (97.33% accuracy), LSTM (94.52%), RNN (82.75%)
- **Traditional ML Models**: SVM (96.38%), Decision Tree (84.99%), AdaBoost (85.50%)
- **AI Agents**: Built-in agents for explanations, insights, and more (powered by LangChain/OpenAI with rule-based fallbacks).

Deployed live at: https://ai-human-text-detection-deep-learning.streamlit.app

## 🚀 Features
### Core Functionality
- **Text Analysis**: Real-time prediction with confidence scores and visualizations.
- **File Upload**: Analyze PDFs, DOCX, or TXT files.
- **Model Comparison**: Run all models side-by-side and compare results.
- **Advanced Analytics**: Word clouds, feature importance, and statistical insights.
- **AI Agents**: Interactive agents for detailed explanations, rewrites, and Q&A (e.g., "Why is this text AI-generated?").

### User Interface
- **Streamlit Web App**: Modern, responsive UI with gradient themes.
- **Report Generation**: Download PDF/Excel reports with charts and agent insights.
- **Batch Processing**: Handle multiple texts/files at once.

## 🛠 Installation
### Prerequisites
- Python 3.8+
- pip

### Quick Start
1. Clone the repo:  
   ```
   git clone https://github.com/sahelmain/AI-Human-Text-Detection-Deep-Learning.git
   cd AI-Human-Text-Detection-Deep-Learning
   ```
2. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```
3. Run NLTK setup (if needed):  
   ```
   python setup_nltk.py
   ```
4. Run the app locally:  
   ```
   streamlit run streamlit_app.py
   ```
5. For deployment, follow README_DEPLOYMENT.md (e.g., to Streamlit Cloud).

## 📱 Usage
- **Home Page**: Overview and navigation.
- **Text Analysis**: Paste text, select model, analyze, and get AI agent explanations.
- **File Upload**: Upload documents for batch analysis.
- **AI Agent Explanation**: Default landing page for interactive agent-based insights.
- Access the deployed version: [ai-human-text-detection-deep-learning.streamlit.app](https://ai-human-text-detection-deep-learning.streamlit.app)

## 📊 Model Performance
| Model          | Accuracy | Type            |
|----------------|----------|-----------------|
| CNN            | 97.33%   | Deep Learning   |
| SVM            | 96.38%   | Traditional ML  |
| LSTM           | 94.52%   | Deep Learning   |
| AdaBoost       | 85.50%   | Traditional ML  |
| Decision Tree  | 84.99%   | Traditional ML  |
| RNN            | 82.75%   | Deep Learning   |

- Cross-validated with 5-fold strategy.
- Ensemble voting available via agents for higher reliability.

## 📁 Project Structure
```
AI-Human-Text-Detection-Deep-Learning/
├── streamlit_app.py       # Main app
├── utils.py               # Helper functions
├── requirements.txt       # Dependencies
├── README.md              # This file
├── README_DEPLOYMENT.md   # Deployment guide
├── models/                # Saved models (CNN, LSTM, etc.)
├── notebooks/             # Development Jupyter notebooks
└── .streamlit/            # Config files
```

## 🔧 Technical Details
- **Preprocessing**: TF-IDF vectorization, tokenization via NLTK.
- **Deep Learning**: PyTorch-based CNN/LSTM/RNN with word embeddings.
- **AI Agents**: LangChain chains with OpenAI GPT-3.5-turbo (fallback to rule-based logic).
- **UI**: Streamlit with Plotly for interactive charts.

## 🤝 Contributing
Fork, create a branch, commit changes, and open a PR. Follow PEP8 style.

## 📄 License
MIT License. 