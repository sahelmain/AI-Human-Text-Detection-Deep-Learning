# AI vs Human Text Detection - Hugging Face Spaces Deployment

## Deployment Instructions for Hugging Face Spaces

### 1. Required Files for Deployment
Make sure you have these files in your repository:

- `streamlit_app.py` - Main application file
- `requirements.txt` - Python dependencies (UPDATED with fixed torch versions)
- `packages.txt` - System dependencies for Ubuntu (NEWLY ADDED)
- `.streamlit/config.toml` - Streamlit configuration (NEWLY ADDED)
- `utils.py` - Utility functions
- `models/` directory with all your .pkl files
- `setup_nltk.py` - NLTK data setup script (NEWLY ADDED)

### 2. Key Fixes Applied

#### PyTorch Compatibility Issues
- Fixed torch version to `torch==2.0.1` and `torchvision==0.15.2`
- Added better error handling for PyTorch model loading
- Added fallback mechanisms for torch.classes errors

#### NLTK Data Issues
- Enhanced NLTK setup with multiple fallback paths
- Added robust NLTK data downloading for cloud environments
- Created separate setup script for pre-downloading NLTK data

#### System Dependencies
- Added `packages.txt` with required build tools
- Added system packages: `build-essential`, `gcc`, `g++`, `python3-dev`

#### Streamlit Configuration
- Added `.streamlit/config.toml` for optimal performance in Hugging Face Spaces
- Configured upload limits and CORS settings

### 3. Deployment Steps

1. **Create a new Hugging Face Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Streamlit" as the SDK
   - Choose appropriate visibility and hardware

2. **Upload your files:**
   - Upload all files maintaining the directory structure
   - Ensure the `models/` directory contains all your .pkl files
   - Make sure `requirements.txt`, `packages.txt`, and `.streamlit/config.toml` are in the root

3. **Monitor the build:**
   - Check the "Logs" tab to monitor the deployment progress
   - Look for any error messages during the build process

### 4. Troubleshooting Common Issues

#### "torch.classes" Error
- Fixed with updated torch loading method using `weights_only=False`
- Fallback to pickle loading if torch.load fails

#### Large Model Files
- If models are too large, consider:
  - Using Git LFS for large files
  - Compressing model files
  - Using model quantization techniques

#### NLTK Download Issues
- Fixed with robust NLTK setup and multiple fallback paths
- Pre-downloads essential NLTK data during app startup

#### Memory Issues
- Consider upgrading to a paid Hugging Face Spaces tier if the app requires more memory
- Optimize model loading to use less memory

### 5. Testing Locally

Before deploying, test the app locally with:

```bash
# Install dependencies
pip install -r requirements.txt

# Run NLTK setup (optional)
python setup_nltk.py

# Run the app
streamlit run streamlit_app.py
```

### 6. Expected Behavior

The app should now:
- Load successfully without torch.classes errors
- Handle NLTK data properly
- Gracefully degrade if PyTorch models fail to load
- Show warnings for any component failures but continue running
- Work with both ML and DL models when available

### 7. Performance Tips

- The app uses `@st.cache_resource` for model loading
- NLTK data is cached after first download
- Consider using Hugging Face Spaces Pro for better performance with large models

### 8. Support

If you continue to have issues:
1. Check the "Logs" tab in your Hugging Face Space
2. Verify all files are uploaded correctly
3. Ensure model files are not corrupted
4. Try restarting the Space from the settings

## Updated File Structure

```
ai_human_project/
├── streamlit_app.py              # Main app (UPDATED)
├── requirements.txt              # Dependencies (UPDATED)
├── packages.txt                  # System deps (NEW)
├── setup_nltk.py                # NLTK setup (NEW)
├── utils.py                      # Utilities
├── README_DEPLOYMENT.md          # This file (NEW)
├── .streamlit/
│   └── config.toml              # Streamlit config (NEW)
└── models/
    ├── *.pkl files              # All model files
    └── vocab_to_idx.pkl         # Vocabulary mapping
``` 