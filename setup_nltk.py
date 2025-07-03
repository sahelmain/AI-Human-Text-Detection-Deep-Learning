#!/usr/bin/env python3
"""
NLTK setup script for Hugging Face Spaces
This script pre-downloads NLTK data to avoid runtime issues
"""

import nltk
import ssl
import os

def setup_nltk_data():
    """Download essential NLTK data"""
    try:
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Set NLTK data path
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Download essential packages
        packages = [
            'punkt',
            'averaged_perceptron_tagger',
            'stopwords',
            'wordnet',
            'omw-1.4'
        ]
        
        for package in packages:
            try:
                print(f"Downloading {package}...")
                nltk.download(package, quiet=True)
                print(f"✓ {package} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {package}: {e}")
        
        print("NLTK setup completed!")
        return True
        
    except Exception as e:
        print(f"NLTK setup failed: {e}")
        return False

if __name__ == "__main__":
    setup_nltk_data() 