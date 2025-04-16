# VibeMatch: A Deep Learning Recommender for Mood-Based Music Discovery

## Project Overview
VibeMatch is an innovative music recommendation system that uses deep learning to match songs based on their "vibe" - the emotional and atmospheric qualities that make songs feel similar, beyond just genre or artist similarity.

## Problem Statement
Traditional music recommendation systems often rely on collaborative filtering or genre-based approaches. VibeMatch addresses the challenge of finding songs that share similar emotional and atmospheric qualities, even if they come from different genres or artists.

## Technical Architecture

### 1. Audio Feature Extraction
- Uses pre-trained CNN (VGGish/YAMNet) for audio feature extraction
- Converts raw audio to mel-spectrograms
- Optional integration with Spotify/Last.fm API for additional metadata

### 2. Song Embedding
- Autoencoder/MLP architecture to create "vibe vectors"
- Dimensionality reduction to capture essential mood characteristics
- Triplet loss implementation for better similarity learning

### 3. Similarity Model
- Cosine similarity for recommendation
- Nearest neighbor search in vibe vector space
- Optional RNN component for lyrics analysis

## Deep Learning Techniques Applied
- Convolutional Neural Networks (CNN) for audio processing
- Autoencoder for feature compression
- Triplet Loss for similarity learning
- Optional RNN for lyrics analysis
- Batch Normalization and Dropout for regularization

## Dataset
- Million Song Dataset
- Spotify Dataset
- Curated vibe playlists

## Evaluation Metrics
- Precision@K
- User study for qualitative assessment
- Similarity clustering analysis

## Project Structure
```
vibematch/
├── data/               # Dataset and processed features
├── models/            # Model architectures
├── preprocessing/     # Audio processing scripts
├── training/         # Training scripts
├── evaluation/       # Evaluation metrics
└── utils/            # Utility functions
```

## Requirements
- Python 3.8+
- PyTorch/TensorFlow
- Librosa for audio processing
- Spotify API (optional)
- Other dependencies listed in requirements.txt

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download and preprocess the dataset
4. Train the model: `python training/train.py`
5. Run evaluation: `python evaluation/evaluate.py`

## Future Work
- Integration with streaming platforms
- Real-time recommendation engine
- User feedback loop for continuous improvement
- Mobile application development

## License
MIT License 