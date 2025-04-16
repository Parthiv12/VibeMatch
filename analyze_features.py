import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_features(features_dir):
    """Load all features from the features directory."""
    features = []
    labels = []
    feature_sizes = {}
    
    for mood_dir in os.listdir(features_dir):
        mood_path = os.path.join(features_dir, mood_dir)
        if not os.path.isdir(mood_path):
            continue
            
        for feature_file in os.listdir(mood_path):
            if not feature_file.endswith('.npz'):
                continue
                
            feature_path = os.path.join(mood_path, feature_file)
            data = np.load(feature_path)
            
            # Get feature sizes on first iteration
            if not feature_sizes:
                for key in data.files:
                    feature_sizes[key] = data[key].size
            
            # Combine all features into a single vector
            feature_vector = np.concatenate([
                data['mel_spectrogram'].flatten(),
                data['mfccs'].flatten(),
                data['chroma'].flatten(),
                data['spectral_contrast'].flatten(),
                data['tonnetz'].flatten()
            ])
            
            features.append(feature_vector)
            labels.append(mood_dir)
    
    return np.array(features), np.array(labels), feature_sizes

def reduce_dimensionality(features, n_components=None):
    """Reduce feature dimensionality using PCA."""
    if n_components is None:
        n_components = min(features.shape[0] - 1, features.shape[1])
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    
    # Print explained variance ratio
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    print("\nPCA Analysis:")
    print(f"Number of components: {n_components}")
    print(f"Explained variance ratio: {cumsum[-1]:.4f}")
    
    return reduced_features

def visualize_features(features, labels):
    """Visualize features using t-SNE."""
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Reduce dimensionality first (use 80% of samples)
    n_components = int(0.8 * len(features))
    features_reduced = reduce_dimensionality(features_scaled, n_components=n_components)
    
    # Adjust perplexity based on sample size
    n_samples = len(features)
    perplexity = min(30, n_samples - 1)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_tsne = tsne.fit_transform(features_reduced)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': features_tsne[:, 0],
        'y': features_tsne[:, 1],
        'mood': labels
    })
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='mood', palette='viridis')
    plt.title('t-SNE Visualization of Audio Features')
    plt.savefig('feature_visualization.png')
    plt.close()

def analyze_feature_types(features, labels, feature_sizes):
    """Analyze each type of feature separately."""
    start_idx = 0
    feature_stats = {}
    
    for feature_name, size in feature_sizes.items():
        end_idx = start_idx + size
        feature_slice = features[:, start_idx:end_idx]
        
        # Calculate statistics for each mood
        for mood in np.unique(labels):
            mood_features = feature_slice[labels == mood]
            if feature_name not in feature_stats:
                feature_stats[feature_name] = {}
            feature_stats[feature_name][mood] = {
                'mean': np.mean(mood_features),
                'std': np.std(mood_features),
                'min': np.min(mood_features),
                'max': np.max(mood_features)
            }
        
        start_idx = end_idx
    
    # Plot feature type distributions
    plt.figure(figsize=(15, 10))
    for i, (feature_name, stats) in enumerate(feature_stats.items()):
        plt.subplot(3, 2, i+1)
        for mood, values in stats.items():
            plt.hist(values['mean'], bins=20, alpha=0.5, label=mood)
        plt.title(f'{feature_name} Distribution')
        plt.xlabel('Mean Value')
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.savefig('feature_type_distributions.png')
    plt.close()
    
    return feature_stats

def print_feature_analysis(feature_stats):
    """Print detailed analysis of features."""
    print("\nFeature Analysis:")
    for feature_name, stats in feature_stats.items():
        print(f"\n{feature_name}:")
        for mood, values in stats.items():
            print(f"  {mood}:")
            print(f"    Mean: {values['mean']:.4f}")
            print(f"    Std:  {values['std']:.4f}")
            print(f"    Min:  {values['min']:.4f}")
            print(f"    Max:  {values['max']:.4f}")

def main():
    # Load features
    features, labels, feature_sizes = load_features('data/features')
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(features)}")
    print(f"Feature dimension: {features.shape[1]}")
    print("\nSamples per mood:")
    for mood in np.unique(labels):
        print(f"{mood}: {np.sum(labels == mood)}")
    
    print("\nFeature sizes:")
    for name, size in feature_sizes.items():
        print(f"{name}: {size}")
    
    # Visualize features
    print("\nVisualizing features using t-SNE...")
    visualize_features(features, labels)
    
    # Analyze feature types
    print("\nAnalyzing feature distributions...")
    feature_stats = analyze_feature_types(features, labels, feature_sizes)
    print_feature_analysis(feature_stats)

if __name__ == "__main__":
    main() 