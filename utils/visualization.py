import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from sklearn.manifold import TSNE

def plot_training_progress(train_losses, val_losses, save_path=None):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_embeddings(embeddings, labels, save_path=None):
    """Visualize song embeddings using t-SNE"""
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Song Embeddings Visualization (t-SNE)')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_model_architecture(model, input_shape, save_path=None):
    """Create a visualization of the model architecture"""
    from torchviz import make_dot
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # Forward pass
    output = model(dummy_input)
    
    # Create visualization
    dot = make_dot(output, params=dict(model.named_parameters()))
    
    if save_path:
        dot.render(save_path, format='png')
    
    return dot

def plot_spectrogram(spectrogram, save_path=None):
    """Plot mel spectrogram"""
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_distribution(features, feature_names, save_path=None):
    """Plot distribution of audio features"""
    plt.figure(figsize=(12, 6))
    for i, (feature, name) in enumerate(zip(features, feature_names)):
        plt.subplot(2, 3, i+1)
        sns.histplot(feature, kde=True)
        plt.title(name)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close() 