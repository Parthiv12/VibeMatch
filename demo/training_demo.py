import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vibe_encoder import VibeEncoder, TripletLoss
from preprocessing.audio_processor import MusicDataset, AudioProcessor
from utils.visualization import (
    plot_training_progress,
    visualize_embeddings,
    plot_model_architecture,
    plot_spectrogram
)
from utils.spotify_utils import SpotifyMetadata

def train_with_visualization(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_dir):
    """Train model with visualization of progress"""
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, log_var = model(data)
            
            # Calculate losses
            recon_loss = criterion(recon_batch, data)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Triplet loss
            anchor = mu[::3]
            positive = mu[1::3]
            negative = mu[2::3]
            triplet_loss = TripletLoss()(anchor, positive, negative)
            
            # Total loss
            loss = recon_loss + 0.1 * kld_loss + 0.1 * triplet_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Save first batch spectrogram
            if batch_idx == 0 and epoch == 0:
                plot_spectrogram(data[0].cpu().numpy(), 
                               os.path.join(save_dir, 'sample_spectrogram.png'))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon_batch, mu, log_var = model(data)
                loss = criterion(recon_batch, data)
                val_loss += loss.item()
        
        # Record losses
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        # Plot progress
        plot_training_progress(train_losses, val_losses,
                             os.path.join(save_dir, f'training_progress_epoch_{epoch}.png'))
        
        # Visualize embeddings every 5 epochs
        if epoch % 5 == 0:
            embeddings = []
            labels = []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    mu, _ = model.encode(data)
                    embeddings.extend(mu.cpu().numpy())
                    labels.extend(np.arange(len(data)))
            
            visualize_embeddings(np.array(embeddings), np.array(labels),
                               os.path.join(save_dir, f'embeddings_epoch_{epoch}.png'))
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'training_visualizations'
    
    # Initialize components
    processor = AudioProcessor()
    dataset = MusicDataset('data/audio', processor)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = VibeEncoder().to(device)
    
    # Plot model architecture
    plot_model_architecture(model, (128, 100), os.path.join(save_dir, 'model_architecture'))
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # Train with visualization
    train_with_visualization(
        model, train_loader, val_loader, optimizer, criterion,
        device, epochs=50, save_dir=save_dir
    )

if __name__ == '__main__':
    main() 