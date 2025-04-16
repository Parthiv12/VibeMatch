import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vibe_encoder import VibeEncoder, TripletLoss
from preprocessing.audio_processor import MusicDataset


#Training with Backpropagation
def train(model, train_loader, optimizer, criterion, device, epoch, writer):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, log_var = model(data)
        
        # Calculate losses
        recon_loss = criterion(recon_batch, data)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Triplet loss (using random triplets for demonstration)
        if len(mu) >= 3:  # Only compute triplet loss if we have enough samples
            anchor = mu[::3]
            positive = mu[1::3]
            negative = mu[2::3]
            triplet_loss = TripletLoss()(anchor, positive, negative)
        else:
            triplet_loss = torch.tensor(0.0).to(device)
        
        # Total loss
        loss = recon_loss + 0.1 * kld_loss + 0.1 * triplet_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/reconstruction', recon_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/kld', kld_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/triplet', triplet_loss.item(), epoch * len(train_loader) + batch_idx)
    
    return total_loss / len(train_loader)

def main():
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Hyperparameters
    batch_size = 4  # Reduced batch size due to large input dimension
    epochs = 10  # Reduced number of epochs for testing
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Starting test run with {epochs} epochs")
    
    # Initialize dataset
    dataset = MusicDataset(
        audio_dir='data/audio',
        metadata_dir='data/raw'
    )
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = VibeEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = nn.MSELoss()
    
    # TensorBoard writer
    writer = SummaryWriter('runs/vibematch_experiment')
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        print(f"\nStarting epoch {epoch}/{epochs}")
        avg_loss = train(model, train_loader, optimizer, criterion, device, epoch, writer)
        print(f'Epoch {epoch} Average Loss: {avg_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, 'checkpoints/best_model.pt')
            print(f"New best model saved with loss: {avg_loss:.6f}")
    
    writer.close()
    print("\nTest run completed!")
    print(f"Best loss achieved: {best_loss:.6f}")

if __name__ == '__main__':
    main() 