import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vibe_encoder import VibeEncoder, TripletLoss
from preprocessing.audio_processor import MusicDataset, AudioProcessor


#Training with Backpropagation
def train(model, train_loader, optimizer, criterion, device, epoch, writer):
    model.train()
    total_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, log_var = model(data)
        
        # Calculate losses
        recon_loss = criterion(recon_batch, data)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Triplet loss (using random triplets for demonstration)
        anchor = mu[::3]
        positive = mu[1::3]
        negative = mu[2::3]
        triplet_loss = TripletLoss()(anchor, positive, negative)
        
        # Total loss
        loss = recon_loss + 0.1 * kld_loss + 0.1 * triplet_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
    
    return total_loss / len(train_loader)

def main():
    # Hyperparameters
    batch_size = 32
    epochs = 50
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize components
    processor = AudioProcessor()
    dataset = MusicDataset('data/audio', processor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = VibeEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # TensorBoard writer
    writer = SummaryWriter('runs/vibematch_experiment')
    
    # Training loop
    for epoch in range(1, epochs + 1):
        avg_loss = train(model, train_loader, optimizer, criterion, device, epoch, writer)
        print(f'Epoch {epoch} Average Loss: {avg_loss:.6f}')
        
        # Save model checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoints/model_epoch_{epoch}.pt')
    
    writer.close()

if __name__ == '__main__':
    main() 