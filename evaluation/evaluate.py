import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vibe_encoder import VibeEncoder
from preprocessing.audio_processor import MusicDataset, AudioProcessor

def evaluate_model(model, test_loader, device):
    model.eval()
    embeddings = []
    file_paths = []
    
    with torch.no_grad():
        for data, paths in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            embeddings.extend(mu.cpu().numpy())
            file_paths.extend(paths)
    
    return np.array(embeddings), file_paths

def find_similar_songs(query_embedding, embeddings, file_paths, k=5):
    """Find k most similar songs to the query"""
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k-1:-1][::-1]  # Exclude the query itself
    return [(file_paths[i], similarities[i]) for i in top_k_indices]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = VibeEncoder().to(device)
    checkpoint = torch.load('checkpoints/model_epoch_50.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize data
    processor = AudioProcessor()
    dataset = MusicDataset('data/audio', processor)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Get embeddings
    embeddings, file_paths = evaluate_model(model, test_loader, device)
    
    # Example: Find similar songs for a query song
    query_idx = 0  # Using first song as query
    query_embedding = embeddings[query_idx]
    similar_songs = find_similar_songs(query_embedding, embeddings, file_paths)
    
    print("\nQuery Song:", file_paths[query_idx])
    print("\nSimilar Songs:")
    for song, similarity in similar_songs:
        print(f"{song}: Similarity = {similarity:.4f}")
    
    # Save embeddings for visualization
    np.save('embeddings.npy', embeddings)
    with open('file_paths.txt', 'w') as f:
        for path in file_paths:
            f.write(f"{path}\n")

if __name__ == '__main__':
    main() 