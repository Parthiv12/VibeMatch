import torch
from models.vibe_encoder import VibeEncoder
from preprocessing.audio_processor import AudioProcessor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

def test_song(song_path):
    # Initialize components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AudioProcessor()
    
    # Load the model
    model_path = 'checkpoints/best_model.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you have trained the model first")
        return
    
    # Initialize model with the same dimensions as the trained model
    model = VibeEncoder(input_dim=169344, hidden_dims=[1024, 512, 256], latent_dim=64)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Process the song
    print(f"\nProcessing song: {os.path.basename(song_path)}")
    print(f"Full path: {os.path.abspath(song_path)}")
    
    try:
        features = processor.process_audio_file(song_path)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        # Get mood prediction
        with torch.no_grad():
            recon_batch, mu, log_var = model(features_tensor)
            # Use the latent representation (mu) for mood prediction
            mood_features = mu.squeeze().cpu().numpy()
            
            # Simple mood classification based on the first dimension of latent space
            predicted_mood = 1 if mood_features[0] > 0 else 0
            confidence = abs(mood_features[0]) / np.linalg.norm(mood_features)
        
        mood_map = {0: 'energetic', 1: 'happy'}
        print(f"\nPredicted Mood: {mood_map[predicted_mood]}")
        print(f"Confidence: {confidence:.4f}")
        
        # Get mood probabilities (approximated from latent space)
        print("\nMood Probabilities:")
        probs = torch.softmax(torch.tensor([1-confidence, confidence]), dim=0)
        for mood_idx, mood_name in mood_map.items():
            print(f"{mood_name}: {probs[mood_idx].item():.4f}")
        
        # Find similar songs
        print("\nFinding similar songs...")
        database = []
        song_names = []
        
        # Load all songs in the database
        for mood_dir in ['energetic', 'happy']:
            mood_path = os.path.join('data/raw', mood_dir)
            if not os.path.exists(mood_path):
                print(f"Warning: Mood directory not found: {mood_path}")
                continue
                
            for song_file in os.listdir(mood_path):
                if song_file.endswith('.mp3'):
                    db_song_path = os.path.join(mood_path, song_file)
                    try:
                        song_features = processor.process_audio_file(db_song_path)
                        song_tensor = torch.FloatTensor(song_features).unsqueeze(0).to(device)
                        with torch.no_grad():
                            _, song_mu, _ = model(song_tensor)
                        database.append(song_mu.squeeze().cpu().numpy())
                        song_names.append(song_file)
                    except Exception as e:
                        print(f"Error processing {song_file}: {str(e)}")
        
        if not database:
            print("Error: No songs found in the database")
            return
        
        # Calculate similarities using the latent representations
        similarities = cosine_similarity([mood_features], database)[0]
        
        # Get top 5 similar songs
        top_indices = np.argsort(similarities)[::-1][:5]
        
        print("\nTop 5 Similar Songs:")
        for idx in top_indices:
            print(f"{song_names[idx]}: {similarities[idx]:.4f}")
            
    except Exception as e:
        print(f"Error processing song: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_song(sys.argv[1])
    else:
        print("Please provide the path to the test song as an argument")
        print("Example: python test_model.py path/to/your/song.mp3") 