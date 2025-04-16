import torch
import os
from models.vibe_encoder import VibeEncoder
from preprocessing.audio_processor import AudioProcessor
from recommend import VibeRecommender
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def analyze_song(song_path):
    # Initialize components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AudioProcessor()
    recommender = VibeRecommender()
    
    # Process the song
    print(f"\nProcessing song: {os.path.basename(song_path)}")
    
    # Get mood prediction
    predicted_mood = recommender.get_song_mood(song_path)
    print(f"\nPredicted Mood: {predicted_mood}")
    
    # Process song features
    features = processor.process_audio_file(song_path)
    print("\nFeature sizes:")
    print(f"Total features: {len(features)}")
    
    # Find similar songs
    print("\nFinding similar songs in the database...")
    similarities = []
    song_names = []
    
    # Load all songs in the database
    for mood_dir in ['energetic', 'happy']:
        mood_path = os.path.join('data/raw', mood_dir)
        if not os.path.exists(mood_path):
            continue
            
        for song_file in os.listdir(mood_path):
            if song_file.endswith('.mp3'):
                song_path = os.path.join(mood_path, song_file)
                song_features = processor.process_audio_file(song_path)
                similarities.append(cosine_similarity([features], [song_features])[0][0])
                #Mel Spectrogram features(165,376 features), VGGish features(31 x 128 features)
                
                song_names.append(song_file)
    
    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    print("\nTop 5 similar songs:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"{i+1}. {song_names[idx]} (Similarity: {similarities[idx]:.4f})")

if __name__ == "__main__":
    song_path = r"C:\Users\Akhila\Downloads\Test.mp3"
    analyze_song(song_path) 