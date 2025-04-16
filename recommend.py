import torch
import numpy as np
from models.vibe_encoder import VibeEncoder
from preprocessing.audio_processor import AudioProcessor
import os
from sklearn.metrics.pairwise import cosine_similarity

class VibeRecommender:
    def __init__(self, model_path='checkpoints/best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AudioProcessor()
        
        # Load the trained model
        self.model = VibeEncoder().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize song database
        self.song_features = {}
        self.song_moods = {}
        
    def process_song(self, file_path):
        """Process a song and extract its features"""
        with torch.no_grad():
            features = self.processor.process_audio_file(file_path)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            _, mu, _ = self.model(features_tensor)
            return mu.squeeze().cpu().numpy()
    
    def add_song_to_database(self, file_path, mood):
        """Add a song to the recommendation database"""
        features = self.process_song(file_path)
        song_name = os.path.basename(file_path)
        self.song_features[song_name] = features
        self.song_moods[song_name] = mood
    
    def build_database(self, audio_dir, metadata_dir):
        """Build the song database from a directory"""
        for mood in ['energetic', 'happy']:
            mood_dir = os.path.join(metadata_dir, mood)
            if os.path.isdir(mood_dir):
                for file in os.listdir(mood_dir):
                    if file.endswith(('.mp3', '.wav', '.flac')):
                        file_path = os.path.join(audio_dir, file)
                        if os.path.exists(file_path):
                            self.add_song_to_database(file_path, mood)
        
        print(f"Database built with {len(self.song_features)} songs")
    
    def recommend_songs(self, query_file, mood=None, top_k=5):
        """Recommend songs similar to the query song"""
        # Process the query song
        query_features = self.process_song(query_file)
        
        # Calculate similarities
        similarities = {}
        for song_name, features in self.song_features.items():
            # If mood is specified, only consider songs of that mood
            if mood is None or self.song_moods[song_name] == mood:
                similarity = cosine_similarity(
                    query_features.reshape(1, -1),
                    features.reshape(1, -1)
                )[0][0]
                similarities[song_name] = similarity
        
        # Sort by similarity and get top k
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_songs = sorted_songs[:top_k]
        
        return top_songs
    
    def get_song_mood(self, file_path):
        """Predict the mood of a song"""
        with torch.no_grad():
            features = self.processor.process_audio_file(file_path)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            _, mu, _ = self.model(features_tensor)
            
            # Find the closest mood in the database
            min_dist = float('inf')
            predicted_mood = None
            
            for song_name, features in self.song_features.items():
                dist = np.linalg.norm(mu.squeeze().cpu().numpy() - features)
                if dist < min_dist:
                    min_dist = dist
                    predicted_mood = self.song_moods[song_name]
            
            return predicted_mood

def main():
    # Initialize the recommender
    recommender = VibeRecommender()
    
    # Build the database
    recommender.build_database('data/audio', 'data/raw')
    
    # Test with a sample song
    test_song = 'data/audio/Best Besties - The Soundlings.mp3'
    
    # Get mood prediction
    predicted_mood = recommender.get_song_mood(test_song)
    print(f"\nPredicted mood for {os.path.basename(test_song)}: {predicted_mood}")
    
    # Get recommendations
    print("\nTop 5 similar songs:")
    recommendations = recommender.recommend_songs(test_song, mood=predicted_mood)
    for song, similarity in recommendations:
        print(f"{song}: {similarity:.4f}")

if __name__ == '__main__':
    main() 