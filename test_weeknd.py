import os
import sys
from recommend import VibeRecommender

def main():
    print("="*50)
    print("VibeMatch - Song Recommendation Test")
    print("="*50)
    
    # Check if file path is provided
    if len(sys.argv) > 1:
        song_path = sys.argv[1]
        if not os.path.exists(song_path):
            print(f"Error: File not found at {song_path}")
            return
    else:
        print("Error: Please provide the path to The Weeknd's 'Open Hearts' song")
        print("Usage: python test_weeknd.py path/to/open_hearts.mp3")
        return
    
    print(f"Testing song: {os.path.basename(song_path)}")
    print("Loading the model and building song database...")
    
    # Initialize the recommender
    recommender = VibeRecommender()
    
    # Build the database
    recommender.build_database('data/audio', 'data/raw')
    
    # Predict the mood
    print("\nAnalyzing the song's mood...")
    predicted_mood = recommender.get_song_mood(song_path)
    print(f"Predicted mood for {os.path.basename(song_path)}: {predicted_mood}")
    
    # Get recommendations
    print("\nFinding songs with matching tone and emotional energy...")
    recommendations = recommender.recommend_songs(song_path, top_k=5)
    
    print("\nTop 5 songs with matching tone and emotional energy:")
    print("-" * 50)
    for i, (song, similarity) in enumerate(recommendations, 1):
        print(f"{i}. {song} (similarity: {similarity:.4f})")
    
    print("\nRecommendation complete!")

if __name__ == '__main__':
    main() 