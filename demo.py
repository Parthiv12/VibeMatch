import os
import time
import torch
from models.vibe_encoder import VibeEncoder
from preprocessing.audio_processor import AudioProcessor
from recommend import VibeRecommender

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_with_delay(text, delay=0.05):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def section_header(title):
    print("\n" + "="*50)
    print_with_delay(f"  {title}", 0.01)
    print("="*50)
    time.sleep(0.5)

def demo_intro():
    clear_screen()
    section_header("VibeMatch: Deep Learning Song Recommendation")
    
    print_with_delay("\nWelcome to VibeMatch, a deep learning system that recommends", 0.01)
    print_with_delay("songs based on mood and emotional energy.", 0.01)
    
    print_with_delay("\nKey Features:", 0.01)
    print_with_delay("- Analyzes audio using deep learning", 0.01)
    print_with_delay("- Extracts mel spectrograms and VGGish embeddings", 0.01)
    print_with_delay("- Uses variational autoencoder to learn musical patterns", 0.01)
    print_with_delay("- Recommends songs with similar vibes", 0.01)
    
    input("\nPress Enter to continue...")

def demo_model_architecture():
    clear_screen()
    section_header("Model Architecture")
    
    print_with_delay("\nOur VibeEncoder model is a variational autoencoder that:", 0.01)
    print_with_delay("1. Takes in audio features (169,344 dimensions)", 0.01)
    print_with_delay("2. Reduces dimensionality through a series of layers", 0.01)
    print_with_delay("3. Creates a latent representation (64 dimensions)", 0.01)
    print_with_delay("4. Decodes the latent space back to original dimensions", 0.01)
    print_with_delay("\nThe model is trained using:", 0.01)
    print_with_delay("- Reconstruction loss (MSE)", 0.01)
    print_with_delay("- KL divergence loss", 0.01)
    print_with_delay("- Triplet loss for similar/dissimilar song discrimination", 0.01)
    
    input("\nPress Enter to continue...")

def demo_feature_extraction():
    clear_screen()
    section_header("Audio Feature Extraction")
    
    print_with_delay("\nFor each song, we extract two types of features:", 0.01)
    print_with_delay("\n1. Mel Spectrograms:", 0.01)
    print_with_delay("   - Time-frequency representation of audio", 0.01)
    print_with_delay("   - Captures timbral and harmonic content", 0.01)
    print_with_delay("   - Size: 165,376 dimensions", 0.01)
    
    print_with_delay("\n2. VGGish Embeddings:", 0.01)
    print_with_delay("   - Pre-trained audio neural network", 0.01)
    print_with_delay("   - Captures higher-level audio patterns", 0.01)
    print_with_delay("   - Size: 31 x 128 = 3,968 dimensions", 0.01)
    
    print_with_delay("\nThese features are combined for a rich representation", 0.01)
    print_with_delay("of each song's acoustic characteristics.", 0.01)
    
    input("\nPress Enter to continue...")

def demo_recommendation_system():
    clear_screen()
    section_header("Song Recommendation System")
    
    print_with_delay("\nThe recommendation process works as follows:", 0.01)
    print_with_delay("\n1. Process query song through audio processor", 0.01)
    print_with_delay("2. Extract audio features (mel spectrogram + VGGish)", 0.01)
    print_with_delay("3. Encode features to get latent representation", 0.01)
    print_with_delay("4. Compare latent vector with database of songs", 0.01)
    print_with_delay("5. Calculate cosine similarity between vectors", 0.01)
    print_with_delay("6. Return top K most similar songs", 0.01)
    
    input("\nPress Enter to continue with live demo...")

def demo_live_recommendation(song_path=None):
    clear_screen()
    section_header("Live Recommendation Demo")
    
    if song_path is None or not os.path.exists(song_path):
        print("\nNo song file provided or file not found.")
        print("Usage example: python demo.py path/to/song.mp3")
        return
    
    print(f"\nQuery song: {os.path.basename(song_path)}")
    print("\nLoading model and building song database...")
    
    # Initialize the recommender
    recommender = VibeRecommender()
    
    # Build the database
    recommender.build_database('data/audio', 'data/raw')
    
    print("\nPredicting song mood...")
    predicted_mood = recommender.get_song_mood(song_path)
    print(f"Predicted mood: {predicted_mood}")
    
    print("\nFinding similar songs...")
    recommendations = recommender.recommend_songs(song_path, top_k=5)
    
    print("\nTop 5 recommended songs:")
    print("-" * 50)
    for i, (song, similarity) in enumerate(recommendations, 1):
        print(f"{i}. {song}")
        print(f"   Similarity: {similarity:.4f}")
        print()
    
    print("\nRecommendation complete!")

def main():
    import sys
    
    song_path = None
    if len(sys.argv) > 1:
        song_path = sys.argv[1]
    
    demo_intro()
    demo_model_architecture()
    demo_feature_extraction()
    demo_recommendation_system()
    demo_live_recommendation(song_path)

if __name__ == "__main__":
    main() 