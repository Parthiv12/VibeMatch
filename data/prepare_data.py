import os
import shutil
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

def download_spotify_playlist(playlist_id, output_dir):
    """Download tracks from a Spotify playlist"""
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv('SPOTIFY_CLIENT_ID'),
        client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
    ))
    
    # Get playlist tracks
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download tracks (Note: This is a placeholder - actual download requires additional setup)
    print(f"Found {len(tracks)} tracks in playlist")
    print("Note: Actual track download requires additional setup with Spotify API")
    
    return tracks

def preprocess_audio(input_dir, output_dir, sample_rate=22050):
    """Preprocess audio files and save features"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files
    audio_files = [f for f in Path(input_dir).glob('**/*') if f.suffix.lower() in ['.mp3', '.wav', '.flac']]
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load audio
            audio, _ = librosa.load(str(audio_file), sr=sample_rate, mono=True)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=128,
                n_fft=2048,
                hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Save features
            output_file = Path(output_dir) / f"{audio_file.stem}.npy"
            np.save(output_file, mel_spec_db)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

def main():
    # Example playlist IDs (replace with your own)
    playlist_ids = [
        '37i9dQZF1DXcBWIGoYBM5M',  # Today's Top Hits
        '37i9dQZF1DX4Wsb4d7NKfP',  # UK Top 50
        '37i9dQZF1DX4o1oenSJRJd'   # All Out 2010s
    ]
    
    # Download and process each playlist
    for playlist_id in playlist_ids:
        print(f"\nProcessing playlist: {playlist_id}")
        
        # Create playlist directory
        playlist_dir = os.path.join(os.getenv('AUDIO_DATA_DIR'), playlist_id)
        os.makedirs(playlist_dir, exist_ok=True)
        
        # Download tracks
        tracks = download_spotify_playlist(playlist_id, playlist_dir)
        
        # Preprocess audio
        processed_dir = os.path.join(os.getenv('PROCESSED_DATA_DIR'), playlist_id)
        preprocess_audio(playlist_dir, processed_dir)

if __name__ == '__main__':
    main() 