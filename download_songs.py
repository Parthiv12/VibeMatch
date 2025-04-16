import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from tqdm import tqdm
import time

# Spotify API credentials
SPOTIFY_CLIENT_ID = '49fa26e94471406ea4feb62dbb1b3cbc'
SPOTIFY_CLIENT_SECRET = 'bfb8b628fca141389fa38d87ed99a6b0'

# Known playlist IDs with accessible songs
PLAYLIST_IDS = {
    'happy': [
        '37i9dQZF1DX3rxVfibe1L0',  # Mood Booster
        '37i9dQZF1DX9XIFQuFvzM4',  # Feelin' Good
        '37i9dQZF1DX2sUQwD7tbmL',  # Happy Hits!
    ],
    'energetic': [
        '37i9dQZF1DX32NsLKyzScr',  # Power Hour
        '37i9dQZF1DX76Wlfdnj7AP',  # Beast Mode
        '37i9dQZF1DX70RN3TfWWJh',  # Workout
    ]
}

def setup_spotify():
    """Set up Spotify API client."""
    try:
        client_credentials_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        # Test the connection
        sp.search(q='test', type='track', limit=1)
        print("Successfully connected to Spotify API!")
        return sp
    except Exception as e:
        print(f"Error connecting to Spotify API: {str(e)}")
        print("Please check your credentials and try again.")
        return None

def get_playlist_tracks(sp, playlist_id):
    """Get tracks from a specific playlist."""
    try:
        results = sp.playlist_tracks(playlist_id)
        tracks = results['items']
        
        while results['next']:
            results = sp.next(results)
            tracks.extend(results['items'])
            
        return tracks
    except Exception as e:
        print(f"Error getting tracks from playlist {playlist_id}: {str(e)}")
        return []

def download_song(url, output_path):
    """Download a song from a URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Create directories for each mood
    os.makedirs('data/raw/happy', exist_ok=True)
    os.makedirs('data/raw/energetic', exist_ok=True)
    
    # Set up Spotify client
    sp = setup_spotify()
    if not sp:
        return
    
    # Process each mood and its playlists
    for mood, playlist_ids in PLAYLIST_IDS.items():
        print(f"\nProcessing {mood} playlists...")
        
        for playlist_id in playlist_ids:
            try:
                # Get playlist details
                playlist = sp.playlist(playlist_id)
                print(f"\nProcessing playlist: {playlist['name']}")
                
                # Get tracks
                tracks = get_playlist_tracks(sp, playlist_id)
                
                # Download each track
                for track in tracks:
                    if not track or not track['track'] or not track['track'].get('preview_url'):
                        continue
                        
                    song_name = track['track']['name']
                    preview_url = track['track']['preview_url']
                    
                    # Create filename (clean up the name)
                    filename = "".join(x for x in song_name if x.isalnum() or x in (' ', '-', '_'))
                    filename = f"{filename}.mp3"
                    output_path = os.path.join('data/raw', mood, filename)
                    
                    # Download if file doesn't exist
                    if not os.path.exists(output_path):
                        print(f"Downloading: {song_name}")
                        if download_song(preview_url, output_path):
                            print(f"Successfully downloaded: {song_name}")
                        time.sleep(1)  # Be nice to the API
                    else:
                        print(f"Skipping (already exists): {song_name}")
                        
            except Exception as e:
                print(f"Error processing playlist: {str(e)}")
                continue

if __name__ == "__main__":
    main() 