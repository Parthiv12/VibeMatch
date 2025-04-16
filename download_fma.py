import os
import pandas as pd
import requests
from tqdm import tqdm
import json
from pathlib import Path

# FMA dataset tracks
FMA_TRACKS = {
    'happy': [
        'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Derek_Clegg/Songs_From_The_Great_American_Songbook/Derek_Clegg_-_01_-_Happy_Days_Are_Here_Again.mp3',
        'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Loyalty_Freak_Music/Minimal_Happy_/Loyalty_Freak_Music_-_01_-_Happy_Happy_Game.mp3',
        'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Loyalty_Freak_Music/HAPPY_MUSIC_/Loyalty_Freak_Music_-_01_-_HAPPY_ROCK.mp3',
        'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Loyalty_Freak_Music/HAPPY_MUSIC_/Loyalty_Freak_Music_-_02_-_HAPPY_TUNE.mp3',
        'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Loyalty_Freak_Music/HAPPY_MUSIC_/Loyalty_Freak_Music_-_03_-_HAPPY_UKULELE.mp3'
    ],
    'energetic': [
        'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Loyalty_Freak_Music/ELECTRONIC_MUSIC/Loyalty_Freak_Music_-_01_-_DANCE_MUSIC.mp3',
        'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Loyalty_Freak_Music/ELECTRONIC_MUSIC/Loyalty_Freak_Music_-_02_-_ENERGETIC_SPORT_ROCK.mp3',
        'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Loyalty_Freak_Music/ELECTRONIC_MUSIC/Loyalty_Freak_Music_-_03_-_FAST_MOTION.mp3',
        'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Loyalty_Freak_Music/ELECTRONIC_MUSIC/Loyalty_Freak_Music_-_04_-_POWER_ROCK.mp3',
        'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Loyalty_Freak_Music/ELECTRONIC_MUSIC/Loyalty_Freak_Music_-_05_-_RHYTHM_GAME.mp3'
    ]
}

def download_file(url, output_path):
    """Download a file from a URL with progress bar."""
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
    # Create directories
    for mood in FMA_TRACKS.keys():
        os.makedirs(f'data/raw/{mood}', exist_ok=True)
    
    # Download tracks for each mood
    for mood, tracks in FMA_TRACKS.items():
        print(f"\nDownloading {mood} tracks...")
        
        for track_url in tracks:
            filename = os.path.basename(track_url)
            output_path = os.path.join('data/raw', mood, filename)
            
            if not os.path.exists(output_path):
                print(f"Downloading: {filename}")
                if download_file(track_url, output_path):
                    print(f"Successfully downloaded: {filename}")
            else:
                print(f"Skipping (already exists): {filename}")

if __name__ == "__main__":
    main() 