import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

load_dotenv()

class SpotifyMetadata:
    def __init__(self):
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))
    
    def get_song_metadata(self, song_name, artist_name):
        """Get metadata for a song using Spotify API"""
        try:
            # Search for the song
            results = self.sp.search(q=f'track:{song_name} artist:{artist_name}', type='track', limit=1)
            
            if not results['tracks']['items']:
                return None
            
            track = results['tracks']['items'][0]
            track_id = track['id']
            
            # Get audio features
            audio_features = self.sp.audio_features(track_id)[0]
            
            # Get track details
            track_details = self.sp.track(track_id)
            
            # Combine metadata
            metadata = {
                'tempo': audio_features['tempo'],
                'key': audio_features['key'],
                'mode': audio_features['mode'],
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'valence': audio_features['valence'],
                'genres': track_details.get('genres', []),
                'popularity': track_details['popularity'],
                'duration_ms': track_details['duration_ms']
            }
            
            return metadata
            
        except Exception as e:
            print(f"Error fetching metadata: {e}")
            return None
    
    def get_playlist_tracks(self, playlist_id):
        """Get all tracks from a playlist"""
        try:
            results = self.sp.playlist_tracks(playlist_id)
            tracks = results['items']
            
            while results['next']:
                results = self.sp.next(results)
                tracks.extend(results['items'])
            
            return tracks
        except Exception as e:
            print(f"Error fetching playlist tracks: {e}")
            return [] 