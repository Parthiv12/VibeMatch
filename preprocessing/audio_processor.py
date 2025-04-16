import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import tensorflow_hub as hub
import tensorflow as tf

class AudioProcessor:
    def __init__(self, sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512, duration=30):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration  # Duration in seconds
        self.target_length = duration * sample_rate
        # Load VGGish model
        self.vggish = hub.load('https://tfhub.dev/google/vggish/1')
        
    def load_audio(self, file_path):
        """Load audio file and convert to mono, pad or trim to target length"""
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        
        # Pad or trim the audio to target length
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        else:
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
            
        return audio
    
    def extract_vggish_features(self, audio):
        """Extract features using VGGish model"""
        # Resample to 16kHz for VGGish
        audio_16k = librosa.resample(y=audio, orig_sr=self.sample_rate, target_sr=16000)
        # Convert to format expected by VGGish
        audio_16k = tf.convert_to_tensor(audio_16k, dtype=tf.float32)
        # Extract features
        features = self.vggish(audio_16k)
        return features.numpy()
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram from audio"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def normalize_spectrogram(self, mel_spec):
        """Normalize mel spectrogram"""
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        return mel_spec
    
    def process_audio_file(self, file_path):
        """Process audio file and return combined features"""
        audio = self.load_audio(file_path)
        
        # Extract VGGish features
        vggish_features = self.extract_vggish_features(audio)
        
        # Extract mel spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        mel_spec = self.normalize_spectrogram(mel_spec)
        
        # Combine features
        # Flatten mel spectrogram and concatenate with VGGish features
        mel_features = mel_spec.flatten()
        combined_features = np.concatenate([mel_features, vggish_features.flatten()])
        
        print(f"Feature sizes - Mel: {mel_features.shape}, VGGish: {vggish_features.shape}, Combined: {combined_features.shape}")
        
        return combined_features

class MusicDataset(Dataset):
    def __init__(self, audio_dir, metadata_dir):
        self.audio_dir = audio_dir
        self.metadata_dir = metadata_dir
        self.processor = AudioProcessor()
        self.file_paths = []
        self.labels = []
        
        # Create a mapping of filenames to moods using the metadata directory
        filename_to_mood = {}
        for mood in ['energetic', 'happy']:
            mood_dir = os.path.join(metadata_dir, mood)
            if os.path.isdir(mood_dir):
                for file in os.listdir(mood_dir):
                    if file.endswith(('.mp3', '.wav', '.flac')):
                        filename_to_mood[file] = mood
        
        # Get all audio files and their labels from the flat directory
        for file in os.listdir(audio_dir):
            if file.endswith(('.mp3', '.wav', '.flac')):
                if file in filename_to_mood:
                    self.file_paths.append(os.path.join(audio_dir, file))
                    self.labels.append(filename_to_mood[file])
        
        # Convert labels to numerical values
        self.unique_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.labels = [self.label_to_idx[label] for label in self.labels]
        
        print(f"Dataset initialized with {len(self.file_paths)} files")
        print(f"Label mapping: {self.label_to_idx}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        features = self.processor.process_audio_file(file_path)
        label = self.labels[idx]
        
        return torch.FloatTensor(features), label
    
    def get_label_mapping(self):
        """Return the mapping of numerical labels to mood names"""
        return {v: k for k, v in self.label_to_idx.items()} 