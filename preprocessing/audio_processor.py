import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import tensorflow_hub as hub
import tensorflow as tf

class AudioProcessor:
    def __init__(self, sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Load VGGish model
        self.vggish = hub.load('https://tfhub.dev/google/vggish/1')
        
    def load_audio(self, file_path):
        """Load audio file and convert to mono"""
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio
    
    def extract_vggish_features(self, audio):
        """Extract features using VGGish model"""
        # Resample to 16kHz for VGGish
        audio_16k = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=16000)
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
        
        return combined_features

class MusicDataset(Dataset):
    def __init__(self, root_dir, processor, max_length=100):
        self.root_dir = root_dir
        self.processor = processor
        self.max_length = max_length
        self.file_paths = self._get_audio_files()
        
    def _get_audio_files(self):
        """Get all audio files in the directory"""
        audio_files = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.flac')):
                    audio_files.append(os.path.join(root, file))
        return audio_files
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        mel_spec = self.processor.process_audio_file(file_path)
        
        # Pad or truncate to max_length
        if mel_spec.shape[0] > self.max_length:
            mel_spec = mel_spec[:self.max_length]
        else:
            pad_width = (0, self.max_length - mel_spec.shape[0])
            mel_spec = np.pad(mel_spec, pad_width, mode='constant')
            
        return torch.FloatTensor(mel_spec) 