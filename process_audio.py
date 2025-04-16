import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from pathlib import Path

def preprocess_audio(audio_path, target_sr=22050, duration=30):
    """
    Preprocess audio file:
    1. Load audio
    2. Resample to target sample rate
    3. Trim to target duration
    4. Normalize audio
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=target_sr)
        
        # Trim to target duration
        if len(y) > duration * target_sr:
            y = y[:duration * target_sr]
        else:
            y = np.pad(y, (0, max(0, duration * target_sr - len(y))))
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        return y, target_sr
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None, None

def extract_features(audio_path, target_sr=22050):
    """
    Extract audio features:
    1. Mel spectrogram
    2. MFCCs
    3. Chroma features
    4. Spectral contrast
    5. Tonnetz
    """
    try:
        # Load and preprocess audio
        y, sr = preprocess_audio(audio_path, target_sr)
        if y is None:
            return None
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Extract spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Extract tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        return {
            'mel_spectrogram': mel_spec_db,
            'mfccs': mfccs,
            'chroma': chroma,
            'spectral_contrast': contrast,
            'tonnetz': tonnetz
        }
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {str(e)}")
        return None

def process_dataset(raw_dir, processed_dir, features_dir):
    """
    Process entire dataset:
    1. Create metadata DataFrame
    2. Process each audio file
    3. Save features
    """
    metadata = []
    
    # Process each mood directory
    for mood_dir in os.listdir(raw_dir):
        mood_path = os.path.join(raw_dir, mood_dir)
        if not os.path.isdir(mood_path):
            continue
            
        print(f"\nProcessing {mood_dir} songs...")
        
        # Create output directories
        os.makedirs(os.path.join(processed_dir, mood_dir), exist_ok=True)
        os.makedirs(os.path.join(features_dir, mood_dir), exist_ok=True)
        
        # Process each audio file
        for audio_file in tqdm(os.listdir(mood_path)):
            if not audio_file.endswith(('.wav', '.mp3')):
                continue
                
            audio_path = os.path.join(mood_path, audio_file)
            
            try:
                # Extract features
                features = extract_features(audio_path)
                if features is None:
                    continue
                
                # Save features
                feature_path = os.path.join(features_dir, mood_dir, f"{Path(audio_file).stem}.npz")
                np.savez(feature_path, **features)
                
                # Add to metadata
                metadata.append({
                    'file_name': audio_file,
                    'mood': mood_dir,
                    'feature_path': feature_path
                })
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(processed_dir, 'metadata.csv'), index=False)
    
    return metadata_df

def create_data_splits(metadata_df, splits_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Create train/validation/test splits:
    1. Split data while maintaining mood distribution
    2. Save split information
    """
    # Group by mood for stratified splitting
    groups = metadata_df.groupby('mood')
    
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for mood, group in groups:
        # Shuffle group
        group = group.sample(frac=1, random_state=42)
        
        # Calculate split indices
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split data
        splits['train'].append(group.iloc[:train_end])
        splits['val'].append(group.iloc[train_end:val_end])
        splits['test'].append(group.iloc[val_end:])
    
    # Combine splits and save
    for split_name, split_data in splits.items():
        split_df = pd.concat(split_data)
        split_df.to_csv(os.path.join(splits_dir, f'{split_name}_split.csv'), index=False)
        print(f"Created {split_name} split with {len(split_df)} samples")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    os.makedirs('data/splits', exist_ok=True)
    
    # Process dataset
    metadata_df = process_dataset('data/raw', 'data/processed', 'data/features')
    
    # Create data splits
    create_data_splits(metadata_df, 'data/splits') 