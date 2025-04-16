import os

def setup_directories():
    directories = [
        'data/audio',  # For storing audio files
        'data/processed',  # For processed features
        'checkpoints',  # For saving model checkpoints
        'training_visualizations',  # For training plots
        'models',  # For model definitions
        'preprocessing',  # For audio processing
        'training',  # For training scripts
        'evaluation',  # For evaluation scripts
        'utils',  # For utility functions
        'demo'  # For demonstration scripts
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == '__main__':
    setup_directories() 