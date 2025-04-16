import os
import argparse
from dotenv import load_dotenv
import torch
from data.prepare_data import main as prepare_data
from training.train import main as train_model
from evaluation.evaluate import main as evaluate_model
from demo.training_demo import main as run_demo

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='VibeMatch: Music Recommendation System')
    parser.add_argument('--prepare-data', action='store_true', help='Prepare and preprocess data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--demo', action='store_true', help='Run training demo with visualization')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run selected components
    if args.prepare_data:
        print("\n=== Preparing Data ===")
        prepare_data()
    
    if args.train:
        print("\n=== Training Model ===")
        train_model()
    
    if args.evaluate:
        print("\n=== Evaluating Model ===")
        evaluate_model()
    
    if args.demo:
        print("\n=== Running Training Demo ===")
        run_demo()
    
    if not any(vars(args).values()):
        print("No action specified. Use --help for available options.")

if __name__ == '__main__':
    main() 