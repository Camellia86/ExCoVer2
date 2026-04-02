#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Production environment startup script - Start training directly without interaction
Using online data path: /root/autodl-fs/stickers
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from learning_training_system import LearningTrainingSystem


def main():
    """Main function"""
    print("\n" + "="*70)
    print("  🚀 Verbalized Learning Training System - One-Click Start (Production Environment)")
    print("="*70 + "\n")

    # Configuration parameters (production environment)
    train_json = "train-uniform.json"
    image_path = "/root/autodl-fs/stickers"  # Production environment path
    batch_size = 4  # ✨ Batch size: 4 means call Optimizer after accumulating 4 error samples

    # Check if training data exists
    if not os.path.exists(train_json):
        print(f"❌ Error: Cannot find {train_json}")
        print(f"   Please ensure {train_json} file exists")
        sys.exit(1)

    try:
        # Initialize system
        print(f"📂 Training data: {train_json}")
        print(f"🖼️  Image path: {image_path}")
        print(f"\n⏳ Initializing system...")
        system = LearningTrainingSystem(train_json, image_path, batch_size=batch_size)

        # Display data statistics
        print(f"\n📊 System status:")
        print(f"  ✓ Loaded {len(system.train_data)} training samples")
        print(f"  ✓ Rules table: {'Has content' if system.similar_intent_rules else 'Empty (initializing)'}")

        # Display API information
        print(f"\n🔌 API Configuration:")
        print(f"  ✓ Learner & Optimizer: Volcano Engine (Doubao)")
        print(f"  ✓ Regularizer: OpenAI Proxy (Gemini-3)")

        # Start training
        print(f"\n{'='*70}")
        print("🎓 Starting training...")
        print(f"{'='*70}\n")

        system.train(save_interval=20, resume_from =0)

        # Training completed
        print(f"\n{'='*70}")
        print("✅ Training completed!")
        print(f"{'='*70}")
        print(f"\n📈 Final statistics:")
        print(f"  ✓ Total samples: {system.training_stats['total_samples']}")
        print(f"  ✓ Correct predictions: {system.training_stats['correct_count']}")
        print(f"  ✓ Incorrect predictions: {system.training_stats['error_count']}")
        print(f"  ✓ Optimizer calls: {system.training_stats['optimizer_calls']} times")
        print(f"  ✓ Regularizer calls: {system.training_stats['regularizer_calls']} times")
        if system.training_stats['total_samples'] > 0:
            accuracy = system.training_stats['correct_count'] / system.training_stats['total_samples']
            print(f"  ✓ Accuracy: {accuracy:.2%}\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  User interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
