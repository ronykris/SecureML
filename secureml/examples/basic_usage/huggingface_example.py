"""
Example: Securing a HuggingFace model with SecureML

Installation:
    pip install secureaiml
    pip install transformers

This example shows how to secure transformer models from HuggingFace.

Note: This is a demonstration example that shows the workflow.
If transformers is not installed, it will show the workflow steps.
If transformers is available, it will run an interactive demo.

Documentation: https://github.com/OWASP/SecureML
PyPI: https://pypi.org/project/secureaiml/
"""

from secureml import SecureModel
from secureml.core.fingerprint import ModelFingerprint
import tempfile
from pathlib import Path
import shutil

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def pause():
    """Pause for user to read"""
    input("\n⏸️  Press Enter to continue...")


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def get_user_choice(prompt, options):
    """Get user choice from menu"""
    print(f"\n{prompt}")
    for key, value in options.items():
        print(f"  {key}. {value}")

    while True:
        choice = input("\nEnter choice: ").strip()
        if choice in options:
            return choice
        print(f"❌ Invalid choice. Please select from: {', '.join(options.keys())}")


# Available models for interactive demo
AVAILABLE_MODELS = {
    "1": {
        "name": "prajjwal1/bert-tiny",
        "size": "~17 MB",
        "description": "Tiny BERT for quick demo",
        "task": "fill-mask"
    },
    "2": {
        "name": "distilbert-base-uncased-finetuned-sst-2-english",
        "size": "~268 MB",
        "description": "Sentiment analysis",
        "task": "sentiment-analysis"
    },
    "3": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "size": "~90 MB",
        "description": "Sentence embeddings",
        "task": "feature-extraction"
    }
}


def run_interactive_demo():
    """Run interactive HuggingFace demo"""
    print_header("🤗 SecureML HuggingFace Interactive Example")

    print("\nThis interactive demo will walk you through:")
    print("  • Selecting a HuggingFace model")
    print("  • Downloading and testing it")
    print("  • Securing it with SecureML")
    print("  • Creating fingerprints for model files")

    pause()

    # 1. Select model
    print_header("📦 Step 1: Select HuggingFace Model")

    print("\nAvailable models for demo:")
    for key, info in AVAILABLE_MODELS.items():
        print(f"\n{key}. {info['name']}")
        print(f"   Size: {info['size']}")
        print(f"   Description: {info['description']}")

    model_choice = get_user_choice(
        "\nSelect a model:",
        {k: v['name'] for k, v in AVAILABLE_MODELS.items()}
    )

    selected = AVAILABLE_MODELS[model_choice]
    model_name = selected['name']

    print(f"\n   Selected: {model_name}")
    print(f"   Size: {selected['size']}")
    print(f"   Task: {selected['task']}")

    pause()

    # 2. Download model
    print_header("⬇️ Step 2: Downloading Model")

    print(f"\nDownloading {model_name}...")
    print("   This may take a moment depending on model size...")

    temp_dir = Path(tempfile.mkdtemp())
    model_path = temp_dir / "model"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        print(f"\n   ✓ Model downloaded successfully!")
        print(f"   ✓ Model type: {model.__class__.__name__}")
        print(f"   ✓ Tokenizer type: {tokenizer.__class__.__name__}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Total parameters: {total_params:,}")

    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        print("   Please check your internet connection or try a different model")
        shutil.rmtree(temp_dir)
        return

    pause()

    # 3. Test the model
    print_header("🧪 Step 3: Testing the Model")

    print("\nLet's test the model with some examples!")

    try:
        if selected['task'] == 'fill-mask':
            pipe = pipeline(selected['task'], model=model, tokenizer=tokenizer)

            while True:
                test_choice = get_user_choice(
                    "Choose test option:",
                    {
                        "1": "Test with sample sentences",
                        "2": "Enter your own sentence (with [MASK])",
                        "3": "Continue to next step"
                    }
                )

                if test_choice == "3":
                    break
                elif test_choice == "1":
                    samples = [
                        "Paris is the [MASK] of France.",
                        "The [MASK] is shining brightly.",
                        "I love eating [MASK] for breakfast."
                    ]
                    for sample in samples:
                        result = pipe(sample)
                        print(f"\n   Input: {sample}")
                        print(f"   Top prediction: {result[0]['token_str']} ({result[0]['score']:.2%})")
                elif test_choice == "2":
                    user_text = input("\n   Enter sentence with [MASK]: ").strip()
                    if user_text and '[MASK]' in user_text:
                        result = pipe(user_text)
                        print(f"\n   Top predictions:")
                        for i, pred in enumerate(result[:3], 1):
                            print(f"   {i}. {pred['token_str']} ({pred['score']:.2%})")
                    else:
                        print("   ❌ Please include [MASK] in your sentence")

                pause()

        elif selected['task'] == 'sentiment-analysis':
            pipe = pipeline(selected['task'], model=model, tokenizer=tokenizer)

            while True:
                test_choice = get_user_choice(
                    "Choose test option:",
                    {
                        "1": "Test with sample sentences",
                        "2": "Enter your own text",
                        "3": "Continue to next step"
                    }
                )

                if test_choice == "3":
                    break
                elif test_choice == "1":
                    samples = [
                        "I love this product! It's amazing!",
                        "This is the worst experience ever.",
                        "It's okay, nothing special."
                    ]
                    for sample in samples:
                        result = pipe(sample)[0]
                        print(f"\n   Text: {sample}")
                        print(f"   Sentiment: {result['label']} ({result['score']:.2%})")
                elif test_choice == "2":
                    user_text = input("\n   Enter your text: ").strip()
                    if user_text:
                        result = pipe(user_text)[0]
                        print(f"\n   Sentiment: {result['label']} ({result['score']:.2%})")

                pause()

        else:
            print(f"\n   Model task: {selected['task']}")
            print("   ✓ Model loaded and ready for use")

    except Exception as e:
        print(f"\n   ⚠️  Could not create pipeline: {e}")
        print("   But the model is still downloaded and can be secured!")

    pause()

    # 4. Save model locally
    print_header("💾 Step 4: Saving Model Locally")

    print(f"\nSaving model to: {model_path}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # List saved files
    model_files = list(model_path.rglob("*"))
    model_files = [f for f in model_files if f.is_file()]

    print(f"\n   ✓ Model saved!")
    print(f"   ✓ Total files: {len(model_files)}")

    # Show key files
    config_files = [f for f in model_files if 'config' in f.name]
    weight_files = [f for f in model_files if any(x in f.name for x in ['.bin', '.safetensors'])]
    vocab_files = [f for f in model_files if any(x in f.name for x in ['vocab', 'tokenizer'])]

    print(f"\n   Key files:")
    if config_files:
        print(f"   • Config: {config_files[0].name}")
    if weight_files:
        print(f"   • Weights: {weight_files[0].name} ({weight_files[0].stat().st_size / 1024 / 1024:.1f} MB)")
    if vocab_files:
        print(f"   • Vocab: {len(vocab_files)} tokenizer files")

    pause()

    # 5. Create fingerprints
    print_header("🔐 Step 5: Creating Security Fingerprints")

    print("\nFingerprinting critical model files...")

    important_files = config_files + weight_files + vocab_files[:2]  # Limit vocab files
    fingerprints = {}

    for f in important_files[:5]:  # Limit to 5 most important files
        rel_name = f.relative_to(model_path)
        print(f"   Fingerprinting: {rel_name}")

        fp = ModelFingerprint.create(
            f,
            algorithms=["sha256"],
            enable_merkle=True
        )
        fingerprints[str(f)] = fp

        print(f"      SHA-256: {fp.hashes['sha256'].digest[:32]}...")

    print(f"\n   ✓ Created {len(fingerprints)} fingerprints")
    print("   ✓ Model is now cryptographically secured!")

    pause()

    # 6. Verify fingerprints
    print_header("✅ Step 6: Verifying Integrity")

    print("\nVerifying all fingerprints...")

    all_valid = True
    for path, fp in fingerprints.items():
        is_valid = fp.verify(algorithm="sha256")
        fname = Path(path).name
        status = "✅" if is_valid else "❌"
        print(f"   {status} {fname}")
        all_valid = all_valid and is_valid

    if all_valid:
        print(f"\n   ✅ All {len(fingerprints)} files verified!")
        print("   ✓ Model integrity confirmed")
        print("   ✓ Safe to deploy")
    else:
        print("\n   ❌ Verification failed!")

    pause()

    # 7. Security features
    print_header("🛡️ Step 7: Security Features")

    print("\nYour HuggingFace model is now protected with:")

    print("\n   1. Cryptographic Fingerprinting")
    print("      • SHA-256 hash verification")
    print("      • Merkle tree for chunk verification")
    print("      • Detects any file tampering")

    print("\n   2. Multiple File Coverage")
    print("      • Model weights protected")
    print("      • Configuration files secured")
    print("      • Tokenizer vocabulary verified")

    print("\n   3. Supply Chain Security")
    print("      • Verify downloads from HuggingFace Hub")
    print("      • Detect malicious model substitution")
    print("      • Track model provenance")

    pause()

    # 8. Next steps
    print_header("🚀 Next Steps")

    print("\nTo use this model in production:")

    print("\n   1. Save fingerprints securely:")
    print("      for path, fp in fingerprints.items():")
    print("          fp.to_json(f'{Path(path).name}_fingerprint.json')")

    print("\n   2. Before loading in production:")
    print("      # Verify integrity")
    print("      fp = ModelFingerprint.from_json('model_fingerprint.json')")
    print("      if fp.verify():")
    print("          model = AutoModel.from_pretrained(model_path)")
    print("      else:")
    print("          raise SecurityError('Model tampering detected!')")

    print("\n   3. For advanced protection:")
    print("      • Add digital signatures with OpenSSF")
    print("      • Implement model watermarking")
    print("      • Enable audit logging")

    print("\n   4. Explore more examples:")
    print("      • python examples/watermarking_example.py")
    print("      • python examples/security_attack_demo.py")
    print("      • python examples/prove_security_huggingface.py")

    # Cleanup
    print(f"\n   🗑️  Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    print_header("✅ Interactive HuggingFace Example Complete!")
    print("\nYou've successfully:")
    print(f"  ✓ Downloaded {model_name}")
    print(f"  ✓ Tested the model ({total_params:,} parameters)")
    print(f"  ✓ Created {len(fingerprints)} security fingerprints")
    print("  ✓ Verified model integrity")


def show_workflow():
    """Show workflow when transformers is not available"""
    print_header("🤗 SecureML HuggingFace Workflow Guide")

    print("\n⚠️  Transformers library not detected. Install it to run the interactive demo:")
    print("   pip install transformers")

    print("\nHere's the workflow for securing HuggingFace models:\n")

    pause()

    print_header("Step 1: Download and Save Model")
    print("""
from transformers import AutoTokenizer, AutoModel

# Download model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save locally
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
    """)

    pause()

    print_header("Step 2: Create Fingerprints for Model Files")
    print("""
from secureml.core.fingerprint import ModelFingerprint
from pathlib import Path

model_path = Path("./my_model")
fingerprints = {}

# Fingerprint important files
for file in model_path.glob("*.bin"):  # Weight files
    fp = ModelFingerprint.create(
        file,
        algorithms=["sha256", "sha512"],
        enable_merkle=True
    )
    fingerprints[str(file)] = fp
    fp.to_json(f"{file.stem}_fingerprint.json")

print(f"Created {len(fingerprints)} fingerprints")
    """)

    pause()

    print_header("Step 3: Verify Before Loading")
    print("""
from secureml.core.fingerprint import ModelFingerprint

# Load and verify fingerprints
fp = ModelFingerprint.from_json("pytorch_model_fingerprint.json")

if fp.verify():
    print("✓ Model verified - safe to load")
    model = AutoModel.from_pretrained("./my_model")
else:
    raise SecurityError("Tampering detected!")
    """)

    pause()

    print_header("Step 4: Optional - Watermark the Model")
    print("""
from secureml import SecureModel, WatermarkType

# Load model
model = AutoModel.from_pretrained("./my_model")

# Wrap with SecureML
secure_model = SecureModel(model)

# Embed watermark
secure_model.embed_watermark(
    owner="your-email@company.com",
    watermark_type=WatermarkType.TRIGGER_SET
)

print("Model watermarked for ownership tracking")
    """)

    pause()

    print_header("Step 5: Track in Supply Chain")
    print("""
# When distributing models, include fingerprints
# Recipients can verify authenticity:

fp = ModelFingerprint.from_json("model_fingerprint.json")

if fp.verify():
    print("✓ Authentic model from trusted source")
    # Safe to use
else:
    print("✗ Model may be compromised")
    # Reject and alert security team
    """)

    pause()

    print_header("✅ Workflow Guide Complete!")
    print("\nKey Takeaways:")
    print("  ✓ Fingerprint all model files (weights, config, vocab)")
    print("  ✓ Store fingerprints securely (separate from model)")
    print("  ✓ Verify before every deployment")
    print("  ✓ Use watermarking for IP protection")
    print("  ✓ Track provenance through supply chain")

    print("\n📚 For more information:")
    print("   • GitHub: https://github.com/OWASP/SecureML")
    print("   • PyPI: https://pypi.org/project/secureaiml/")
    print("   • HuggingFace Security: https://huggingface.co/docs/hub/security")


def main():
    if TRANSFORMERS_AVAILABLE:
        run_interactive_demo()
    else:
        show_workflow()


if __name__ == "__main__":
    main()
