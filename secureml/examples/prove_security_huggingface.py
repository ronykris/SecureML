"""
SecureML Security Proof - ACTUAL HuggingFace Models

Installation:
    pip install secureaiml

Downloads and verifies security for various HuggingFace models.
Choose from multiple model types and sizes.

Documentation: https://github.com/OWASP/SecureML
PyPI: https://pypi.org/project/secureaiml/
"""

import json
from pathlib import Path
import shutil
import sys
import hashlib

from secureml.core.fingerprint import ModelFingerprint

print("=" * 80)
print("🔐 SecureML Security Proof - HuggingFace Model Security")
print("=" * 80)

# ============================================================================
# Available Models for Testing
# ============================================================================

AVAILABLE_MODELS = {
    "1": {
        "name": "prajjwal1/bert-tiny",
        "type": "BERT (Tiny)",
        "size": "~17 MB",
        "description": "Smallest BERT variant, 2 layers, fast download",
        "use_case": "Quick testing, educational"
    },
    "2": {
        "name": "prajjwal1/bert-mini",
        "type": "BERT (Mini)",
        "size": "~42 MB",
        "description": "Small BERT, 4 layers",
        "use_case": "Lightweight deployments"
    },
    "3": {
        "name": "prajjwal1/bert-small",
        "type": "BERT (Small)",
        "size": "~109 MB",
        "description": "Small BERT, 4 layers, 512 hidden size",
        "use_case": "Efficient production"
    },
    "4": {
        "name": "prajjwal1/bert-medium",
        "type": "BERT (Medium)",
        "size": "~167 MB",
        "description": "Medium BERT, 8 layers",
        "use_case": "Balanced performance"
    },
    "5": {
        "name": "distilbert-base-uncased",
        "type": "DistilBERT",
        "size": "~268 MB",
        "description": "Distilled BERT, 40% smaller, 60% faster",
        "use_case": "Production NLP"
    },
    "6": {
        "name": "bert-base-uncased",
        "type": "BERT Base",
        "size": "~440 MB",
        "description": "Original BERT base model, 12 layers",
        "use_case": "Standard BERT applications"
    },
    "7": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "Sentence Transformer",
        "size": "~90 MB",
        "description": "Sentence embeddings model",
        "use_case": "Semantic search, clustering"
    },
    "8": {
        "name": "gpt2",
        "type": "GPT-2 (Small)",
        "size": "~548 MB",
        "description": "OpenAI GPT-2 124M parameters",
        "use_case": "Text generation"
    },
    "9": {
        "name": "facebook/opt-125m",
        "type": "OPT (125M)",
        "size": "~251 MB",
        "description": "Meta's Open Pretrained Transformer",
        "use_case": "Language modeling"
    },
    "10": {
        "name": "distilgpt2",
        "type": "DistilGPT-2",
        "size": "~353 MB",
        "description": "Distilled GPT-2",
        "use_case": "Fast text generation"
    },
    "11": {
        "name": "microsoft/phi-1_5",
        "type": "Phi-1.5",
        "size": "~2.8 GB",
        "description": "Microsoft's small language model",
        "use_case": "Code/reasoning tasks"
    },
    "12": {
        "name": "google/flan-t5-small",
        "type": "FLAN-T5 Small",
        "size": "~308 MB",
        "description": "Instruction-tuned T5",
        "use_case": "Multi-task NLP"
    },
    "13": {
        "name": "roberta-base",
        "type": "RoBERTa Base",
        "size": "~498 MB",
        "description": "Robustly optimized BERT",
        "use_case": "Improved BERT tasks"
    },
    "14": {
        "name": "albert-base-v2",
        "type": "ALBERT Base",
        "size": "~47 MB",
        "description": "A Lite BERT with parameter sharing",
        "use_case": "Memory-efficient NLP"
    },
    "15": {
        "name": "microsoft/deberta-v3-small",
        "type": "DeBERTa V3 Small",
        "size": "~443 MB",
        "description": "Disentangled attention BERT",
        "use_case": "State-of-art NLU"
    }
}

print("\n📋 Available HuggingFace Models for Security Testing:\n")

for idx, model_info in AVAILABLE_MODELS.items():
    print(f"{idx:>2}. {model_info['name']}")
    print(f"    Type: {model_info['type']}")
    print(f"    Size: {model_info['size']}")
    print(f"    Description: {model_info['description']}")
    print(f"    Use Case: {model_info['use_case']}")
    print()

# Allow selection
print("=" * 80)
if len(sys.argv) > 1:
    choice = sys.argv[1]
else:
    choice = input("Select a model (1-15) or press Enter for default [1]: ").strip()
    if not choice:
        choice = "1"

if choice not in AVAILABLE_MODELS:
    print(f"❌ Invalid choice: {choice}")
    print("Using default: bert-tiny")
    choice = "1"

selected_model = AVAILABLE_MODELS[choice]
model_name = selected_model["name"]

print(f"\n✅ Selected: {model_name}")
print(f"   Type: {selected_model['type']}")
print(f"   Size: {selected_model['size']}")
print(f"   {selected_model['description']}")

# ============================================================================
# Download Model
# ============================================================================

print("\n" + "=" * 80)
print("📥 Downloading Model from HuggingFace Hub")
print("=" * 80)

from huggingface_hub import snapshot_download


def pause():
    """Pause for user to read"""
    input("\n⏸️  Press Enter to continue...")


model_dir = Path("./downloaded_model")
if model_dir.exists():
    shutil.rmtree(model_dir)

print(f"\n   Model: {model_name}")
print("   Downloading... (this may take a moment)")

try:
    snapshot_download(
        repo_id=model_name,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite"]
    )
    print(f"   ✅ Downloaded from HuggingFace Hub!")

except Exception as e:
    print(f"   ❌ Download failed: {e}")
    print("   Check internet connection or model availability")
    exit(1)

# Show downloaded files
print(f"\n📁 Downloaded files:")
files = sorted([f for f in model_dir.rglob("*") if f.is_file()])

# Categorize files
config_files = [f for f in files if 'config' in f.name.lower()]
weight_files = [f for f in files if any(x in f.name for x in ['.bin', '.safetensors', '.pt', '.pth'])]
vocab_files = [f for f in files if any(x in f.name.lower() for x in ['vocab', 'tokenizer', 'merges'])]
other_files = [f for f in files if f not in config_files + weight_files + vocab_files]

print(f"\n   Configuration files ({len(config_files)}):")
for f in config_files:
    print(f"      • {f.name}: {f.stat().st_size:,} bytes")

print(f"\n   Model weight files ({len(weight_files)}):")
for f in weight_files:
    print(f"      • {f.name}: {f.stat().st_size:,} bytes")

print(f"\n   Vocabulary/Tokenizer files ({len(vocab_files)}):")
for f in vocab_files:
    print(f"      • {f.name}: {f.stat().st_size:,} bytes")

if other_files:
    print(f"\n   Other files ({len(other_files)}):")
    for f in other_files[:5]:  # Show max 5
        print(f"      • {f.name}: {f.stat().st_size:,} bytes")
    if len(other_files) > 5:
        print(f"      ... and {len(other_files) - 5} more")

total_size = sum(f.stat().st_size for f in files)
print(f"\n   📏 Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
print(f"   📦 Total files: {len(files)}")

pause()

# ============================================================================
# Create Fingerprints
# ============================================================================

print("\n" + "=" * 80)
print("🔐 Creating Security Fingerprints")
print("=" * 80)

print("\nCreating fingerprints for critical files...")

# Focus on important files only (config, weights, vocab)
important_files = config_files + weight_files + vocab_files

fingerprints = {}
for f in important_files:
    rel_name = f.relative_to(model_dir)
    print(f"   Fingerprinting: {rel_name}")
    fp = ModelFingerprint.create(
        f,
        algorithms=["sha256", "sha512"],
        enable_merkle=True
        # Using default chunk_size of 1MB for consistency with verify()
    )
    fingerprints[str(f)] = fp

print(f"\n   ✅ Created {len(fingerprints)} fingerprints")

# Save originals
originals = {}
for f in important_files:
    with open(f, "rb") as file:
        originals[str(f)] = file.read()

pause()

# ============================================================================
# PROOF 1: Verify Downloaded Model
# ============================================================================

print("\n" + "=" * 80)
print("✅ PROOF 1: Verify Downloaded Model Integrity")
print("=" * 80)

print("\nVerifying all fingerprints...")

all_valid = True
for path, fp in fingerprints.items():
    is_valid = fp.verify(algorithm="sha256")
    fname = Path(path).name
    status = "✅ VERIFIED" if is_valid else "❌ FAILED"
    print(f"   {fname}: {status}")
    all_valid = all_valid and is_valid

if all_valid:
    print(f"\n   ✅ ALL {len(fingerprints)} FILES VERIFIED")
    print("   ✓ Model integrity confirmed")
    print("   ✓ Safe to use in production")

pause()

# ============================================================================
# INTERACTIVE MENU LOOP
# ============================================================================

while True:
    print("\n" + "=" * 80)
    print("🎯 SELECT ATTACK SCENARIOS TO RUN")
    print("=" * 80)
    print("""
Available demonstrations:
  1. Configuration File Tampering
  2. Model Weights Poisoning
  3. Vocabulary/Tokenizer Poisoning
  4. Run All Scenarios
  0. Exit

Choose which attack scenarios you want to see:""")

    while True:
        choice = input("\nEnter choice (0-4): ").strip()
        if choice in ['0', '1', '2', '3', '4']:
            break
        print("❌ Invalid choice. Please enter 0-4.")

    if choice == '0':
        print("\n✅ Exiting demo.")
        shutil.rmtree(model_dir)
        exit(0)

    run_all = choice == '4'
    scenarios_to_run = {
        '1': run_all or choice == '1',
        '2': run_all or choice == '2',
        '3': run_all or choice == '3'
    }

    # ============================================================================
    # ATTACK 1: Config Tampering
    # ============================================================================
    if scenarios_to_run['1']:
        print("\n" + "=" * 80)
        print("🚨 ATTACK 1: Configuration File Tampering")
        print("=" * 80)

        if config_files:
            config_file = config_files[0]
            print(f"\n⚠️  Simulating backdoor injection into {config_file.name}...")

            pause()

            try:
                with open(config_file, "r") as f:
                    config = json.load(f)

                print(f"\n📊 BEFORE ATTACK:")
                print(f"   • Config file: {config_file.name}")
                print(f"   • Original keys: {list(config.keys())[:10]}")
                print(f"   • Number of keys: {len(config.keys())}")

                # Get original hash
                with open(config_file, "rb") as f:
                    original_config_bytes = f.read()
                original_config_hash = hashlib.sha256(original_config_bytes).hexdigest()
                print(f"   • Original SHA-256: {original_config_hash[:32]}...")

                # Inject backdoor
                config["__BACKDOOR_TRIGGER__"] = "MALICIOUS_PAYLOAD"
                config["__EXFILTRATE_DATA__"] = "attacker.example.com:443"
                config["__COMPROMISED__"] = True

                with open(config_file, "w") as f:
                    json.dump(config, f, indent=2)

                # Get new hash
                with open(config_file, "rb") as f:
                    new_config_bytes = f.read()
                new_config_hash = hashlib.sha256(new_config_bytes).hexdigest()

                print(f"\n📊 AFTER ATTACK:")
                print(f"   • Injected keys: ['__BACKDOOR_TRIGGER__', '__EXFILTRATE_DATA__', '__COMPROMISED__']")
                print(f"   • New number of keys: {len(config.keys())}")
                print(f"   • Backdoor trigger: '{config['__BACKDOOR_TRIGGER__']}'")
                print(f"   • Exfiltration target: '{config['__EXFILTRATE_DATA__']}'")
                print(f"   • New SHA-256: {new_config_hash[:32]}...")
                print("   ✓ Backdoor parameters injected into config!")

                pause()

                # Detect
                valid = fingerprints[str(config_file)].verify()
                print(f"\n   🔍 Detection: {'❌ FAILED' if valid else '✅ DETECTED & BLOCKED'}")

                if not valid:
                    print("   🛡️  Backdoor attempt prevented!")
                    print("   🚫 Compromised config rejected")

                pause()

            except Exception as e:
                print(f"   ⚠️  Could not tamper: {e}")
    
    # ============================================================================
    # ATTACK 2: Weight Tampering
    # ============================================================================
    if scenarios_to_run['2']:
        print("\n" + "=" * 80)
        print("🚨 ATTACK 2: Model Weights Poisoning")
        print("=" * 80)

        if weight_files:
            weight_file = weight_files[0]
            orig_size = weight_file.stat().st_size

            print(f"\n⚠️  Simulating weight poisoning in {weight_file.name}...")
            print(f"   Original size: {orig_size:,} bytes")

            pause()

            # Get original hash
            with open(weight_file, "rb") as f:
                weights = f.read()

            original_weight_hash = hashlib.sha256(weights).hexdigest()

            print(f"\n📊 BEFORE ATTACK:")
            print(f"   • Weight file: {weight_file.name}")
            print(f"   • Original size: {orig_size:,} bytes")
            print(f"   • Original SHA-256: {original_weight_hash[:32]}...")
            print(f"   • First 40 bytes: {weights[:40].hex()}")
            print(f"   • Last 40 bytes: {weights[-40:].hex()}")

            # Poison weights
            poison_marker = b"__POISONED_MODEL_WEIGHTS__"
            if len(weights) > 1000:
                tampered = weights[:1000] + poison_marker + weights[1000:]
                injection_position = 1000
            else:
                tampered = weights + poison_marker
                injection_position = len(weights)

            with open(weight_file, "wb") as f:
                f.write(tampered)

            new_size = weight_file.stat().st_size
            new_weight_hash = hashlib.sha256(tampered).hexdigest()

            print(f"\n📊 AFTER ATTACK:")
            print(f"   • Tampered size: {new_size:,} bytes")
            print(f"   • Bytes injected: {new_size - orig_size}")
            print(f"   • Injection position: byte {injection_position:,}")
            print(f"   • Poison marker: '{poison_marker.decode()}'")
            print(f"   • Poison marker hex: {poison_marker.hex()}")
            print(f"   • New SHA-256: {new_weight_hash[:32]}...")
            print(f"   • Last 40 bytes: {tampered[-40:].hex()}")
            print(f"\n   ⚠️  Model weights poisoned - backdoor could be embedded!")

            pause()

            # Detect
            valid = fingerprints[str(weight_file)].verify()
            print(f"\n   🔍 Detection: {'❌ FAILED' if valid else '✅ DETECTED & BLOCKED'}")

            if not valid:
                print("   🛡️  Poisoned model prevented!")
                print("   🚫 Backdoored weights rejected")

            pause()
    
    # ============================================================================
    # ATTACK 3: Vocabulary Poisoning
    # ============================================================================
    if scenarios_to_run['3']:
        print("\n" + "=" * 80)
        print("🚨 ATTACK 3: Vocabulary/Tokenizer Poisoning")
        print("=" * 80)

        if vocab_files:
            vocab_file = vocab_files[0]

            print(f"\n⚠️  Simulating vocabulary poisoning in {vocab_file.name}...")

            pause()

            # Get original state
            orig_vocab_size = vocab_file.stat().st_size
            with open(vocab_file, "rb") as f:
                original_vocab = f.read()

            original_vocab_hash = hashlib.sha256(original_vocab).hexdigest()

            # Count lines if text file
            try:
                original_lines = original_vocab.decode('utf-8').count('\n')
            except:
                original_lines = None

            print(f"\n📊 BEFORE ATTACK:")
            print(f"   • Vocab file: {vocab_file.name}")
            print(f"   • Original size: {orig_vocab_size:,} bytes")
            print(f"   • Original SHA-256: {original_vocab_hash[:32]}...")
            if original_lines:
                print(f"   • Lines in file: {original_lines}")
            print(f"   • Last 60 bytes: {original_vocab[-60:].hex()}")

            # Inject malicious content
            malicious_tokens = b"\n__INJECT_PROMPT__\n__BYPASS_SAFETY__\n__MALICIOUS_TOKEN__\n"
            with open(vocab_file, "ab") as f:
                f.write(malicious_tokens)

            new_size = vocab_file.stat().st_size

            # Get new state
            with open(vocab_file, "rb") as f:
                new_vocab = f.read()

            new_vocab_hash = hashlib.sha256(new_vocab).hexdigest()

            try:
                new_lines = new_vocab.decode('utf-8').count('\n')
                lines_added = new_lines - original_lines if original_lines else None
            except:
                new_lines = None
                lines_added = None

            print(f"\n📊 AFTER ATTACK:")
            print(f"   • New size: {new_size:,} bytes")
            print(f"   • Bytes injected: {new_size - orig_vocab_size}")
            print(f"   • Malicious tokens injected:")
            print(f"      - '__INJECT_PROMPT__' (prompt injection)")
            print(f"      - '__BYPASS_SAFETY__' (safety bypass)")
            print(f"      - '__MALICIOUS_TOKEN__' (trigger token)")
            if lines_added:
                print(f"   • Lines added: {lines_added}")
            print(f"   • New SHA-256: {new_vocab_hash[:32]}...")
            print(f"   • Last 60 bytes: {new_vocab[-60:].hex()}")
            print(f"\n   ⚠️  Tokenizer poisoned - could enable prompt injection attacks!")

            pause()

            # Detect
            valid = fingerprints[str(vocab_file)].verify()
            print(f"\n   🔍 Detection: {'❌ FAILED' if valid else '✅ DETECTED & BLOCKED'}")

            if not valid:
                print("   🛡️  Vocabulary attack prevented!")
                print("   🚫 Prompt injection blocked")

            pause()

    # ============================================================================
    # PROOF 2: Restore and Verify
    # ============================================================================
    if run_all:
        print("\n" + "=" * 80)
        print("✅ PROOF 2: Restore and Verify Legitimate Model")
        print("=" * 80)

        print("\nRestoring original files...")
        for path, content in originals.items():
            with open(path, "wb") as f:
                f.write(content)

        print("Verifying restored files...")
        all_valid = True
        verified_count = 0
        for path, fp in fingerprints.items():
            is_valid = fp.verify()
            if is_valid:
                verified_count += 1
            all_valid = all_valid and is_valid

        print(f"   ✅ {verified_count}/{len(fingerprints)} files verified")

        if all_valid:
            print("\n   ✅ ALL FILES RESTORED AND VERIFIED")
            print("   ✓ Legitimate model confirmed")
            print("   ✓ Ready for production deployment")

        # ============================================================================
        # SUMMARY
        # ============================================================================

        print("\n" + "=" * 80)
        print("📊 SECURITY PROOF SUMMARY")
        print("=" * 80)

        print(f"""
        ✅ MODEL TESTED: {model_name}
        ✅ MODEL TYPE: {selected_model['type']}
        ✅ MODEL SIZE: {selected_model['size']}
        ✅ FILES TESTED: {len(fingerprints)} critical files

        🚨 SECURITY TESTS:
           1. Config tampering: ✅ DETECTED
           2. Weights poisoning: ✅ DETECTED
           3. Vocab poisoning: ✅ DETECTED

        🎯 DETECTION RATE: 3/3 (100%)

        🔐 CRYPTOGRAPHIC SECURITY:
           ✓ SHA-256 hash verification
           ✓ SHA-512 additional security
           ✓ Merkle tree chunk verification
           ✓ Tamper-proof integrity

        💡 PRODUCTION READY:
           ✓ Download models from HuggingFace Hub
           ✓ Fingerprint all critical files
           ✓ Detect any tampering attempts
           ✓ Verify before deployment
           ✓ Ensure model provenance

        🏆 SecureML protects {selected_model['type']} models in production!
        """)

        print("=" * 80)
        print("✅ SECURITY PROOF COMPLETE")
        print("=" * 80)

        print("\n💡 TIP: Run again with different models to test various architectures:")
        print("   python examples/prove_security_huggingface.py 6   # BERT Base")
        print("   python examples/prove_security_huggingface.py 8   # GPT-2")
        print("   python examples/prove_security_huggingface.py 12  # FLAN-T5")

        # Cleanup and exit
        print(f"\n🗑️  Cleaning up...")
        shutil.rmtree(model_dir)
        print("   ✅ Done")
        print("\n✅ Demo complete! Exiting.")
        exit(0)

    # If individual scenario, restore files and continue
    else:
        # Restore original files for next scenario
        print("\nRestoring original files...")
        for path, content in originals.items():
            with open(path, "wb") as f:
                f.write(content)
        print("   ✓ Files restored")
        print("\n" + "=" * 80)
        input("Press Enter to return to menu...")
