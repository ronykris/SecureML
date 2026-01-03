"""
Example: Securing a PyTorch model with SecureML

Installation:
    pip install secureaiml
    pip install torch

This example demonstrates securing a simple PyTorch neural network.

Note: This is a demonstration example showing the workflow.
If PyTorch is not installed, it will show the workflow steps.
If PyTorch is available, it will run an interactive demo.

Documentation: https://github.com/OWASP/SecureML
PyPI: https://pypi.org/project/secureaiml/
"""

from secureml import SecureModel
from secureml.utils.config import SecurityLevel
import sys

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    # Create dummy nn module to avoid errors
    class DummyModule:
        pass
    nn = DummyModule()


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


if PYTORCH_AVAILABLE:
    class SimpleNN(nn.Module):
        """Simple neural network for demonstration"""
        def __init__(self, input_dim, hidden_dim=64, output_dim=2):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        def forward(self, x):
            return self.network(x)


def run_interactive_demo():
    """Run interactive PyTorch demo with actual training"""
    print_header("🔥 SecureML PyTorch Interactive Example")

    print("\nThis interactive demo will walk you through:")
    print("  • Building a PyTorch neural network")
    print("  • Training the model")
    print("  • Securing it with SecureML")
    print("  • Making predictions")

    pause()

    # 1. Generate data
    print_header("📊 Step 1: Generating Training Data")

    print("\nGenerating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    print(f"   ✓ Training samples: {len(X_train)}")
    print(f"   ✓ Test samples: {len(X_test)}")
    print(f"   ✓ Features: {X_train.shape[1]}")
    print(f"   ✓ Classes: {len(np.unique(y))}")

    pause()

    # 2. Build model
    print_header("🏗️ Step 2: Building Neural Network")

    print("\nChoose network architecture:")
    arch_choice = get_user_choice(
        "Select network complexity:",
        {
            "1": "Small (32 hidden units)",
            "2": "Medium (64 hidden units) - Recommended",
            "3": "Large (128 hidden units)"
        }
    )

    hidden_map = {"1": 32, "2": 64, "3": 128}
    hidden_dim = hidden_map[arch_choice]

    print(f"\n   Building network with {hidden_dim} hidden units...")
    model = SimpleNN(input_dim=20, hidden_dim=hidden_dim, output_dim=2)

    print(f"\n   Model architecture:")
    print(f"   ✓ Input layer: 20 features")
    print(f"   ✓ Hidden layer 1: {hidden_dim} units + ReLU + Dropout(0.2)")
    print(f"   ✓ Hidden layer 2: {hidden_dim//2} units + ReLU")
    print(f"   ✓ Output layer: 2 classes")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")

    pause()

    # 3. Train model
    print_header("🎯 Step 3: Training Neural Network")

    print("\nChoose training configuration:")
    train_choice = get_user_choice(
        "Select training duration:",
        {
            "1": "Quick (10 epochs)",
            "2": "Standard (50 epochs) - Recommended",
            "3": "Extended (100 epochs)"
        }
    )

    epochs_map = {"1": 10, "2": 50, "3": 100}
    num_epochs = epochs_map[train_choice]

    print(f"\n   Training for {num_epochs} epochs...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(10, num_epochs // 5) == 0:
            print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_preds = torch.argmax(train_outputs, dim=1)
        train_accuracy = (train_preds == y_train_tensor).float().mean().item()

        test_outputs = model(X_test_tensor)
        test_preds = torch.argmax(test_outputs, dim=1)
        test_accuracy = (test_preds == y_test_tensor).float().mean().item()

    print(f"\n   ✓ Training complete!")
    print(f"   ✓ Training accuracy: {train_accuracy:.2%}")
    print(f"   ✓ Test accuracy: {test_accuracy:.2%}")

    pause()

    # 4. Secure the model
    print_header("🔒 Step 4: Securing Model with SecureML")

    print("\nChoose security level:")
    security_choice = get_user_choice(
        "Select security configuration:",
        {
            "1": "BASIC - Quick verification",
            "2": "STANDARD - Balanced security (Recommended)",
            "3": "ENTERPRISE - High security",
            "4": "MAXIMUM - Maximum protection"
        }
    )

    security_map = {
        "1": SecurityLevel.BASIC,
        "2": SecurityLevel.STANDARD,
        "3": SecurityLevel.ENTERPRISE,
        "4": SecurityLevel.MAXIMUM
    }
    security_level = security_map[security_choice]

    print(f"\n   Creating SecureModel with {security_level.name} security...")

    # For PyTorch, we can secure the state dict
    secure_model = SecureModel(
        model=model.state_dict(),
        security_level=security_level
    )

    print(f"   ✓ Model secured with {security_level.name} level")

    pause()

    # 5. Make predictions
    print_header("🎲 Step 5: Making Predictions")

    while True:
        print("\nChoose prediction mode:")
        pred_choice = get_user_choice(
            "What would you like to do?",
            {
                "1": "Predict single random sample",
                "2": "Predict batch of 5 samples",
                "3": "Show prediction probabilities",
                "4": "Continue to next step"
            }
        )

        if pred_choice == "4":
            break
        elif pred_choice == "1":
            idx = np.random.randint(0, len(X_test))
            sample = X_test_tensor[idx:idx+1]

            with torch.no_grad():
                output = model(sample)
                prediction = torch.argmax(output, dim=1).item()
            actual = y_test[idx]

            print(f"\n   Sample #{idx}:")
            print(f"   ✓ Predicted: {prediction}")
            print(f"   ✓ Actual: {actual}")
            print(f"   ✓ {'Correct!' if prediction == actual else 'Incorrect'}")

        elif pred_choice == "2":
            with torch.no_grad():
                outputs = model(X_test_tensor[:5])
                predictions = torch.argmax(outputs, dim=1).numpy()
            actuals = y_test[:5]

            print(f"\n   Batch predictions:")
            for i, (pred, actual) in enumerate(zip(predictions, actuals)):
                status = "✓" if pred == actual else "✗"
                print(f"   {status} Sample {i}: Predicted={pred}, Actual={actual}")

            accuracy = np.mean(predictions == actuals)
            print(f"\n   Batch accuracy: {accuracy:.1%}")

        elif pred_choice == "3":
            with torch.no_grad():
                outputs = model(X_test_tensor[:3])
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1).numpy()

            print(f"\n   Prediction probabilities:")
            print(f"   {'Sample':<10} {'Class 0':<12} {'Class 1':<12} {'Predicted':<12}")
            print(f"   {'-'*50}")
            for i, (prob, pred) in enumerate(zip(probs, predictions)):
                print(f"   Sample {i:<3} {prob[0].item():<11.2%} {prob[1].item():<11.2%} {pred:<12}")

        pause()

    # 6. Next steps
    print_header("🚀 Next Steps")

    print("\nYour PyTorch model is now secured!")
    print("\n   To save and sign the model:")
    print("   1. Save the state dict:")
    print("      torch.save(model.state_dict(), 'pytorch_model.pth')")

    print("\n   2. Create fingerprint:")
    print("      from secureml.core.fingerprint import ModelFingerprint")
    print("      fp = ModelFingerprint.create('pytorch_model.pth')")
    print("      fp.to_json('pytorch_model_fingerprint.json')")

    print("\n   3. For signing (requires OpenSSF):")
    print("      pip install git+https://github.com/sigstore/model-transparency")

    print("\n   4. Explore more examples:")
    print("      • python examples/watermarking_example.py")
    print("      • python examples/security_attack_demo.py")

    print_header("✅ Interactive PyTorch Example Complete!")
    print("\nYou've successfully:")
    print(f"  ✓ Built a neural network with {total_params:,} parameters")
    print(f"  ✓ Trained it to {test_accuracy:.2%} accuracy")
    print(f"  ✓ Secured it with {security_level.name} protection")
    print("  ✓ Made predictions with the secured model")


def show_workflow():
    """Show workflow when PyTorch is not available"""
    print_header("🔥 SecureML PyTorch Workflow Guide")

    print("\n⚠️  PyTorch not detected. Install it to run the interactive demo:")
    print("   pip install torch")

    print("\nHere's the workflow for securing PyTorch models:\n")

    pause()

    print_header("Step 1: Define Your Model")
    print("""
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)

# Create and train your model
model = SimpleNN(input_dim=20)
# ... training code ...
    """)

    pause()

    print_header("Step 2: Save the Model")
    print("""
# Save state dict (recommended)
torch.save(model.state_dict(), "model.pth")

# Or save entire model
torch.save(model, "model_full.pth")
    """)

    pause()

    print_header("Step 3: Create Security Fingerprint")
    print("""
from secureml.core.fingerprint import ModelFingerprint

# Create fingerprint
fp = ModelFingerprint.create(
    "model.pth",
    algorithms=["sha256", "sha512"],
    enable_merkle=True
)

# Save fingerprint
fp.to_json("model_fingerprint.json")

print("Model fingerprinted!")
    """)

    pause()

    print_header("Step 4: Verify Before Loading")
    print("""
from secureml.core.fingerprint import ModelFingerprint

# Load saved fingerprint
fp = ModelFingerprint.from_json("model_fingerprint.json")

# Verify integrity
if fp.verify():
    print("✓ Model verified - safe to load")
    model = SimpleNN(input_dim=20)
    model.load_state_dict(torch.load("model.pth"))
else:
    print("✗ Tampering detected - do not use!")
    """)

    pause()

    print_header("Step 5: Optional - Digital Signatures")
    print("""
# Install OpenSSF model signing (optional)
# pip install git+https://github.com/sigstore/model-transparency

from secureml import SecureModel

# Sign the model
secure_model = SecureModel.load_from_path("model.pth")
secure_model.sign_and_save(
    "pytorch_model.sml",
    identity="dl-team@company.com",
    version="1.0.0",
    architecture="SimpleNN",
    framework="pytorch"
)

# Load and verify signature
verified = SecureModel.load("pytorch_model.sml", verify=True)
    """)

    pause()

    print_header("✅ Workflow Guide Complete!")
    print("\nKey Takeaways:")
    print("  ✓ Always fingerprint models before deployment")
    print("  ✓ Verify integrity before loading")
    print("  ✓ Use digital signatures for provenance tracking")
    print("  ✓ Combine multiple security layers")

    print("\n📚 For more information:")
    print("   • GitHub: https://github.com/OWASP/SecureML")
    print("   • PyPI: https://pypi.org/project/secureaiml/")


def main():
    if PYTORCH_AVAILABLE:
        run_interactive_demo()
    else:
        show_workflow()


if __name__ == "__main__":
    main()
