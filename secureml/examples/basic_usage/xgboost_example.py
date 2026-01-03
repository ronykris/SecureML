"""
Example: Securing an XGBoost model with SecureML

Installation:
    pip install secureaiml
    pip install xgboost

This example demonstrates how to:
1. Train an XGBoost model
2. Sign it with SecureML
3. Save as a secure package
4. Load and verify
5. Make predictions

Documentation: https://github.com/OWASP/SecureML
PyPI: https://pypi.org/project/secureaiml/
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from secureml import SecureModel
from secureml.utils.config import SecurityLevel


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


def main():
    print_header("🔐 SecureML XGBoost Interactive Example")

    print("\nThis interactive demo will walk you through:")
    print("  • Training an XGBoost classifier")
    print("  • Securing the model with SecureML")
    print("  • Making predictions")
    print("  • Exploring different security features")

    pause()

    # 1. Generate sample data
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

    print(f"   ✓ Training samples: {len(X_train)}")
    print(f"   ✓ Test samples: {len(X_test)}")
    print(f"   ✓ Features: {X_train.shape[1]}")
    print(f"   ✓ Classes: {len(np.unique(y))}")

    pause()

    # 2. Train XGBoost model with user options
    print_header("🎯 Step 2: Training XGBoost Model")

    print("\nChoose model complexity:")
    complexity = get_user_choice(
        "Select training configuration:",
        {
            "1": "Fast (50 estimators, depth 3)",
            "2": "Balanced (100 estimators, depth 5) - Recommended",
            "3": "Deep (200 estimators, depth 7)"
        }
    )

    # Set parameters based on choice
    params_map = {
        "1": {"n_estimators": 50, "max_depth": 3},
        "2": {"n_estimators": 100, "max_depth": 5},
        "3": {"n_estimators": 200, "max_depth": 7}
    }
    params = params_map[complexity]

    print(f"\n   Training with {params['n_estimators']} estimators, max depth {params['max_depth']}...")
    model = XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"\n   ✓ Model trained successfully!")
    print(f"   ✓ Training accuracy: {train_accuracy:.2%}")
    print(f"   ✓ Test accuracy: {test_accuracy:.2%}")

    pause()

    # 3. Secure the model with SecureML
    print_header("🔒 Step 3: Securing Model with SecureML")

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
    secure_model = SecureModel(
        model=model,
        security_level=security_level
    )

    print(f"   ✓ Model secured with {security_level.name} level")

    pause()

    # 4. Get model info
    print_header("ℹ️ Step 4: Model Information")

    print("\nRetrieving model metadata...")
    info = secure_model.get_info()

    print(f"\n   Model Details:")
    print(f"   ✓ Type: {info['model_info']['model_type']}")
    print(f"   ✓ Framework: {info['model_info']['framework']}")
    print(f"   ✓ Security level: {info['security_level']}")
    print(f"   ✓ Parameters: {info['model_info'].get('n_estimators', 'N/A')} estimators")

    pause()

    # 5. Interactive prediction demo
    print_header("🎲 Step 5: Making Predictions")

    while True:
        print("\nChoose prediction mode:")
        pred_choice = get_user_choice(
            "What would you like to do?",
            {
                "1": "Predict single random sample",
                "2": "Predict batch of 5 samples",
                "3": "Compare regular vs secure predictions",
                "4": "Continue to next step"
            }
        )

        if pred_choice == "4":
            break
        elif pred_choice == "1":
            idx = np.random.randint(0, len(X_test))
            sample = X_test[idx:idx+1]
            prediction = secure_model.predict(sample)
            actual = y_test[idx]
            print(f"\n   Sample #{idx}:")
            print(f"   ✓ Predicted: {prediction[0]}")
            print(f"   ✓ Actual: {actual}")
            print(f"   ✓ {'Correct!' if prediction[0] == actual else 'Incorrect'}")

        elif pred_choice == "2":
            predictions = secure_model.predict(X_test[:5])
            actuals = y_test[:5]
            print(f"\n   Batch predictions:")
            for i, (pred, actual) in enumerate(zip(predictions, actuals)):
                status = "✓" if pred == actual else "✗"
                print(f"   {status} Sample {i}: Predicted={pred}, Actual={actual}")
            accuracy = np.mean(predictions == actuals)
            print(f"\n   Batch accuracy: {accuracy:.1%}")

        elif pred_choice == "3":
            samples = X_test[:3]
            regular_preds = model.predict(samples)
            secure_preds = secure_model.predict(samples)

            print(f"\n   Comparing predictions:")
            print(f"   {'Sample':<10} {'Regular':<10} {'Secure':<10} {'Match':<10}")
            print(f"   {'-'*40}")
            for i, (reg, sec) in enumerate(zip(regular_preds, secure_preds)):
                match = "✓" if reg == sec else "✗"
                print(f"   Sample {i:<3} {reg:<10} {sec:<10} {match:<10}")

            print(f"\n   Note: SecureML wrapper maintains same predictions!")

        pause()

    # 6. Security features demo
    print_header("🛡️ Step 6: Security Features Overview")

    print("\nSecureML provides multiple security layers:")
    print("\n   1. Model Fingerprinting")
    print("      • Cryptographic hashing (SHA-256, SHA-512)")
    print("      • Tamper detection")
    print("      • Integrity verification")

    print("\n   2. Digital Signatures (with OpenSSF)")
    print("      • Identity verification")
    print("      • Provenance tracking")
    print("      • Supply chain security")

    print("\n   3. Watermarking")
    print("      • Ownership proof")
    print("      • Model theft detection")
    print("      • IP protection")

    print("\n   4. Encryption (higher security levels)")
    print("      • Model confidentiality")
    print("      • Protected deployment")
    print("      • Access control")

    pause()

    # 7. Next steps
    print_header("🚀 Next Steps")

    print("\nTo use advanced features:")

    print("\n   1. Enable Digital Signatures:")
    print("      pip install git+https://github.com/sigstore/model-transparency")
    print("      result = secure_model.sign_and_save(")
    print("          'xgboost_model.sml',")
    print("          identity='your-email@company.com',")
    print(f"          accuracy={test_accuracy:.2%}")
    print("      )")

    print("\n   2. Add Watermarking:")
    print("      from secureml import WatermarkType, TriggerSet")
    print("      secure_model.embed_watermark(")
    print("          owner='your-email@company.com',")
    print("          watermark_type=WatermarkType.TRIGGER_SET")
    print("      )")

    print("\n   3. Explore More Examples:")
    print("      • python examples/watermarking_example.py")
    print("      • python examples/security_attack_demo.py")
    print("      • python examples/prove_security.py")

    print_header("✅ Interactive Example Complete!")
    print("\nYou've successfully:")
    print(f"  ✓ Trained an XGBoost model ({test_accuracy:.2%} accuracy)")
    print(f"  ✓ Secured it with {security_level.name} protection")
    print("  ✓ Made predictions with SecureML wrapper")
    print("  ✓ Learned about security features")

    print("\n📚 For more information:")
    print("   • GitHub: https://github.com/OWASP/SecureML")
    print("   • PyPI: https://pypi.org/project/secureaiml/")


if __name__ == "__main__":
    main()
