"""
BioVerify: Multi-Modal Fake Account Detector
Training Script - Trains Random Forest on Instagram Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath='data/train.csv'):
    """
    Load and preprocess the Instagram fake account dataset
    """
    print("📂 Loading dataset...")
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n❌ ERROR: {filepath} not found!\n"
            f"Please download the dataset from:\n"
            f"https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts\n"
            f"And place 'train.csv' in the 'data/' folder."
        )
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display first few rows
    print("\n📊 Dataset Preview:")
    print(df.head())
    
    # Check for missing values
    print("\n🔍 Missing Values Check:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")
    
    # Handle missing values (fill with median for numeric, mode for categorical)
    if df.isnull().sum().sum() > 0:
        print("\n🔧 Handling missing values...")
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Class distribution
    print("\n📈 Class Distribution:")
    print(df['fake'].value_counts())
    print(f"Fake Percentage: {df['fake'].mean() * 100:.2f}%")
    
    return df

def prepare_features(df):
    """
    Prepare feature matrix and target vector
    """
    print("\n🎯 Preparing features...")
    
    # Define feature columns (excluding target)
    feature_columns = [
        'profile pic', 'nums/length username', 'fullname words',
        'nums/length fullname', 'name==username', 'description length',
        'external URL', 'private', '#posts', '#followers', '#follows'
    ]
    
    # Check if all features exist
    available_features = [col for col in feature_columns if col in df.columns]
    
    if len(available_features) < len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        print(f"⚠️ Warning: Missing features: {missing}")
    
    print(f"✅ Using {len(available_features)} features:")
    for feat in available_features:
        print(f"   • {feat}")
    
    # Prepare X and y
    X = df[available_features].values
    y = df['fake'].values
    
    return X, y, available_features

def train_model(X, y):
    """
    Train Random Forest Classifier with SMOTE for imbalanced data
    """
    print("\n🚀 Training Model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Apply SMOTE to handle class imbalance
    print("\n⚖️ Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Training samples: {X_train_balanced.shape[0]}")
    print(f"Fake accounts: {y_train_balanced.sum()}")
    print(f"Real accounts: {len(y_train_balanced) - y_train_balanced.sum()}")
    
    # Train Random Forest
    print("\n🌲 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    print("\n📊 Model Evaluation:")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✨ Accuracy: {accuracy * 100:.2f}%")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives (Real predicted as Real): {cm[0][0]}")
    print(f"False Positives (Real predicted as Fake): {cm[0][1]}")
    print(f"False Negatives (Fake predicted as Real): {cm[1][0]}")
    print(f"True Positives (Fake predicted as Fake): {cm[1][1]}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Top 5 Most Important Features:")
    print(feature_importance.head())
    
    return model, accuracy

def save_model(model, feature_names, accuracy):
    """
    Save trained model and metadata
    """
    print("\n💾 Saving model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and metadata
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'accuracy': accuracy
    }
    
    model_path = 'models/fake_account_model.pkl'
    joblib.dump(model_data, model_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"📈 Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    print("="*60)
    print("🔬 BioVerify: Fake Account Detector - Training Pipeline")
    print("="*60)
    
    try:
        # Step 1: Load and preprocess data
        df = load_and_preprocess_data('data/train.csv')
        
        # Step 2: Prepare features
        X, y, available_features = prepare_features(df)
        
        # Step 3: Train model
        model, accuracy = train_model(X, y)
        
        # Step 4: Save model
        save_model(model, available_features, accuracy)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print("\n🎉 You can now run the Streamlit app:")
        print("   streamlit run app.py")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease ensure:")
        print("1. You have downloaded the dataset from Kaggle")
        print("2. The 'train.csv' file is in the 'data/' folder")
        print("3. All required packages are installed: pip install -r requirements.txt")