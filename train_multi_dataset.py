"""
BioVerify: Multi-Dataset Training Pipeline
Combines multiple Instagram datasets for improved fake account detection.

Datasets:
- train.csv: Original Instagram fake/real account dataset
- fakeAccountData.json: InstaFake dataset (fake accounts)
- realAccountData.json: InstaFake dataset (real accounts)
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using RandomForest")

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️ SMOTE not available, skipping class balancing")


def load_original_dataset(filepath='data/train.csv'):
    """Load the original Instagram dataset"""
    print(f"📂 Loading original dataset: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"   ❌ File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    # Standardize column names
    df.columns = [col.strip().lower().replace(' ', '_').replace('#', 'num_') for col in df.columns]
    
    print(f"   ✅ Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def load_instafake_json(fake_path='data/fakeAccountData.json', real_path='data/realAccountData.json'):
    """Load and combine InstaFake JSON datasets"""
    print(f"\n📂 Loading InstaFake JSON datasets...")
    
    combined_data = []
    
    # Load fake accounts
    if os.path.exists(fake_path):
        with open(fake_path, 'r') as f:
            fake_data = json.load(f)
        print(f"   ✅ Loaded {len(fake_data)} fake accounts")
        for item in fake_data:
            item['isFake'] = 1
        combined_data.extend(fake_data)
    else:
        print(f"   ❌ File not found: {fake_path}")
    
    # Load real accounts
    if os.path.exists(real_path):
        with open(real_path, 'r') as f:
            real_data = json.load(f)
        print(f"   ✅ Loaded {len(real_data)} real accounts")
        for item in real_data:
            item['isFake'] = 0
        combined_data.extend(real_data)
    else:
        print(f"   ❌ File not found: {real_path}")
    
    if not combined_data:
        return None
    
    df = pd.DataFrame(combined_data)
    print(f"   📊 Combined InstaFake dataset: {len(df)} rows")
    return df


def standardize_instafake_columns(df):
    """Map InstaFake columns to match original dataset format"""
    print("\n🔄 Standardizing InstaFake columns...")
    
    # Create mapping from InstaFake to original format
    standardized = pd.DataFrame()
    
    # Map columns
    standardized['profile_pic'] = df.get('userHasProfilPic', 1)
    standardized['nums/length_username'] = df['usernameDigitCount'] / df['usernameLength'].clip(lower=1)
    standardized['fullname_words'] = 1  # Default to 1 (not available in InstaFake)
    standardized['nums/length_fullname'] = 0  # Default (not available)
    standardized['name==username'] = 0  # Default (not available)
    standardized['description_length'] = df.get('userBiographyLength', 0)
    standardized['external_url'] = 0  # Default (not available)
    standardized['private'] = df.get('userIsPrivate', 0)
    standardized['num_posts'] = df.get('userMediaCount', 0)
    standardized['num_followers'] = df.get('userFollowerCount', 0)
    standardized['num_follows'] = df.get('userFollowingCount', 0)
    standardized['fake'] = df['isFake']
    
    print(f"   ✅ Standardized {len(standardized)} rows")
    return standardized


def combine_datasets(original_df, instafake_df):
    """Combine original and InstaFake datasets"""
    print("\n🔗 Combining datasets...")
    
    # Ensure same columns in same order
    if original_df is not None:
        # Standardize original column names
        original_df.columns = [col.strip().lower().replace(' ', '_').replace('#', 'num_') for col in original_df.columns]
        
        # Combine
        combined = pd.concat([original_df, instafake_df], ignore_index=True)
        print(f"   ✅ Combined dataset: {len(combined)} rows")
        print(f"   📊 Original: {len(original_df)} + InstaFake: {len(instafake_df)}")
    else:
        combined = instafake_df
        print(f"   ⚠️ Using only InstaFake dataset: {len(combined)} rows")
    
    # Show class distribution
    fake_count = combined['fake'].sum()
    real_count = len(combined) - fake_count
    print(f"   📊 Class distribution: Real={real_count}, Fake={fake_count}")
    
    return combined


def engineer_features(df):
    """Create advanced features for better detection"""
    print("\n🔧 Engineering features...")
    
    # Extract base features
    followers = df.get('num_followers', df.get('#followers', 0)).fillna(0)
    following = df.get('num_follows', df.get('#follows', 0)).fillna(0)
    posts = df.get('num_posts', df.get('#posts', 0)).fillna(0)
    profile_pic = df.get('profile_pic', 1).fillna(1)
    bio_length = df.get('description_length', 0).fillna(0)
    username_ratio = df.get('nums/length_username', 0).fillna(0)
    
    # Ratio features
    df['followers_following_ratio'] = followers / (following + 1)
    df['posts_per_follower'] = posts / (followers + 1)
    df['engagement_potential'] = posts * followers / (following + 1)
    
    # Log transforms for heavily skewed features
    df['log_followers'] = np.log1p(followers)
    df['log_following'] = np.log1p(following)
    df['log_posts'] = np.log1p(posts)
    
    # Suspicious score
    df['username_suspicion'] = username_ratio * (1 - profile_pic)
    
    # Profile completeness score
    df['profile_completeness'] = (
        profile_pic * 0.3 +
        (bio_length > 0).astype(int) * 0.3 +
        (posts > 5).astype(int) * 0.2 +
        (followers > 10).astype(int) * 0.2
    )
    
    # Combined suspicious indicators
    df['suspicious_score'] = (
        ((following > followers * 3).astype(int) * 0.3) +
        ((posts < 3).astype(int) * 0.2) +
        ((profile_pic == 0).astype(int) * 0.3) +
        ((username_ratio > 0.3).astype(int) * 0.2)
    )
    
    # Handle infinities and NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    print(f"   ✅ Created {len(df.columns)} total features")
    return df


def prepare_features(df):
    """Prepare feature matrix and target"""
    print("\n📋 Preparing features...")
    
    # Define target
    y = df['fake'].values
    
    # Select feature columns (exclude non-numeric and target)
    exclude_cols = ['fake', 'isfake', 'label', 'username', 'fullname']
    feature_cols = [col for col in df.columns if col.lower() not in exclude_cols]
    
    # Keep only numeric columns
    X = df[feature_cols].select_dtypes(include=[np.number])
    feature_names = list(X.columns)
    
    print(f"   ✅ Features: {len(feature_names)}")
    print(f"   📊 Sample shape: {X.shape}")
    
    return X.values, y, feature_names


def train_model(X, y, feature_names):
    """Train the model with cross-validation"""
    print("\n🎯 Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for class balancing
    if SMOTE_AVAILABLE:
        print("   ⚖️ Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"   📊 After SMOTE: {len(X_train_balanced)} samples")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    # Create model
    if XGBOOST_AVAILABLE:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        print("   🚀 Using XGBoost classifier")
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        print("   🌲 Using RandomForest classifier")
    
    # Cross-validation
    print("   🔄 Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='accuracy')
    print(f"   📊 CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    # Train final model
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n   📈 Test Set Results:")
    print(f"   ├── Accuracy:  {accuracy*100:.2f}%")
    print(f"   ├── Precision: {precision*100:.2f}%")
    print(f"   ├── Recall:    {recall*100:.2f}%")
    print(f"   ├── F1-Score:  {f1*100:.2f}%")
    print(f"   └── AUC-ROC:   {auc*100:.2f}%")
    
    print("\n   📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    return model, scaler, accuracy, auc, feature_names


def save_model(model, scaler, feature_names, accuracy, auc):
    """Save the trained model"""
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'accuracy': accuracy,
        'auc': auc
    }
    
    model_path = 'models/fake_account_model.pkl'
    joblib.dump(model_data, model_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"   ├── Accuracy: {accuracy*100:.2f}%")
    print(f"   ├── AUC: {auc*100:.2f}%")
    print(f"   └── Features: {len(feature_names)}")


def main():
    print("="*60)
    print("🔬 BioVerify: Multi-Dataset Training Pipeline")
    print("   Combining Instagram datasets for improved detection")
    print("="*60)
    
    # Step 1: Load original dataset
    original_df = load_original_dataset('data/train.csv')
    
    # Step 2: Load InstaFake datasets
    instafake_df = load_instafake_json('data/fakeAccountData.json', 'data/realAccountData.json')
    
    if instafake_df is not None:
        # Step 3: Standardize InstaFake columns
        instafake_standardized = standardize_instafake_columns(instafake_df)
        
        # Step 4: Combine datasets
        combined_df = combine_datasets(original_df, instafake_standardized)
    elif original_df is not None:
        combined_df = original_df
        print("\n⚠️ Using only original dataset")
    else:
        print("\n❌ No datasets available!")
        return
    
    # Step 5: Engineer features
    combined_df = engineer_features(combined_df)
    
    # Step 6: Prepare features
    X, y, feature_names = prepare_features(combined_df)
    
    # Step 7: Train model
    model, scaler, accuracy, auc, feature_names = train_model(X, y, feature_names)
    
    # Step 8: Save model
    save_model(model, scaler, feature_names, accuracy, auc)
    
    print("\n" + "="*60)
    print("✅ MULTI-DATASET TRAINING COMPLETE!")
    print("="*60)
    print(f"\n🎉 Model trained on {len(combined_df)} samples")
    print(f"🎯 Final Accuracy: {accuracy*100:.2f}%")
    print(f"📊 Final AUC: {auc*100:.2f}%")
    print("\n   Run: streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
