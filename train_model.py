"""
BioVerify: Multi-Modal Fake Account Detector
Training Script - Enhanced with XGBoost & Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️ imbalanced-learn not available. Training without SMOTE.")

# Try to import XGBoost, fall back to RandomForest if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Using RandomForest. Install with: pip install xgboost")


def load_and_preprocess_data(filepath='data/train.csv'):
    """Load and preprocess the Instagram fake account dataset"""
    print("📂 Loading dataset...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n❌ ERROR: {filepath} not found!\n"
            f"Please download the dataset from:\n"
            f"https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts\n"
            f"And place 'train.csv' in the 'data/' folder."
        )
    
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n📊 Dataset Preview:")
    print(df.head())
    
    print("\n🔍 Missing Values Check:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")
    
    if df.isnull().sum().sum() > 0:
        print("\n🔧 Handling missing values...")
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("\n📈 Class Distribution:")
    print(df['fake'].value_counts())
    print(f"Fake Percentage: {df['fake'].mean() * 100:.2f}%")
    
    return df


def engineer_features(df):
    """Create advanced features for better detection"""
    print("\n🔧 Engineering advanced features...")
    
    df = df.copy()
    
    # Ratio features
    df['followers_following_ratio'] = df['#followers'] / (df['#follows'] + 1)
    df['posts_per_follower'] = df['#posts'] / (df['#followers'] + 1)
    df['engagement_potential'] = df['#posts'] * df['#followers'] / (df['#follows'] + 1)
    
    # Log transforms for skewed distributions
    df['log_followers'] = np.log1p(df['#followers'])
    df['log_following'] = np.log1p(df['#follows'])
    df['log_posts'] = np.log1p(df['#posts'])
    
    # Username analysis features
    df['username_suspicion'] = df['nums/length username'] * (1 - df['profile pic'])
    
    # Account completeness score
    df['profile_completeness'] = (
        df['profile pic'] * 0.3 +
        (df['description length'] > 0).astype(int) * 0.3 +
        (df['#posts'] > 5).astype(int) * 0.2 +
        (df['#followers'] > 10).astype(int) * 0.2
    )
    
    # Suspicious pattern score
    df['suspicious_score'] = (
        (df['#follows'] > df['#followers'] * 3).astype(int) * 0.3 +
        (df['#posts'] < 3).astype(int) * 0.2 +
        (df['profile pic'] == 0).astype(int) * 0.3 +
        (df['nums/length username'] > 0.3).astype(int) * 0.2
    )
    
    print("✅ Created 10 new engineered features!")
    
    return df


def prepare_features(df):
    """Prepare feature matrix and target vector with engineered features"""
    print("\n🎯 Preparing features...")
    
    # Original features
    base_features = [
        'profile pic', 'nums/length username', 'fullname words',
        'nums/length fullname', 'name==username', 'description length',
        'external URL', 'private', '#posts', '#followers', '#follows'
    ]
    
    # Engineered features
    engineered_features = [
        'followers_following_ratio', 'posts_per_follower', 'engagement_potential',
        'log_followers', 'log_following', 'log_posts',
        'username_suspicion', 'profile_completeness', 'suspicious_score'
    ]
    
    all_features = base_features + engineered_features
    available_features = [col for col in all_features if col in df.columns]
    
    print(f"✅ Using {len(available_features)} features:")
    print(f"   • Base features: {len([f for f in base_features if f in available_features])}")
    print(f"   • Engineered features: {len([f for f in engineered_features if f in available_features])}")
    
    X = df[available_features].values
    y = df['fake'].values
    
    # Handle infinities and NaN
    X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
    
    return X, y, available_features


def train_model_with_cv(X, y, feature_names):
    """Train XGBoost with cross-validation and hyperparameter tuning"""
    print("\n🚀 Training Enhanced Model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE if available
    if SMOTE_AVAILABLE:
        print("\n⚖️ Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE - Training samples: {X_train_balanced.shape[0]}")
    else:
        print("\n⚠️ Training without SMOTE (class weights will be used)")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    if XGBOOST_AVAILABLE:
        print("\n🚀 Training XGBoost with GridSearchCV...")
        
        # XGBoost with hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 7, 10],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        base_model = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # Use stratified k-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Grid search (use smaller grid for speed)
        grid_search = GridSearchCV(
            base_model,
            {'n_estimators': [100, 150], 'max_depth': [5, 7], 'learning_rate': [0.1]},
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_balanced, y_train_balanced)
        model = grid_search.best_estimator_
        
        print(f"\n🏆 Best Parameters: {grid_search.best_params_}")
        print(f"🏆 Best CV Score: {grid_search.best_score_ * 100:.2f}%")
    else:
        print("\n🌲 Training Random Forest Classifier...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    print("\n📊 Model Evaluation:")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n✨ Accuracy: {accuracy * 100:.2f}%")
    print(f"📈 AUC-ROC Score: {auc_score * 100:.2f}%")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives (Real predicted as Real): {cm[0][0]}")
    print(f"False Positives (Real predicted as Fake): {cm[0][1]}")
    print(f"False Negatives (Fake predicted as Real): {cm[1][0]}")
    print(f"True Positives (Fake predicted as Fake): {cm[1][1]}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = np.zeros(len(feature_names))
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, scaler, accuracy, auc_score


def save_model(model, scaler, feature_names, accuracy, auc_score):
    """Save trained model, scaler, and metadata"""
    print("\n💾 Saving model...")
    
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'model_type': 'XGBoost' if XGBOOST_AVAILABLE else 'RandomForest'
    }
    
    model_path = 'models/fake_account_model.pkl'
    joblib.dump(model_data, model_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"📈 Model Accuracy: {accuracy * 100:.2f}%")
    print(f"📈 AUC-ROC Score: {auc_score * 100:.2f}%")


if __name__ == "__main__":
    print("="*60)
    print("🔬 BioVerify: Enhanced Training Pipeline")
    print("   Using XGBoost + Feature Engineering")
    print("="*60)
    
    try:
        # Step 1: Load data
        df = load_and_preprocess_data('data/train.csv')
        
        # Step 2: Engineer features
        df = engineer_features(df)
        
        # Step 3: Prepare features
        X, y, feature_names = prepare_features(df)
        
        # Step 4: Train with CV
        model, scaler, accuracy, auc_score = train_model_with_cv(X, y, feature_names)
        
        # Step 5: Save model
        save_model(model, scaler, feature_names, accuracy, auc_score)
        
        print("\n" + "="*60)
        print("✅ ENHANCED TRAINING COMPLETE!")
        print("="*60)
        print(f"\n🎉 Model Type: {'XGBoost' if XGBOOST_AVAILABLE else 'RandomForest'}")
        print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
        print(f"📊 AUC Score: {auc_score * 100:.2f}%")
        print("\n   streamlit run app.py")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()