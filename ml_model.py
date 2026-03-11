"""
Delhi Outdoor Safety Prediction - Machine Learning Model
Step 3: Training and Evaluating the Prediction Model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

class DelhiSafetyPredictor:
    """
    Machine Learning model to predict outdoor safety
    Uses multiple algorithms and selects the best one
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = {}
    
    def prepare_data(self, df):
        """
        Prepare data for machine learning
        """
        print("\n[INFO] Preparing Data for Machine Learning...")
        print("=" * 60)
        
        # Select features (input variables)
        feature_columns = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'rainfall',
            'pm25', 'pm10', 'co', 'no2', 'so2', 'o3', 'aqi'
        ]
        
        # Check which features exist
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_names = available_features
        
        print(f"Features being used: {len(available_features)}")
        for feat in available_features:
            print(f"  - {feat}")
        
        # Get features (X) and target (y)
        X = df[available_features].copy()
        y = df['outdoor_safe'].copy()
        
        # Handle missing values
        print(f"\nMissing values before handling:")
        missing = X.isnull().sum()
        for col in missing[missing > 0].index:
            print(f"  {col}: {missing[col]} missing")
        
        # Fill missing values with median
        X = X.fillna(X.median())
        
        print(f"\n[SUCCESS] Data prepared:")
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Target distribution:")
        print(f"    Safe (1): {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"    Unsafe (0): {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets
        Scale features for better model performance
        """
        print(f"\n[INFO] Splitting Data...")
        
        # Split into train and test (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Testing set: {len(X_test)} samples")
        
        # Scale features (standardize)
        print(f"\n[INFO] Scaling Features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Train multiple ML models
        """
        print("\n[INFO] Training Machine Learning Models...")
        print("=" * 60)
        
        # Define models to try
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"  [SUCCESS] {name} trained successfully!")
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models and select the best one
        """
        print("\n[INFO] Evaluating Models...")
        print("=" * 60)
        
        best_accuracy = 0
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"  Accuracy: {accuracy*100:.2f}%")
            print(f"  AUC Score: {auc:.4f}")
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'y_pred': y_pred,
                'y_test': y_test,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n[SUCCESS] Best Model: {self.best_model_name}")
        print(f"   Accuracy: {best_accuracy*100:.2f}%")
        
        return self.results
    
    def print_detailed_results(self, y_test):
        """
        Print detailed classification report for best model
        """
        print(f"\n[INFO] Detailed Results for {self.best_model_name}")
        print("=" * 60)
        print(self.results[self.best_model_name]['classification_report'])
        
        # Confusion Matrix
        y_pred = self.results[self.best_model_name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        print("\nConfusion Matrix:")
        print(f"                 Predicted NO  Predicted YES")
        print(f"Actual NO        {cm[0][0]:>12}  {cm[0][1]:>13}")
        print(f"Actual YES       {cm[1][0]:>12}  {cm[1][1]:>13}")
    
    def plot_feature_importance(self):
        """
        Show which features are most important for prediction
        """
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            importances = self.best_model.feature_importances_
            
            # Create DataFrame
            feat_imp = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(f"\n[INFO] Feature Importance (Top 10):")
            print("=" * 60)
            for idx, row in feat_imp.head(10).iterrows():
                print(f"  {row['Feature']:<20} {row['Importance']:.4f}")
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(feat_imp['Feature'].head(10), feat_imp['Importance'].head(10))
            plt.xlabel('Importance')
            plt.title(f'Top 10 Feature Importance - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"\n[SUCCESS] Feature importance plot saved: feature_importance.png")
            plt.close()
    
    def save_model(self, filename='delhi_safety_model.pkl', metrics_filename='model_metrics.json'):
        """
        Save the trained model and its performance metrics for later use
        """
        import json
        
        # 1. Save Model
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name,
            'timestamp': datetime.now()
        }
        
        joblib.dump(model_package, filename)
        
        # 2. Save Metrics
        if self.best_model_name in self.results:
            best_results = self.results[self.best_model_name]
            
            # Recalculate metrics for export
            from sklearn.metrics import precision_score, recall_score, f1_score
            y_test = best_results['y_test'] # We need to store this in evaluate_models
            y_pred = best_results['y_pred']
            
            metrics_export = {
                'model_name': self.best_model_name,
                'accuracy': float(best_results['accuracy']),
                'auc': float(best_results['auc']),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(metrics_filename, 'w') as f:
                json.dump(metrics_export, f, indent=4)
                
            print(f"\n[SUCCESS] Model saved: {filename}")
            print(f"[SUCCESS] Metrics saved: {metrics_filename}")
        else:
            print(f"\n[SUCCESS] Model saved: {filename} (Metrics not saved)")
            
        print(f"   Model type: {self.best_model_name}")
        print(f"   Features: {len(self.feature_names)}")
    
    def predict_new_data(self, new_data):
        """
        Make prediction on new environmental data
        """
        # Prepare features
        X_new = new_data[self.feature_names].copy()
        X_new = X_new.fillna(X_new.median())
        
        # Scale
        X_new_scaled = self.scaler.transform(X_new)
        
        # Predict
        prediction = self.best_model.predict(X_new_scaled)
        probability = self.best_model.predict_proba(X_new_scaled)
        
        return prediction, probability


# ====================
# HOW TO USE THIS CODE
# ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Delhi Outdoor Safety - Machine Learning Model")
    print("=" * 60)
    
    # STEP 1: Load labeled data
    try:
        df = pd.read_csv('delhi_data_with_labels.csv')
        print(f"\n[SUCCESS] Data loaded: {len(df)} samples")
    except FileNotFoundError:
        print("\n[WARNING] Labeled data file not found!")
        print("Please run the label creation script first.")
        exit()
    
    # STEP 2: Create predictor
    predictor = DelhiSafetyPredictor()
    
    # STEP 3: Prepare data
    X, y = predictor.prepare_data(df)
    
    # STEP 4: Split and scale
    X_train, X_test, y_train, y_test = predictor.split_and_scale_data(X, y)
    
    # STEP 5: Train models
    models = predictor.train_models(X_train, y_train)
    
    # STEP 6: Evaluate models
    results = predictor.evaluate_models(X_test, y_test)
    
    # STEP 7: Show detailed results
    predictor.print_detailed_results(y_test)
    
    # STEP 8: Feature importance
    predictor.plot_feature_importance()
    
    # STEP 9: Save model
    predictor.save_model('delhi_safety_model.pkl')
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Review the feature importance plot")
    print("2. Check model accuracy and reports")
    print("3. Deploy model to Databricks (next phase)")