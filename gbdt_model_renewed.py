import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class PXRFMaterialClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42):
        """
        Initialize the PXRF Material Classifier with Gradient Boosted Decision Tree
        
        Args:
            n_estimators: Number of boosting stages to perform
            learning_rate: Learning rate shrinks the contribution of each tree
            max_depth: Maximum depth of the individual regression estimators
            random_state: Random state for reproducibility
        """
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.element_columns = []
        self.is_trained = False
        
    def load_and_prepare_data(self, file_path):
        """
        Load CSV data and prepare it for training
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            X: Feature matrix (element concentrations)
            y: Target vector (material types)
        """
        print(f"Loading data from {file_path}...")
        
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} samples")
        
        # Define element columns (all the chemical elements measured by pXRF)
        self.element_columns = ['Ag', 'Al', 'As', 'Au', 'Ca', 'Cr', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'Ni', 'P', 'Pb', 'Sr', 'Ti', 'V', 'Zn']
        
        # Filter to only include element columns that exist in the data
        available_elements = [col for col in self.element_columns if col in df.columns]
        self.element_columns = available_elements
        print(f"Using {len(self.element_columns)} element features: {self.element_columns}")
        
        # Extract features (X) - element concentrations
        X = df[self.element_columns].copy()
        
        # Handle missing values by filling with 0 (below detection limit)
        X = X.fillna(0)
        
        # Handle negative values (set to 0)
        X[X < 0] = 0
        
        # Extract target variable (y) - material types
        y = df['material'].copy()
        
        # Handle blank and unknown materials - treat as soil per instructions
        y = y.fillna('soil')  # Handle any NaN values
        y = y.replace(['blank', 'unknown', ''], 'soil')  # Replace blank/unknown with soil
        
        print(f"\nMaterial distribution:")
        print(y.value_counts())
        
        return X, y
    
    def train_test_by_materials(self, X, y, test_size=0.2, random_state=42):
        """
        Split data ensuring all material types are represented in both train and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test: Split datasets
        """
        print("\nSplitting data with stratification to ensure all material types in train/test...")
        
        # Use stratified split to ensure all material types are in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y  # This ensures proportional representation of each material type
        )
        
        print(f"Training set size: {len(X_train)} samples")
        print(f"Test set size: {len(X_test)} samples")
        
        print(f"\nTraining set material distribution:")
        print(y_train.value_counts())
        
        print(f"\nTest set material distribution:")
        print(y_test.value_counts())
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the gradient boosted decision tree model
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
        """
        print("\nTraining Gradient Boosted Decision Tree model...")
        
        # Encode categorical labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train_encoded)
        
        self.is_trained = True
        print("Model training completed!")
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'element': self.element_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Elements:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def evaluate_model_multiclass(self, X_test, y_test):
        """
        Evaluate the model for multi-class material type classification
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            predictions, accuracy, probabilities: Model predictions, accuracy, and class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("\n" + "="*70)
        print("EVALUATING MODEL: MULTI-CLASS MATERIAL TYPE CLASSIFICATION")
        print("="*70)
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions and get probabilities
        y_pred_encoded = self.model.predict(X_test_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nMULTI-CLASS CLASSIFICATION ACCURACY: {accuracy:.4f}")
        print(f"Percentage Correct: {accuracy*100:.2f}%")
        
        # Detailed classification report
        print("\nMULTI-CLASS CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion matrix
        unique_labels = sorted(set(y_test.tolist() + y_pred.tolist()))
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        
        print("\nCONFUSION MATRIX:")
        print(f"{'Actual \\ Predicted':<15}", end="")
        for label in unique_labels:
            print(f"{label:<10}", end="")
        print()
        
        for i, true_label in enumerate(unique_labels):
            print(f"{true_label:<15}", end="")
            for j, pred_label in enumerate(unique_labels):
                print(f"{cm[i][j]:<10}", end="")
            print()
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=unique_labels, average=None, zero_division=0
        )
        
        print("\nPER-CLASS DETAILED METRICS:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
        print("-" * 63)
        for i, label in enumerate(unique_labels):
            print(f"{label:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12}")
        
        # Average metrics
        avg_precision = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        print("\nWEIGHTED AVERAGE METRICS:")
        print(f"Precision: {avg_precision[0]:.4f}")
        print(f"Recall: {avg_precision[1]:.4f}")
        print(f"F1-Score: {avg_precision[2]:.4f}")
        
        # Class probability statistics
        print("\nCLASS PROBABILITY STATISTICS:")
        class_names = self.label_encoder.classes_
        for i, class_name in enumerate(class_names):
            class_probs = y_pred_proba[:, i]
            print(f"{class_name}: Mean={class_probs.mean():.3f}, Max={class_probs.max():.3f}, Min={class_probs.min():.3f}")
        
        return y_pred, accuracy, y_pred_proba
    
    def evaluate_model_binary(self, X_test, y_test):
        """
        Evaluate the model for binary soil vs non-soil classification
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            binary_predictions, binary_accuracy, binary_probabilities: Binary classification results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("\n" + "="*70)
        print("EVALUATING MODEL: BINARY SOIL VS NON-SOIL CLASSIFICATION")
        print("="*70)
        
        # Convert multi-class labels to binary (soil vs non-soil)
        y_binary = (y_test == 'soil').astype(str)
        y_binary = y_binary.replace({'True': 'soil', 'False': 'non-soil'})
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get multi-class probabilities
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Extract soil probabilities
        soil_class_idx = np.where(self.label_encoder.classes_ == 'soil')[0]
        if len(soil_class_idx) > 0:
            soil_probs = y_pred_proba[:, soil_class_idx[0]]
        else:
            # If no soil class, assume all non-soil
            soil_probs = np.zeros(len(X_test))
        
        non_soil_probs = 1.0 - soil_probs
        
        # Make binary predictions (soil if probability > 0.5)
        y_binary_pred = np.where(soil_probs > 0.5, 'soil', 'non-soil')
        
        # Calculate binary accuracy
        binary_accuracy = accuracy_score(y_binary, y_binary_pred)
        
        print(f"\nBINARY CLASSIFICATION ACCURACY: {binary_accuracy:.4f}")
        print(f"Percentage Correct: {binary_accuracy*100:.2f}%")
        
        # Detailed classification report for binary classification
        print("\nBINARY CLASSIFICATION REPORT:")
        print(classification_report(y_binary, y_binary_pred, zero_division=0))
        
        # Binary confusion matrix
        binary_cm = confusion_matrix(y_binary, y_binary_pred, labels=['soil', 'non-soil'])
        
        print("\nBINARY CONFUSION MATRIX:")
        print(f"{'Actual \\ Predicted':<20} {'soil':<10} {'non-soil':<10}")
        print(f"{'soil':<20} {binary_cm[0][0]:<10} {binary_cm[0][1]:<10}")
        print(f"{'non-soil':<20} {binary_cm[1][0]:<10} {binary_cm[1][1]:<10}")
        
        # Calculate additional metrics
        true_positives = binary_cm[0][0]  # soil predicted as soil
        false_positives = binary_cm[1][0]  # non-soil predicted as soil
        false_negatives = binary_cm[0][1]  # soil predicted as non-soil
        true_negatives = binary_cm[1][1]  # non-soil predicted as non-soil
        
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        print(f"\nBINARY CLASSIFICATION DETAILED METRICS:")
        print(f"Sensitivity (Recall for Soil): {sensitivity:.4f}")
        print(f"Specificity (Recall for Non-Soil): {specificity:.4f}")
        print(f"Precision for Soil: {precision:.4f}")
        
        # Probability distribution statistics
        print(f"\nPROBABILITY STATISTICS:")
        print(f"Soil Probability - Mean: {soil_probs.mean():.3f}, Std: {soil_probs.std():.3f}")
        print(f"Non-Soil Probability - Mean: {non_soil_probs.mean():.3f}, Std: {non_soil_probs.std():.3f}")
        
        binary_probabilities = {
            'soil': soil_probs,
            'non-soil': non_soil_probs
        }
        
        return y_binary_pred, binary_accuracy, binary_probabilities
    
    def cross_validate(self, X, y, cv_folds=5):
        """
        Perform cross-validation to assess model stability
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            cv_scores: Cross-validation scores
        """
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        # Encode labels and scale features
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        # Use stratified k-fold to ensure balanced folds
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=skf, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def export_model_compact(self, output_path='pxrf_model_compact.json'):
        """
        Export a compact JSON representation of the model for Flutter mobile app
        Only includes essential decision parameters for predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before exporting")
        
        print(f"\nExporting compact model for Flutter app...")
        
        def tree_to_compact_dict(tree):
            """Convert a single tree to compact dictionary format"""
            tree_ = tree.tree_
            
            # Extract all node information in compact arrays
            nodes = []
            for i in range(tree_.node_count):
                if tree_.children_left[i] != -1:  # Decision node
                    node = {
                        'f': int(tree_.feature[i]),  # feature index
                        't': float(tree_.threshold[i]),  # threshold
                        'l': int(tree_.children_left[i]),  # left child index
                        'r': int(tree_.children_right[i])  # right child index
                    }
                else:  # Leaf node
                    # For multi-class, we need the value for each class
                    node = {
                        'v': tree_.value[i][0].tolist()  # leaf values
                    }
                nodes.append(node)
            
            return nodes
        
        # Build compact model structure
        model_dict = {
            "metadata": {
                "model_type": "GradientBoostingClassifier",
                "n_estimators": int(self.model.n_estimators),
                "n_classes": int(len(self.label_encoder.classes_)),
                "n_features": int(len(self.element_columns)),
                "learning_rate": float(self.model.learning_rate),
                "classes": self.label_encoder.classes_.tolist(),
                "feature_names": self.element_columns,
                "scaler_mean": self.scaler.mean_.tolist(),
                "scaler_scale": self.scaler.scale_.tolist(),
                "supports_binary_classification": True,
                "soil_class_index": int(np.where(self.label_encoder.classes_ == 'soil')[0][0]) if 'soil' in self.label_encoder.classes_ else None,
                "classification_types": {
                    "binary": "soil vs non-soil",
                    "multiclass": "specific material types"
                }
            },
            "trees": []
        }
        
        # Export each tree in compact format
        print(f"Exporting {len(self.model.estimators_)} trees...")
        for i, estimator_array in enumerate(self.model.estimators_):
            trees_for_estimator = []
            
            # For multiclass, each estimator has one tree per class
            for class_idx, tree_estimator in enumerate(estimator_array):
                compact_tree = tree_to_compact_dict(tree_estimator)
                trees_for_estimator.append({
                    "c": int(class_idx),  # class index
                    "n": compact_tree  # nodes
                })
            
            model_dict["trees"].append({
                "e": int(i),  # estimator index
                "t": trees_for_estimator  # trees
            })
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(model_dict, f, separators=(',', ':'))  # Compact JSON
        
        file_size_kb = Path(output_path).stat().st_size / 1024
        print(f"Compact model exported to {output_path}")
        print(f"File size: {file_size_kb:.2f} KB")
        print(f"Total trees: {len(self.model.estimators_)}")
    
        return model_dict

    def predict_single_sample(self, element_concentrations):
        """
        Predict a single sample with both soil/non-soil and detailed material type probabilities
        
        Args:
            element_concentrations: Dictionary of element concentrations
                                e.g., {'Fe': 5.2, 'Cu': 0.1, ...}
        
        Returns:
            Dictionary with predicted class, probabilities for soil/non-soil classification,
            probabilities for all material types, and confidence measures
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Create feature vector in correct order
        features = []
        for element in self.element_columns:
            features.append(element_concentrations.get(element, 0.0))
        
        # Scale features
        features_scaled = (np.array(features) - self.scaler.mean_) / self.scaler.scale_
        
        # Make prediction
        prediction_encoded = self.model.predict([features_scaled])[0]
        probabilities = self.model.predict_proba([features_scaled])[0]
        
        predicted_class = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Create probability dictionary for all material types
        prob_dict = dict(zip(self.label_encoder.classes_, probabilities))
        
        # Calculate binary soil/non-soil probabilities
        soil_prob = prob_dict.get('soil', 0.0)
        non_soil_prob = 1.0 - soil_prob
        
        # Get confidence (highest probability)
        confidence = max(probabilities)
        
        # Determine binary classification
        is_soil = predicted_class == 'soil'
        binary_prediction = 'soil' if is_soil else 'non-soil'
        binary_confidence = soil_prob if is_soil else non_soil_prob
        
        return {
            # Multi-class classification results
            'predicted_class': predicted_class,
            'material_probabilities': prob_dict,
            'confidence': confidence,
            'top_predictions': sorted(prob_dict.items(), key=lambda x: x[1], reverse=True),
            
            # Binary soil/non-soil classification results
            'binary_prediction': binary_prediction,
            'binary_probabilities': {
                'soil': soil_prob,
                'non-soil': non_soil_prob
            },
            'binary_confidence': binary_confidence,
            'is_soil': is_soil,
            
            # Combined interpretation
            'interpretation': {
                'primary_classification': f"{binary_prediction} ({binary_confidence:.1%} confidence)",
                'specific_material': f"{predicted_class} ({confidence:.1%} confidence)" if not is_soil else None,
                'archaeological_significance': 'High' if not is_soil and confidence > 0.7 else 'Medium' if not is_soil else 'Low'
            }
        }
    
    def export_test_data_for_flutter(self, X_test, y_test=None, y_pred=None, output_path='test_data_for_flutter.csv'):
        """
        Export test data in CSV format compatible with Flutter app
        Only includes coordinates and element data - no true labels
        """
        print(f"\nExporting test data for Flutter app...")
        
        # Create a copy of the test data
        test_df = X_test.copy()
        
        # Add coordinates (using index as dummy coordinates for testing)
        test_df['x_coord'] = range(len(test_df))
        test_df['y_coord'] = range(len(test_df))
        
        # Reorder columns to have coordinates first
        column_order = ['x_coord', 'y_coord'] + self.element_columns
        
        test_df = test_df[column_order]
        
        # Save to CSV (without the statistics)
        test_df.to_csv(output_path, index=False)
        
        # Calculate and print statistics (but don't save to CSV)
        total_samples = len(test_df)
        
        if y_pred is not None:
            # Count material type predictions
            from collections import Counter
            pred_counts = Counter(y_pred)
            
            print(f"Test data exported to {output_path}")
            print(f"Total samples: {total_samples}")
            print("\nPrediction distribution:")
            for material, count in pred_counts.most_common():
                print(f"  - {material}: {count} ({count/total_samples*100:.1f}%)")
            
            # If we have true labels, show accuracy comparison
            if y_test is not None:
                correct_predictions = sum(1 for true, pred in zip(y_test, y_pred) if true == pred)
                accuracy = correct_predictions / total_samples
                print(f"\nModel accuracy on test set: {accuracy*100:.1f}%")
        else:
            print(f"Test data exported to {output_path}")
            print(f"Total samples: {total_samples}")
            print("Note: No predictions available for statistics")
        
        print(f"\nColumns: {', '.join(test_df.columns.tolist())}")
        
        return test_df
    
    def export_test_data_with_predictions_json(self, X_test, y_test=None, output_path='test_data_with_predictions.json'):
        """
        Export test data with predictions and probabilities in JSON format
        Includes both binary (soil/non-soil) and multiclass probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before exporting test data with predictions")
        
        print(f"\nExporting test data with predictions to JSON...")
        
        # Prepare the data structure
        test_samples = []
        
        # Get predictions and probabilities for all test samples
        X_scaled = self.scaler.transform(X_test)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        # Convert predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        for i in range(len(X_test)):
            # Get element concentrations for this sample
            sample_elements = dict(zip(self.element_columns, X_test.iloc[i].values.tolist()))
            
            # Get multiclass probabilities
            multiclass_probs = dict(zip(self.label_encoder.classes_, y_pred_proba[i].tolist()))
            
            # Calculate binary soil/non-soil probabilities
            soil_prob = float(multiclass_probs.get('soil', 0.0))
            non_soil_prob = 1.0 - soil_prob
            
            # Determine binary prediction
            binary_prediction = 'soil' if y_pred[i] == 'soil' else 'non-soil'
            binary_confidence = soil_prob if binary_prediction == 'soil' else non_soil_prob
            
            # Create sample data
            sample_data = {
                'sample_id': int(i),
                'coordinates': {
                    'x': float(i),  # Using index as dummy coordinates
                    'y': float(i)
                },
                'elements': sample_elements,
                'predictions': {
                    'multiclass': {
                        'predicted_class': str(y_pred[i]),
                        'confidence': float(max(y_pred_proba[i])),
                        'probabilities': multiclass_probs
                    },
                    'binary': {
                        'predicted_class': binary_prediction,
                        'confidence': float(binary_confidence),
                        'probabilities': {
                            'soil': float(soil_prob),
                            'non-soil': float(non_soil_prob)
                        }
                    }
                }
            }
            
            # Add true label if available (for validation)
            if y_test is not None:
                sample_data['true_label'] = str(y_test.iloc[i])
                sample_data['correct_prediction'] = str(y_pred[i]) == str(y_test.iloc[i])
            
            test_samples.append(sample_data)
        
        # Create the complete data structure
        export_data = {
            'metadata': {
                'total_samples': int(len(test_samples)),
                'feature_names': self.element_columns,
                'class_names': self.label_encoder.classes_.tolist(),
                'export_timestamp': str(pd.Timestamp.now()),
                'model_info': {
                    'n_estimators': int(self.model.n_estimators),
                    'learning_rate': float(self.model.learning_rate),
                    'max_depth': int(self.model.max_depth)
                }
            },
            'samples': test_samples
        }
        
        # Add accuracy statistics if true labels are available
        if y_test is not None:
            correct_multiclass = sum(1 for sample in test_samples if sample.get('correct_prediction', False))
            multiclass_accuracy = correct_multiclass / len(test_samples)
            
            # Calculate binary accuracy
            correct_binary = sum(1 for i, sample in enumerate(test_samples) 
                               if (sample['predictions']['binary']['predicted_class'] == 'soil') == (y_test.iloc[i] == 'soil'))
            binary_accuracy = correct_binary / len(test_samples)
            
            export_data['metadata']['accuracy'] = {
                'multiclass': float(multiclass_accuracy),
                'binary': float(binary_accuracy)
            }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        file_size_kb = Path(output_path).stat().st_size / 1024
        print(f"Test data with predictions exported to {output_path}")
        print(f"File size: {file_size_kb:.2f} KB")
        print(f"Total samples: {len(test_samples)}")
        
        if y_test is not None:
            print(f"Multiclass accuracy: {export_data['metadata']['accuracy']['multiclass']:.1%}")
            print(f"Binary accuracy: {export_data['metadata']['accuracy']['binary']:.1%}")
        
        return export_data
    
    def demonstrate_material_prediction(self, X_test, y_test, n_samples=5):
        """
        Demonstrate both soil/non-soil and material type prediction on sample data
        Shows both binary and multi-class probabilities for each sample
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            n_samples: Number of samples to demonstrate
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        print("\n" + "="*80)
        print("DUAL CLASSIFICATION PREDICTION DEMONSTRATION")
        print("Showing both Soil/Non-Soil AND Specific Material Type Probabilities")
        print("="*80)
        
        # Select random samples
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        
        for i, idx in enumerate(indices):
            print(f"\n{'='*20} Sample {i+1} {'='*20}")
            
            # Get element concentrations
            sample_data = dict(zip(self.element_columns, X_test.iloc[idx].values))
            
            # Make prediction
            result = self.predict_single_sample(sample_data)
            
            print(f"True Material: {y_test.iloc[idx]}")
            print()
            
            # Binary Classification Results
            print("🔍 BINARY CLASSIFICATION (Soil vs Non-Soil):")
            print(f"  Prediction: {result['binary_prediction'].upper()}")
            print(f"  Confidence: {result['binary_confidence']:.1%}")
            print(f"  Probabilities:")
            print(f"    - Soil:     {result['binary_probabilities']['soil']:.3f} ({result['binary_probabilities']['soil']*100:.1f}%)")
            print(f"    - Non-Soil: {result['binary_probabilities']['non-soil']:.3f} ({result['binary_probabilities']['non-soil']*100:.1f}%)")
            print()
            
            # Multi-Class Classification Results  
            print("SPECIFIC MATERIAL TYPE CLASSIFICATION:")
            print(f"  Prediction: {result['predicted_class'].upper()}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  All Material Probabilities:")
            for material, prob in result['top_predictions']:
                indicator = "★" if material == result['predicted_class'] else " "
                print(f"   {indicator} {material:>8}: {prob:.3f} ({prob*100:.1f}%)")
            print()
            
            # Interpretation
            print(" INTERPRETATION:")
            print(f"  Primary: {result['interpretation']['primary_classification']}")
            if result['interpretation']['specific_material']:
                print(f"  Material: {result['interpretation']['specific_material']}")
            print(f"  Archaeological Significance: {result['interpretation']['archaeological_significance']}")
            print()
            
            # Show top 3 elements for this sample
            top_elements = sorted(sample_data.items(), key=lambda x: x[1], reverse=True)[:3]
            print("🧪 TOP ELEMENT CONCENTRATIONS:")
            for element, concentration in top_elements:
                print(f"  {element}: {concentration:.1f} ppm")


def main():
    """
    Main execution function
    """
    # Initialize the classifier
    classifier = PXRFMaterialClassifier(
        n_estimators=100,  
        learning_rate=0.1,
        max_depth=4,  
        random_state=42
    )
    
    # Load and prepare data
    # change to accept a file as input when running ARCH2.0
    data_file = input('Enter File Name: ')
    
    if not Path(data_file).exists():
        print(f"Data file {data_file} not found. Please ensure it's in the current directory.")
        return
    
    try:
        # Load and prepare data
        X, y = classifier.load_and_prepare_data(data_file)
        
        # Split data ensuring all material types are represented
        X_train, X_test, y_train, y_test = classifier.train_test_by_materials(X, y, test_size=0.2)
        
        # Perform cross-validation before final training
        cv_scores = classifier.cross_validate(X, y, cv_folds=5)
        
        # Train the model
        feature_importance = classifier.train_model(X_train, y_train)
        
        # Evaluate the model for multi-class material type classification
        y_pred, accuracy, y_pred_proba = classifier.evaluate_model_multiclass(X_test, y_test)


        # Train the model
        feature_importance = classifier.train_model(X_train, y_train)

        # Evaluate model
        y_pred, accuracy, y_pred_proba = classifier.evaluate_model_multiclass(X_test, y_test)

        # =========================
        # SAVE TRAINED MODEL HERE
        # =========================
        import joblib

        joblib.dump(classifier, "pxrf_full_model.pkl")
        print("Model saved as pxrf_full_model.pkl")


        
        # Get class labels
        classes = classifier.label_encoder.classes_

        # Find index of "soil" (if it exists)
        if 'soil' in classes:
            soil_idx = np.where(classes == 'soil')[0][0]
            
            # Non-soil probability = 1 - soil
            non_soil_probs = 1.0 - y_pred_proba[:, soil_idx]

        else:
            # If no soil class exists, just take the max probability of non-soil classes
            print("No 'soil' class found, using max probability as proxy")
            non_soil_probs = np.max(y_pred_proba, axis=1)

        # Store in array
        non_soil_probs_array = np.array(non_soil_probs)
        
        # Evaluate the model for binary soil vs non-soil classification
        y_binary_pred, binary_accuracy, binary_probabilities = classifier.evaluate_model_binary(X_test, y_test)
        
        # Export the compact model for Flutter
        classifier.export_model_compact("pxrf_model.json")
        
        # Demonstrate both binary and material type predictions with probabilities
        classifier.demonstrate_material_prediction(X_test, y_test, n_samples=5)
        
        # Export test data for Flutter app (with statistics printed but not saved)
        test_data = classifier.export_test_data_for_flutter(X_test, y_test, y_pred, 'test_data_for_flutter.csv')
        
        # Export test data with predictions and probabilities in JSON format
        test_predictions = classifier.export_test_data_with_predictions_json(X_test, y_test, 'test_data_with_predictions.json')
        
        print("\n" + "="*80)
        print("FINAL RESULTS - DUAL CLASSIFICATION MODEL")
        print("="*80)
        print(f"\n MULTI-CLASS CLASSIFICATION (Specific Materials):")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"\n BINARY CLASSIFICATION (Soil vs Non-Soil):")
        print(f"   Accuracy: {binary_accuracy:.4f} ({binary_accuracy*100:.2f}%)")
        print(f"\n Model successfully trained and exported!")
        print(f"   - Model saved as 'pxrf_model.json' (supports both classification types)")
        print(f"   - Test data saved as 'test_data_for_flutter.csv'")
        print(f"   - Test data with predictions saved as 'test_data_with_predictions.json'")
        print("\nMOBILE APP FEATURES:")
        print(f"   - Binary classification: Soil probability vs Non-soil probability")
        print(f"   - Multi-class classification: Specific material type probabilities")
        print(f"   - Archaeological significance assessment")
        
        # Show sample of the exported test data
        print(f"\nSample of exported test data (first 3 rows):")
        print(test_data.head(3).to_string())

        print("\nNon-soil probabilities (0–1 scale):")
        print(non_soil_probs_array)


    except Exception as e:
        print(f"Error during model training/testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
