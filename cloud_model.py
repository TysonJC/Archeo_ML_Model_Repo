import numpy as np
import pandas as pd
import joblib
import __main__
from pathlib import Path

from gbdt_model_renewed import PXRFMaterialCLassifier

#For compatibility when saving the model
__main__.PXRFMaterialClassifier = PXRFMaterialClassifier

FEATURE_COLUMNS = [
    "Ag","Al","As","Au","Ca","Cr",
    "Cu","Fe","K","Mg","Mn","Ni",
    "P","Pb","Sr","Ti","V","Zn"
]



class PXRFModel:

    def __init__(self, model_path="pxrf_classifier.pkl"):

        base_dir = Path(__file__).resolve().parent
        model_file = base_dir / model_path

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_file}"
            )

        self.classifier = joblib.load(model_file)

        print("PXRF classifier loaded")


    def prepare_features(self, df):
        """
        Convert uploaded pXRF data into the exact feature
        structure used when the model was trained.
        """

        X = pd.DataFrame(index=df.index)

        for element in FEATURE_COLUMNS:

            if element in df.columns:

                X[element] = pd.to_numeric(
                    df[element],
                    errors="coerce"
                )

            else:
                # Column absent from uploaded data
                X[element] = np.nan

        # This reproduces training preprocessing:
        # missing measurements become zero.
        X = X.fillna(0)

        # Defensive cleanup
        X[X < 0] = 0

        # Make sure column order is identical to training
        X = X[FEATURE_COLUMNS]

        return X


    def predict(self, df):

        X = self.prepare_features(df)

        # Apply scaler learned during training
        X_scaled = self.classifier.scaler.transform(X)

        # Predictions
        encoded_predictions = (
            self.classifier.model.predict(X_scaled)
        )

        probabilities = (
            self.classifier.model.predict_proba(X_scaled)
        )

        predicted_materials = (
            self.classifier.label_encoder.inverse_transform(
                encoded_predictions
            )
        )

        classes = self.classifier.label_encoder.classes_

        # Find soil class
        soil_index_array = np.where(
            classes == "soil"
        )[0]

        if len(soil_index_array) == 0:
            raise ValueError(
                "The trained classifier does not contain a soil class."
            )

        soil_index = soil_index_array[0]

        soil_probabilities = probabilities[:, soil_index]
        non_soil_probabilities = 1 - soil_probabilities

        # Preserve uploaded data
        result = df.copy()

        # Add predictions
        result["predicted_material"] = predicted_materials
        result["soil_probability"] = soil_probabilities
        result["non_soil_probability"] = (
            non_soil_probabilities
        )

        result["prediction_confidence"] = (
            np.max(probabilities, axis=1)
        )

        return result
