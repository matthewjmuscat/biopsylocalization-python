import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import dataframe_builders
from sklearn.model_selection import KFold


def random_forest_global_tissue_class_score_predicted_by_tumor_morphology(cohort_global_tissue_scores_with_target_dil_radiomic_features_df):

    df = cohort_global_tissue_scores_with_target_dil_radiomic_features_df
   

    # Assuming `df` is your DataFrame
    # df = pd.read_csv("your_dataset.csv") # Load your data

    # One-hot encode the 'Simulated Type' column
    df_encoded = pandas.get_dummies(df, columns=[
    'Simulated type', 
    'DIL prostate sextant (LR)', 
    'DIL prostate sextant (AP)', 
    'DIL prostate sextant (SI)'], drop_first=False)

    # Initialize the regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Prepare DataFrame to store results and metrics
    results = pandas.DataFrame(columns=['Patient ID', 'Bx ID', 'Relative structure ROI', 'Actual', 'Predicted'])
    metrics_df = pandas.DataFrame(columns=['Patient ID', 'MAE', 'MSE', 'RMSE', 'R²'])

    # Leave-One-Patient-Out CV
    logo = LeavePGroupsOut(n_groups=1)

    # Define feature columns (now including the one-hot encoded 'Simulated Type' columns)
    feature_columns = [
    "Volume", "Surface area", "Surface area to volume ratio", "Sphericity", 
    "Compactness 1", "Compactness 2", "Spherical disproportion", "Maximum 3D diameter",
    "PCA major", "PCA minor", "PCA least", "Major axis (equivalent ellipse)", 
    "Minor axis (equivalent ellipse)", "Least axis (equivalent ellipse)", "Elongation", 
    "Flatness", "L/R dimension at centroid", "A/P dimension at centroid", 
    "S/I dimension at centroid",
    "DIL centroid (X, prostate frame)", "DIL centroid (Y, prostate frame)",
    "DIL centroid (Z, prostate frame)", "DIL centroid distance (prostate frame)"
    ] + [
        col for col in df_encoded.columns 
        if col.startswith('Simulated type_') 
        or col.startswith('DIL prostate sextant (LR)_')
        or col.startswith('DIL prostate sextant (AP)_')
        or col.startswith('DIL prostate sextant (SI)_')
    ]

    # Extract features, target variable, Bx ID, and Relative structure ROI
    X = df_encoded[feature_columns]
    y = df_encoded['Global mean binom est']
    bx_ids = df_encoded['Bx ID']
    relative_rois = df_encoded['Relative structure ROI']
    groups = df_encoded['Patient ID']

    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        bx_id_test = bx_ids.iloc[test_index]
        relative_roi_test = relative_rois.iloc[test_index]
        
        # Train the model
        rf_regressor.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_regressor.predict(X_test)
        
        # Append actual vs predicted values to the results DataFrame
        for bx_id, relative_roi, actual, predicted in zip(bx_id_test, relative_roi_test, y_test, y_pred):
            
            # .append method was removed in pandas>2.0 update. Revised code uses concat instead
            """
            results = results.append({
                'Patient ID': groups.iloc[test_index].iloc[0],
                'Bx ID': bx_id,
                'Relative structure ROI': relative_roi,
                'Actual': actual,
                'Predicted': predicted
            }, ignore_index=True)
            """
            # For results DataFrame
            new_result = pandas.DataFrame([{
                'Patient ID': groups.iloc[test_index].iloc[0],
                'Bx ID': bx_id,
                'Relative structure ROI': relative_roi,
                'Actual': actual,
                'Predicted': predicted
            }])
            results = pandas.concat([results, new_result], ignore_index=True)

        # Calculate error metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # .append method was removed in pandas>2.0 update. Revised code uses concat instead
        """
        # Append metrics to the metrics DataFrame
        metrics_df = metrics_df.append({
            'Patient ID': groups.iloc[test_index].iloc[0],
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }, ignore_index=True)
        """
        new_metric = pandas.DataFrame([{
        'Patient ID': groups.iloc[test_index].iloc[0],
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
        }])
        metrics_df = pandas.concat([metrics_df, new_metric], ignore_index=True)


    metrics_df = dataframe_builders.convert_columns_to_categorical_and_downcast(metrics_df, threshold=0.25)

    results = dataframe_builders.convert_columns_to_categorical_and_downcast(results, threshold=0.25)

    return metrics_df, rf_regressor, results, feature_columns



def random_forest_global_tissue_class_score_predicted_by_tumor_morphology_single_patient_data(df):
    # One-hot encode the relevant columns
    df_encoded = pandas.get_dummies(df, columns=[
        'Simulated type', 
        'DIL prostate sextant (LR)', 
        'DIL prostate sextant (AP)', 
        'DIL prostate sextant (SI)'
    ], drop_first=False)

    # Initialize the regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Prepare DataFrame to store results and metrics
    results = pandas.DataFrame(columns=['Patient ID', 'Bx ID', 'Relative structure ROI', 'Actual', 'Predicted'])
    metrics_df = pandas.DataFrame(columns=['Patient ID', 'MAE', 'MSE', 'RMSE', 'R²'])

    # Initialize K-Fold cross-validator
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Define feature columns (including the one-hot encoded columns)
    feature_columns = [col for col in df_encoded.columns if col not in ['Global mean binom est', 'Bx ID', 'Relative structure ROI', 'Patient ID']]
    
    # Extract features, target variable, Bx ID, and Relative structure ROI
    X = df_encoded[feature_columns]
    y = df_encoded['Global mean binom est']
    bx_ids = df_encoded['Bx ID']
    relative_rois = df_encoded['Relative structure ROI']
    groups = df_encoded['Patient ID']

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        bx_id_test = bx_ids.iloc[test_index]
        relative_roi_test = relative_rois.iloc[test_index]
        patient_id_test = groups.iloc[test_index]

        # Train the model
        rf_regressor.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_regressor.predict(X_test)

        # Append actual vs predicted values to the results DataFrame
        new_result = pandas.DataFrame({
            'Patient ID': patient_id_test,
            'Bx ID': bx_id_test,
            'Relative structure ROI': relative_roi_test,
            'Actual': y_test,
            'Predicted': y_pred
        })
        results = pandas.concat([results, new_result], ignore_index=True)

        # Calculate error metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        new_metric = pandas.DataFrame({
            'Patient ID': patient_id_test.iloc[0],  # Assuming each test set has a unique patient ID
            'MAE': [mae],
            'MSE': [mse],
            'RMSE': [rmse],
            'R²': [r2]
        })
        metrics_df = pandas.concat([metrics_df, new_metric], ignore_index=True)

    return metrics_df, rf_regressor, results, feature_columns