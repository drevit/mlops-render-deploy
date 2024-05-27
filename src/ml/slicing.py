import numpy as np
from src.ml.data import process_data
from src.ml.model import inference, compute_model_metrics


def compute_slice_metrics(df, cat_features, slice_feature, label, model, encoder, lb):
    """ 
    Computes fbeta, precision and recall on the unique values of a feature.

    Inputs
    ------
    df : pd.DataFrame
        DataFrame that will undergo evaluation
    cat_features: list
        list of categorical features
    slice_feature: str
        df column where the catogorical variable is stored
    label: str
        df column where the target variable is stored
    model:
        Trained machine learning model
    encoder:
        Trained one hot encoder
    lb:
        Trained label binarizer
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    print(f"\n==========\n{slice_feature}\n==========\n")
    for unique_val in df[slice_feature].unique():
        
        

        df_unique = df.loc[df[slice_feature] == unique_val, :]
        X_unique, y_unique, _, _ = process_data(
            df_unique, categorical_features=cat_features, label=label, training=False,
            encoder=encoder, lb=lb
            )

        y_pred = inference(model, X_unique)

        precision, recall, fbeta = compute_model_metrics(y_unique, y_pred)

        print(f"{slice_feature} - {unique_val}:".ljust(30) + f" | prec: {np.round(precision, 3)} | recall: {np.round(recall, 3)} | fbeta: {np.round(fbeta, 3)}")

    return 
