from re import sub
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import time
import warnings

random_state = 42

def compute_points(predictions, gold_standard):
    points = 0
    for pred, gold in zip(predictions, gold_standard):
        if pred == gold:
            points += 1 if pred == 0 else 3    
        points += 1 if (pred > 0 and gold > 0) else 0
    
    return points
        

def evaluate_models(X_train, y_train, X_test, y_test, score_name):

    # Create a list of models
    models = []
    models.append(("LogisticRegression", LogisticRegression()))
    models.append(("XGBoost Classifier", XGBClassifier()))
    # models.append(("RandomForestClassifier", RandomForestClassifier()))

    # A dictionary for all the distinct models 
    model_scores_dict = {"model_name" : [], 
                     score_name   : [], 
                    }

    #Sampling
    sampler = SMOTE(sampling_strategy="not majority", random_state=random_state)

    X_train_sampled, y_train_sampled = sampler.fit_resample(X_train, y_train)

    # Testing all models
    best_score = 0
    with warnings.catch_warnings():
        # Ignore warnings for readability
        warnings.simplefilter("ignore")

        for model_name, model in models:

            model.fit(X_train_sampled, y_train_sampled)

            labels = model.predict(X_test)

            points = compute_points(labels, y_test)
            max_points = compute_points(y_test, y_test)
            score = points/max_points

            if score > best_score:
                best_labels = labels

            model_scores_dict["model_name"].append(model_name)
            model_scores_dict[score_name].append(score)
        
        model_score_df = pd.DataFrame(model_scores_dict).set_index("model_name")
        model_score_df.sort_values(by=[score_name], ascending=False)

    return model_score_df, best_labels

def get_test_data(X_train, month,  path_to_data):
    test = pd.read_csv(f"{path_to_data}/test_{month}.csv", sep="|")
    test = pd.merge(test, X_train, on=["userID", "itemID"], how="left").drop_duplicates(["userID", "itemID"])

    test_target = test["prediction"]
    test_body = test.drop(columns="prediction")

    return test_body, test_target

def get_submission_csv(pred_dec, pred_jan, path_to_data):
    submission_dec = pd.read_csv(f"{path_to_data}/test_dec.csv", sep="|")
    submission_dec["prediction"] = pred_dec

    submission_jan = pd.read_csv(f"{path_to_data}/test_jan.csv", sep="|")
    submission_jan["prediction"] = pred_jan

    return submission_dec, submission_jan

def calculate_feature_gain(X_train_dec, y_train_dec, X_train_jan, y_train_jan, features_to_test, path_to_data):

    X_test_dec, y_test_dec = get_test_data(X_train_dec, "dec", path_to_data)
    
    base_features = ["userID", "itemID", "order", "brand", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

    features = base_features + features_to_test

    df_dec_without_features, labels_dec_without_features = evaluate_models(X_train_dec[base_features], y_train_dec, X_test_dec[base_features], y_test_dec, "w/o features dec")
    df_dec_with_features, labels_dec_with_features = evaluate_models(X_train_dec[features], y_train_dec, X_test_dec[features], y_test_dec, "with features dec")

    df_dec = pd.merge(df_dec_without_features, df_dec_with_features, on=["model_name"])
    df_dec["gain dec"] = df_dec["with features dec"] - df_dec["w/o features dec"]

    X_test_jan, y_test_jan = get_test_data(X_train_jan, "jan", path_to_data)

    df_jan_without_features, labels_jan_without_features = evaluate_models(X_train_jan[base_features], y_train_jan, X_test_jan[base_features], y_test_jan, "w/o features jan")
    df_jan_with_features, labels_jan_with_features = evaluate_models(X_train_jan[features], y_train_jan, X_test_jan[features], y_test_jan, "with features jan")

    df_jan = pd.merge(df_jan_without_features, df_jan_with_features, on=["model_name"])
    df_jan["gain jan"] = df_jan["with features jan"] - df_jan["w/o features jan"]

    df = pd.merge(df_dec, df_jan, on=["model_name"])
    df["overall score w/o features"] = (df["w/o features dec"] + df["w/o features jan"]) / 2
    df["overall score with features"] = (df["with features dec"] + df["with features jan"]) / 2
    df["overall gain"] = df["overall score with features"] - df["overall score w/o features"]

    if df_dec_without_features.max().max() > df_dec_with_features.max().max():
        best_labels_dec = labels_dec_without_features
    else: 
        best_labels_dec = labels_dec_with_features

    if df_jan_without_features.max().max() > df_jan_with_features.max().max():
        best_labels_jan = labels_jan_without_features
    else: 
        best_labels_jan = labels_jan_with_features

    submission_dec, submission_jan = get_submission_csv(best_labels_dec, best_labels_jan, path_to_data)

    return df, submission_dec, submission_jan