from re import M, sub
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
import warnings

random_state = 42

def compute_points(predictions, gold_standard):
    points = 0
    for pred, gold in zip(predictions, gold_standard):
        if pred == gold:
            points += 1 if pred == 0 else 3    
        points += 1 if (pred > 0 and gold > 0) else 0
    
    return points

def compute_score(y_pred, y_test):
    points = compute_points(y_pred, y_test)
    max_points = compute_points(y_test, y_test)
    score = points/max_points
        
    return score

def evaluate_models(X_train, y_train, X_test, y_test, score_name):

    # Create a list of models
    models = []
    models.append(("LogisticRegression", LogisticRegression()))
    models.append(("XGBoost Classifier", XGBClassifier()))
    models.append(("RandomForestClassifier", RandomForestClassifier()))

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

            score = compute_score(labels, y_test)

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


def run_tuning(model, params, X_train_dec_sampled, y_train_dec_sampled, X_train_jan_sampled, y_train_jan_sampled, X_test_dec, y_test_dec, X_test_jan, y_test_jan, n_iter=5):
    # n_iter, cv and n_jobs can be adjusted to differentiate between accuracy and speed.
    bayes_search_estimator_dec = BayesSearchCV(estimator=model, search_spaces=params, n_iter=n_iter, cv=3, n_jobs=-1, random_state=random_state)
    bayes_search_estimator_jan = BayesSearchCV(estimator=model, search_spaces=params, n_iter=n_iter, cv=3, n_jobs=-1, random_state=random_state)

    with warnings.catch_warnings():
        # Ignore warnings that stem from illegal combinations of some hyperparameters for certain models.
        warnings.simplefilter("ignore")

        # Fit model on training data.
        bayes_search_estimator_dec.fit(X_train_dec_sampled, y_train_dec_sampled)
        bayes_search_estimator_jan.fit(X_train_jan_sampled, y_train_jan_sampled)

    # Get best estimator from bayes search.
    estimator_dec = bayes_search_estimator_dec.best_estimator_
    estimator_jan = bayes_search_estimator_jan.best_estimator_

    y_pred_dec = estimator_dec.predict(X_test_dec)
    y_pred_jan = estimator_jan.predict(X_test_jan)

    score_dec = compute_score(y_pred_dec, y_test_dec)
    score_jan = compute_score(y_pred_jan, y_test_jan)
    overall_score = (score_dec + score_jan) / 2

    return y_pred_dec, y_pred_jan, overall_score, estimator_jan


def get_final_eval(X_train_dec, y_train_dec, X_train_jan, y_train_jan, path_to_data, model="lr", n_iter=5):
    X_test_dec, y_test_dec = get_test_data(X_train_dec, "dec", path_to_data)
    X_test_jan, y_test_jan = get_test_data(X_train_jan, "jan", path_to_data)

    X_test_dec.fillna(0, inplace=True)
    X_test_jan.fillna(0, inplace=True)

    #Sampling
    sampler = SMOTE(sampling_strategy="not majority", random_state=random_state)

    X_train_dec_sampled, y_train_dec_sampled = sampler.fit_resample(X_train_dec, y_train_dec)
    X_train_jan_sampled, y_train_jan_sampled = sampler.fit_resample(X_train_jan, y_train_jan)

    # Create the logistic regression model.
    lr = LogisticRegression(random_state=random_state)

    # Create the random forest model.
    rf = RandomForestClassifier(random_state=random_state)

    # Create XGBoost model. 
    xgb = XGBClassifier(random_state=random_state)

    # Search parameters for bayesian optimization.
    param_grid_lr = {
        "penalty": ["l2", "none"],
        "C": (1e-5, 1e-3, "log-uniform"),
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        "max_iter": (100, 500)
    }

    # Search parameters for bayesian optimization.
    param_grid_rf = {
        "n_estimators": (100, 1000),
        "max_depth": (10, 1500),
        "min_samples_split": (2, 10),
        "min_samples_leaf": (1, 10)
    }

    # Search parameters for bayesian optimization.
    param_grid_xgb = {
        "learning_rate": (0.01, 1.0, "log-uniform"),
        "min_child_weight": (0, 10),
        "max_depth": (5, 50),
        "max_delta_step": (0, 20),
        "subsample": (0.01, 1.0, "uniform"),
        "colsample_bytree": (0.01, 1.0, "uniform"),
        "reg_lambda": (1e-9, 1000, "log-uniform"),
        "reg_alpha": (1e-9, 1.0, "log-uniform"),
        #"gamma": (1e-9, 0.5, "log-uniform"),
        "n_estimators": (50, 500),
    }

    # The model for paramter tuning is selected by comparing their base scores.
    if model == "lr":
        print("Currently fitting Logistic Regression Model")
        y_pred_dec_lr, y_pred_jan_lr, score_lr, model = run_tuning(lr, param_grid_lr, X_train_dec_sampled, y_train_dec_sampled, X_train_jan_sampled, y_train_jan_sampled, X_test_dec, y_test_dec, X_test_jan, y_test_jan, n_iter)
        print(f"Final score is: {score_lr}")
        return y_pred_dec_lr, y_pred_jan_lr, model
    elif model == "rf":
        print("Currently fitting Random Forest Model")
        y_pred_dec_rf, y_pred_jan_rf, score_rf, model = run_tuning(rf, param_grid_rf, X_train_dec_sampled, y_train_dec_sampled, X_train_jan_sampled, y_train_jan_sampled, X_test_dec, y_test_dec, X_test_jan, y_test_jan, n_iter)
        print(f"Final score is: {score_rf}")
        return y_pred_dec_rf, y_pred_jan_rf, model
    elif model == "xgb":
        print("Currently fitting XGBoost Model")
        y_pred_dec_xgb, y_pred_jan_xgb, score_xgb, model = run_tuning(xgb, param_grid_xgb, X_train_dec_sampled, y_train_dec_sampled, X_train_jan_sampled, y_train_jan_sampled, X_test_dec, y_test_dec, X_test_jan, y_test_jan, n_iter)
        print(f"Final score is: {score_xgb}")
        return y_pred_dec_xgb, y_pred_jan_xgb, model

