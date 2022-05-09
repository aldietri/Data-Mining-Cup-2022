import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
        

def evaluate_models(X_train, y_train, X_test, y_test):

    # Create a list of models
    models = []
    models.append(("LogisticRegression", LogisticRegression()))
    models.append(("GaussianNB", GaussianNB()))
    models.append(("KNeighborsClassifier", KNeighborsClassifier()))
    models.append(("DecisionTreeClassifier", DecisionTreeClassifier()))
    models.append(("RandomForestClassifier", RandomForestClassifier()))

    # A dictionary for all the distinct models 
    model_scores_dict = {"model_name" : [], 
                     "score"   : [], 
                     "time"    : []
                    }

    #Sampling
    sampler = SMOTE(sampling_strategy="not majority", random_state=random_state)

    X_train_sampled, y_train_sampled = sampler.fit_resample(X_train, y_train)

    # Testing all models
    best_score = 0
    best_labels = []
    with warnings.catch_warnings():
        # Ignore warnings for readability
        warnings.simplefilter("ignore")

        for model_name, model in models:

            start = time.time()

            model.fit(X_train_sampled, y_train_sampled)

            labels = model.predict(X_test)

            points = compute_points(labels, y_test)
            max_points = compute_points(y_test, y_test)
            score = points/max_points

            if score > best_score:
                best_labels = labels

            model_scores_dict["model_name"].append(model_name)
            model_scores_dict["score"].append(score)
            model_scores_dict["time"].append(time.time() - start)
        
        model_score_df = pd.DataFrame(model_scores_dict).set_index("model_name")
        model_score_df.sort_values(by=["score"], ascending=False)
        display(model_score_df)

    return best_labels

