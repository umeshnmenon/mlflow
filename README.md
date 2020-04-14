# mlflow

An integrated set of tools to manage the ML model end-to-end workflow.

The idea is to help enable a successful model orchestration at enterprise scale by using simple set of apis. As a grand 
vision, the package aims to cover the below components:

- Data Processing
- Model Training
- Model Governance
- Model Publishing
- Model Serving
- Evaluation Governance
- Prediction Pipeline
- Continuous Monitoring

## How to Use

Have a look at the `tester.py` file or `MLFlow Demo' Jupyter Notebook to know how to use this package. 

```python
    from pymodel import *
    import sys
    from client import Client

    # Training is independent of MLFLow
    import sklearn
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    import pandas as pd

    # Load dataset
    data = load_breast_cancer()

    # Organize our data
    label_names = data['target_names']
    labels = data['target']
    feature_names = data['feature_names']
    features = data['data']

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.33,
                                                        random_state=42)

    # Initialize our classifier
    gnb = GaussianNB()

    # Train our classifier
    model = gnb.fit(X_train, y_train)

    train_pred_probs = model.predict_proba(X_train)[:, 0]

    valid_pred_probs = model.predict_proba(X_test)[:, 0]

    # # Gaussian Naive Bayes
    # from sklearn import datasets
    # from sklearn import metrics
    # from sklearn.naive_bayes import GaussianNB
    # # load the iris datasets
    # dataset = datasets.load_iris()
    # # fit a Naive Bayes model to the data
    # model = GaussianNB()
    # model.fit(dataset.data, dataset.target)
    # train_pred_probs = model.predict_proba(dataset.data)[:, 0]
    # create a pymodel
    pymodel = PyModel(model=model, features=feature_names, split=0.33, feature_columns=feature_names,
                      target_column=label_names,
                      model_uuid="Demo_Model1", model_tag="Demo_Model", model_version="0.1",
                      train_actuals=y_train, train_pred_probs=train_pred_probs, valid_actuals=y_test,
                      valid_pred_probs=valid_pred_probs)
    df = pymodel()
    print(df)
    print(df["model_uuid"])
    print(df["model_tag"])
    print(df["model_version"])
    print(df["feature_columns"])
    print(df["target_column"])
    print(df["training_args"])
    print(df["hyperparams"])
    print(df["split"])
    print(df["training_metrics"])
    print(df["validation_metrics"])

```

**Using Client**
```python
    from client import Client
    client = Client()
    #model = test_model()
    #client.upload_model(model)
    print(client.list_models("test"))
```
