# A comparative analysis of ML techniques for bug report classification
This repository contains code and information about data for evaluating leading ML techniques for bug report classification.
# Datasets
We have used three datasets of bug reports.

**D1:** An open-source data provided by Herzig et al., [2013], https://doi.org/10.1109/ICSE.2013.6606585.

**D2:** An open-source data provided by Kallis et al., [2023], https://doi.org/10.1007/978-3-031-21388-5_34.

**D3:** The third dataset is from a product of a large closed-source company.
# Classical ML techniques for bug report classification
We have evaluated several classical ML techniques for bug report classification: Naive Bayes (NB), Logistic Regression (LR), Support Vector Machines (SVM), Random Forest (RF), Decision Tree (DT), K-Nearest Neighbors (KNN), and Stochastic Gradient Descent (SGD)-based classifier.
We used the sklearn library to implement the above techniques. The best parameters for the techniques were determined by performing the grid search.
# BERT-based techniques for bug report classification
We evaluated Roberta on the three datasets. To implement Roberta, we modified the script by Siddiq and Santos (https://github.com/s2e-lab/BERT-Based-GitHub-Issue-Classification).
# AutoML for bug report classification
In addition to the above-mentioned techniques that require manual training and optimization, we have also proposed and evaluated AutoML for bug report classification.
For AutoML, we used the Auto-sklearn library.
# How to reproduce
## For classical ML techniques
### Step 1: Run Grid search using GridSearchCV; see the example below.

<code>#Define the parameter grid for grid search for SGD classifier
sgd_param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'penalty': ['l1', 'l2'],
    'max_iter': [1000, 2000],
    'loss': ['hinge', 'log']
}
sgdclassifier = SGDClassifier()
sgd_grid_search = GridSearchCV(sgdclassifier, sgd_param_grid,  cv=10, n_jobs=-1)
sgd_grid_search.fit(X_tfidf, y)
print("Best Parameters:", sgd_grid_search.best_params_)
print("Best Score:", sgd_grid_search.best_score_)
</code>
### Step 2: Use the the best parameters for training a model.

<code>clf = SGDClassifier(**best_params)</code>
