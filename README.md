# ML techniques for issue report classification
This repository contains code and information about data for evaluating leading ML techniques for issue report classification.
# Related articles
**Paper-1:** "A comparative analysis of ML techniques for bug report classification" by Laiq et al., [2025], [https://doi.org/10.1109/ICSE.2013.6606585](https://doi.org/10.1016/j.jss.2025.112457).

**Paper-2:** "Industrial adoption of machine learning techniques for early identification of invalid bug reports" by Laiq et al., [2024], [https://doi.org/10.1109/ICSE.2013.6606585](https://link.springer.com/article/10.1007/s10664-024-10502-3).

**Paper-3:** "Early Identification of Invalid Bug Reports in Industrial Settings – A Case Study" by Laiq et al., [2022], [https://doi.org/10.1109/ICSE.2013.6606585](https://link.springer.com/chapter/10.1007/978-3-031-21388-5_34).

# Datasets
We have used three datasets of issue reports.

**D1:** An open-source dataset provided by Herzig et al., [2013], https://doi.org/10.1109/ICSE.2013.6606585.

**D2:** An open-source dataset provided by Kallis et al., [2023], [https://doi.org/10.1007/978-3-031-21388-5_34](https://github.com/nlbse2023/issue-report-classification).

**D3:** The third dataset is from a product of a large closed-source company.

# Classical ML techniques for issue report classification
We have evaluated several classical ML techniques for issue report classification: Naive Bayes (NB), Logistic Regression (LR), Support Vector Machines (SVM), Random Forest (RF), Decision Tree (DT), K-Nearest Neighbors (KNN), and Stochastic Gradient Descent (SGD)-based classifier.
We used the sklearn library to implement the above techniques. The best parameters for the techniques were determined by performing the grid search.
# BERT-based techniques for issue report classification
We evaluated BERT and RoBERTa on the three datasets.
# AutoML for issue report classification
In addition to the above-mentioned techniques that require manual training and optimization, we have also proposed and evaluated AutoML for bug report classification.
For AutoML, we used the Auto-sklearn library.
# How to reproduce
## For classical ML techniques
### Step 1: Run Grid search using GridSearchCV; see the example below. 
To apply Grid search to each technique, use the parameters provided in the bottom cells of the Jupyter notebook file (Classical_ML_for_IRC.ipynb).

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
### Step 2: Use the best parameters for training a model.
<code>clf = SGDClassifier(**best_params)</code>
## For AutoML, see details in AutoML_for_IRC.ipynb
<code>#Create an AutoML-based model
automl_classifier = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=90*60, # set time
    resampling_strategy="cv",
    resampling_strategy_arguments={"folds": 10},
    metric=roc_auc, # set evaluation metric
    scoring_functions=[roc_auc, average_precision, accuracy, f1, precision, recall]
)
#fit the model 
automl_classifier.fit(X_train, y_train)
#Check results
perfor_over_time =automl_classifier.performance_over_time_
#Save results
perfor_over_time.to_csv('perfor_over_time_90mintues_cv_scores.csv', index=False) 
perfor_over_time
</code>
## For BERT-based models (RoBERTa.ipynb)
Use RobertaForSequenceClassification and RobertaTokenizer or BertForSequenceClassification and BertTokenizer from transformers.
To implement RoBERTa, we modified the script by Siddiq and Santos (https://github.com/s2e-lab/BERT-Based-GitHub-Issue-Classification).

<code>E.g.,
#Load the RoBERTa tokenizer
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
#Load the RoBERTa model for sequence classification:
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)</code>

Further details for customization can be found at: https://huggingface.co/docs/transformers/en/model_doc/roberta#transformers.RobertaModel.

# Requirements & Libraries
<code>Python 3.9 or 3.9.13
transformers
torch
scikit-learn
auto-sklearn
pandas
numpy
gensim
psutil
PipelineProfiler
cuda setup if GPU is available </code>
