# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
## Model Details
Andrea Vitali created the model, as a results of the project "Deploying a ML Model to Cloud Application Platform with FastAPI" of Udacity's "Machine Learning DevOps Engineer" nanodegree. It is logistic regression using the default hyperparameters in scikit-learn 1.4.2.
## Intended Use
The model should be used to predict whether a person income exceeds $50K/yr.
## Training Data
The training data is a 80% split of the [Census Income dataset](https://archive.ics.uci.edu/dataset/20/census+income).
More information about the data can be found on [UC Irvine's website](https://archive.ics.uci.edu/dataset/20/census+income).
## Evaluation Data
Evaluation data is a 20% split of the [Census Income dataset](https://archive.ics.uci.edu/dataset/20/census+income).
More information about the data can be found on [UC Irvine's website](https://archive.ics.uci.edu/dataset/20/census+income).
## Metrics
Train: Precision = 0.725 | Recall = 0.453 | Fbeta = 0.558
Test : Precision = 0.713 | Recall = 0.454 | Fbeta = 0.555
## Ethical Considerations
The dataset is representative of a specific social group: from the [UC Irvine's website](https://archive.ics.uci.edu/dataset/20/census+income): 
```
A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
```
Therefore careful considerations should be performed before making conclusions on the model's predictions.
## Caveats and Recommendations
I am not the owner of the data, contact [UC Irvine](https://archive.ics.uci.edu/dataset/20/census+income) for more informations about it.