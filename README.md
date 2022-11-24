# Credit_Risk_Analysis
## Overview

This challenge assignment examines the use of supervised machine learning to categorize loan applications as high-risk or low-risk. The nature of supervised machine learning requires training a machine learning algorithm on historical data of whether or not loan applications were approved or rejected based on a number of variables concerning the applicant's credit risk. Because the vast majority of the loan applications in the dataset were approved, they were assessed as being "low-risk." Since so few of the applicants were rejected and considered "high-risk," the best strategy to develop a useful algorithm for credit risk assessment would be to re-sample the data so that the relatively small number of high-risk applications are amplified for the machine learning training step to more accurately determine the differences between the two categorizations. Six different re-sampling techniques were conducted so that the precision and accuracy of the different methods can be contrasted to determine which model performs best. 

### Resources
- Data: LoanStats_2019Q1.csv
- Software: Visual Studio Code 1.70.1, Python 3.7.13, Anaconda command line client 1.9.0, Conda 4.13.0, Jupyter Notebook 6.4.8, Pandas 1.3.5, NumPy 1.21.5, SciPy 1.7.3, SciKit-Learn 1.0.2, Imbalanced-Learn 0.7.0

## Results
### Naive Random Oversampling

![Random Oversampling](https://github.com/tfish110/Credit_Risk_Analysis/blob/main/Resources/random_oversampler_results.jpg)

- Balanced accuracy score: 62.94%
- Precision score:
    - Overall: 99%
    - Majority class ("low-risk"): 100%
    - Minority class ("high-risk"): 1%
- Recall Score:
    - Overall: 68%
    - Majority class ("low-risk"): 68%
    - Minority class ("high-risk"): 57%

### SMOTE Oversampling

![SMOTE Oversampling](https://github.com/tfish110/Credit_Risk_Analysis/blob/main/Resources/SMOTE_results.jpg)

- Balanced accuracy score: 62.77%
- Precision score:
    - Overall: 99%
    - Majority class ("low-risk"): 100%
    - Minority class ("high-risk"): 1%
- Recall Score:
    - Overall: 63%
    - Majority class ("low-risk"): 63%
    - Minority class ("high-risk"): 62%

### Clustered Centroid Undersampling

![Clustered Centroid Undersampling](https://github.com/tfish110/Credit_Risk_Analysis/blob/main/Resources/clustered_centroid_results.jpg)

- Balanced accuracy score: 51.32%
- Precision score:
    - Overall: 99%
    - Majority class ("low-risk"): 100%
    - Minority class ("high-risk"): 1%
- Recall Score:
    - Overall: 43%
    - Majority class ("low-risk"): 43%
    - Minority class ("high-risk"): 60%

### SMOTEENN Combination Sampling

![SMOTEENN Combination Sampling](https://github.com/tfish110/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN_results.jpg)

- Balanced accuracy score: 65.48%
- Precision score:
    - Overall: 99%
    - Majority class ("low-risk"): 100%
    - Minority class ("high-risk"): 1%
- Recall Score:
    - Overall: 61%
    - Majority class ("low-risk"): 61%
    - Minority class ("high-risk"): 70%

### Balanced Random Forest Classifier

![Balanced Random Forest Classifier](https://github.com/tfish110/Credit_Risk_Analysis/blob/main/Resources/random_forest_classifier_results.jpg)

- Balanced accuracy score: 67.21%
- Precision score:
    - Overall: 100%
    - Majority class ("low-risk"): 100%
    - Minority class ("high-risk"): 73%
- Recall Score:
    - Overall: 100%
    - Majority class ("low-risk"): 100%
    - Minority class ("high-risk"): 34%

### Easy Ensemble Classifier

![Easy Ensemble Classifier error](https://github.com/tfish110/Credit_Risk_Analysis/blob/main/Resources/easy_ensemble_classifier_error.jpg)

Unfortunately, I was unable to get the final resampling technique to function properly. As shown here, the error message indicated that the classifier object I imported does not have an attribute called "n_features_in_" that the "fit" function requires to execute properly.

![Easy Ensemble Classifier documentation](https://github.com/tfish110/Credit_Risk_Analysis/blob/main/Resources/easy_ensemble_classifier_documentation.jpg)

Here, in the documentation for Easy Ensemble Classifier, the attributes listed seem to indicate that "n_feautres_in_" should indeed be present for the "fit" function to utilize, so I'm not sure why it didn't run properly. I was under the assumption that the documentation page cited here that the module and challenge materials directed me to would be for the same version as I was instructed to import. But, I suspect that assumption may not be true, since the "n_features_" entry in the list attributes is very similar to the one flagged in the error message. As per this documentation, it seems like "n_features_in_" should have replaced "n_features_" but the error seems to indicate that I do not have the correct version of this library imported.

Since I was not able to get past this step to get results of my own for this resampling technique, I will use the example provided in the starter code for the purposes of this analysis.

![Easy Ensemble Classifier results](https://github.com/tfish110/Credit_Risk_Analysis/blob/main/Resources/easy_ensemble_classifier_results.jpg)

- Balanced accuracy score: 93.17%
- Precision score:
    - Overall: 99%
    - Majority class ("low-risk"): 100%
    - Minority class ("high-risk"): 9%
- Recall Score:
    - Overall: 94%
    - Majority class ("low-risk"): 94%
    - Minority class ("high-risk"): 92%

## Summary

These six different models for logistic regression classification of credit risk assessment do a good job of highlighting the issues surrouding resampling as a solution for imbalanced datasets. The first characteristic of note is that the precision scores for the majority class in all six models was 100%. Because of the large imbalance between the majority and minority classes in this dataset, all resampling techniques were able to achieve such high precision since they did not need to generate any artificial data.

Clustered Centroid Undersampling can easily be discarded here due to its low accuracy score, barely performing better than random chance. Naive Random Oversampling and SMOTE Oversampling both achieved an accuracy score approaching 63%, which is a bit bitter than Clustered Centroid Undersampling, but is still lower than would be ideal. Between those two, the SMOTE technique seems slightly better due to its higher, and more balanced, recall scores for the two classes.

SMOTEENN's accuracy score is a bit higher than the previous techniques at about 65%, which is an improvement, but would preferably be even higher. However, the major problem with these first four technighes is that the precision scores for the two classes is so disparate.

The Random Forest Classifier is the only one of these techniques which has a relatively high precision score for the minorty class, which certainly makes it stand out from the rest. With a score of 67%, it is approaching a much more respectable level of accuracy. But, it also has the worst minority class recall score.

While I had to rely on an example set of results rather than those I generated on my own, the Easy Ensemble Classifier results here have a clear advantage over the other techniques with an impressive accuract score of 93%, and an overall recall score of 94%, well-balanced between both classes. Minority class precision is still quite low, but this seems like it may just be an artifact of how disparate the sample sizes of the two classes are, and can't quite be overcome by any resampling technique. If it was much higher, over-fitting might be a concern. But because this low precision for the minority class is reflected in the original dataset, the impressive accuracy and recall scores indicate that the Easy Ensemble Classifier is likely the best technique for this particular dataset.