# Hospital-Length-of-Stay
Predict length of stay following major surgery at a large metropolitan US hospital

Hospital length of stay (LOS) is defined as the length of time a patient remains in the hospital following a medical procedure. Being able to accurately predict LOS would reduce hospital and insurance costs, and free up healthcare workers to attend to the needs of additonal patients, and is therefore a major goal of the healthcare community. In this project, I made use of a rich dataset containing demographic information, lab values and other health data, and detailed information related to the procedure and post-operative status from nearly 800 patients who underwent major colorectal surgery to identify the features that are most predictive of LOS. To do so, I took the following steps:

1) Cleaned the data
2) Imputated missing values using random forest classification/regression exploratory data analysis
3) Engineered features to account for multicollinearity and class imbalance of categorical features
4) Compared the fit of Poisson and negative binomial distributions to the highly-skewed LOS distribution
5) Visualized data to glean relationships with LOS
6) Developed a multivariable negative binomial regression model using k-fold cross validation and root-mean-squared error

The major findings of this work are as follows:

    - Self-identification as Asian was associated with shorter length of stay relative to self-identified white patients
    - Higher serum albumin levels were associated with shorter length of stay
    - Pre-existing conditions were associated with longer length of stay, including:
      -Heart failure
      -Chronic obstructive pulmonary disease
      -Preoperative sepsis, which is systemic infalmmation brough on by the body's response to infection
    - The model was considerably more accurate at predicting LOS for patients with median or shorter LOS times. This is likely due to the undersampling
    of long LOS times in the data.
    
