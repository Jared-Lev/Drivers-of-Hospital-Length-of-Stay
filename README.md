# Hospital Length of Stay
Predict length of stay following major surgery at a large metropolitan US hospital

Hospital length of stay (LOS) is defined as the length of time a patient remains in the hospital following a medical procedure. Being able to accurately predict LOS is a major goal of the healthcare community as it would reduce hospital and insurance costs, and free up healthcare workers to attend to the needs of more patients. Here I used a dataset containing demographic information, general health data, lab values, and detailed procedural data
from patients who underwent major colorectal surgery to identify the features that are most predictive of LOS. To do so, I took the following steps:

1) Cleaned the data and imputed missing values using random forest classification/regression
2) Performed exploratory data analysis/visualization, and hypothesis-testing with parametric and non-parametric methods
to identify features of interest for predictive modeling
4) Engineered features to account for multicollinearity and class imbalance of categorical features
5) Compared the fit of Poisson and negative binomial distributions to the highly-skewed LOS distribution, before developing
a multivariable negative binomial regression model using k-fold cross validation to predict LOS. 
![image](https://user-images.githubusercontent.com/89553765/195164871-6085115b-1d44-4f34-bbb6-37cf43389d06.png)


The major findings of this work are as follows:

- Self-identification as Asian was associated with shorter length of stay relative to self-identified white patients
- Higher serum albumin levels were associated with shorter length of stay
- Pre-existing conditions were associated with longer length of stay, including:
      -Heart failure
      -Chronic obstructive pulmonary disease
      -Preoperative sepsis, which is systemic inflammation caused by the body's response to infection
    ![image](https://user-images.githubusercontent.com/89553765/195147500-a204d597-4ebd-454b-a550-e81bba171ce3.png)
    
- The model was considerably more accurate at predicting LOS for patients with median or shorter LOS times. 
    This is likely due to the undersampling of long LOS times in the data.

![image](https://user-images.githubusercontent.com/89553765/195165798-52bb44aa-0b5d-4d4b-8792-1e52cd7c9e6a.png)






