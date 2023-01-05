# Predicting Hospital Length of Stay
Predict length of stay following major surgery at a large metropolitan US hospital

Hospital length of stay (LOS) is defined as the length of time a patient remains in the hospital following a medical procedure. Being able to accurately predict LOS is a major goal of the healthcare community as it would reduce hospital and insurance costs, and free up healthcare workers to attend to the needs of more patients. Here I used a dataset containing demographic information, general health status, lab values, and detailed procedural information
from patients who underwent major colorectal surgery to identify the features that are most predictive of LOS. To do so, I took the following steps:

1) Cleaned the data and imputed missing values using random forest classification/regression
2) Performed exploratory data analysis/visualization, and hypothesis-testing with parametric and non-parametric methods
to identify features of interest for predictive modeling
3) Engineered features to account for multicollinearity and class imbalance of categorical features
4) Developed a multivariable negative binomial regression model using k-fold cross validation to: 1) predict LOS, and 2)identify
important predictors of LOS.

![image](https://user-images.githubusercontent.com/89553765/197885769-83942a66-670e-4b1f-939a-daf9dc0c2cdc.png)



The major findings of this work are as follows:


- Higher serum albumin levels were associated with shorter length of stay: each unit increase of albumin shortens LOS
by 0.8 days.
- Higher international normalized ratio (INR) is associated with longer LOS: each unit increase extends LOS by 1.8 days
- Pre-existing conditions were associated with longer length of stay, including:
      -Heart failure (1.65 days longer)
      -Chronic obstructive pulmonary disease (1.2 days longer)
      
    ![image](https://user-images.githubusercontent.com/89553765/195166397-2200463c-c5de-4c65-a8b6-b8ef093bba10.png)
    
- The model was considerably more accurate at predicting LOS for patients with LOS less than 12 days, who make up 88% of the 
dataset. 
    The error in predicting longer LOS is due to the undersampling of long LOS times in the dataset, and would be reduced with more data.
    
![image](https://user-images.githubusercontent.com/89553765/198186332-6b6d062c-f143-4530-a0c5-44f0b34eff98.png)



Implications and future directions:

- Albumin is a protein made by the liver that has two important functions in the body. Firstly, it acts as a carrier
in the blood for many small molecules (e.g. ions, hormones). Secondly, it prevents blood from leaking out of the blood
vessels. Given the substantial effect of albumin on LOS, knowing what factors might influence albumin levels could well
improve surgical outcomes. 

![image](https://user-images.githubusercontent.com/89553765/197847978-c4479edc-d8c3-4b23-9a86-1b45dec62197.png)

 Clearly, albumin decreases with age. Furthermore, functional dependence is associated with lower albumin levels (an older age). It may 
 therefore be advisable to increase albumin levels in these groups through preoperative protein supplementation.

- INR is a measure of blood clotting efficiency. The present results indicate that increased INR (slower clotting) is associated
with longer LOS. Vitamin K has been shown to enhance clotting ability and lower INR, and the implication of the present analysis 
is that prescription of vitamin k preoperatively might decrease LOS, perhaps substantially, and is indicated for those with 
hight preoperative INR.

