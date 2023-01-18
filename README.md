# Predicting Hospital Length of Stay
Predict length of stay following major surgery at a large metropolitan hospital

Hospital length of stay (LOS) is defined as the length of time a patient remains in the hospital following a medical procedure. Being able to accurately predict LOS is a major goal of the healthcare community as it would streamline care, and help to efficiently direct resources to patients most in need.  

In a collaboration with surgeons at a major metropolitan hospital, I used a dataset containing demographic information, general health status, lab values, and detailed procedural information
from patients who underwent major colorectal surgery to identify the features that are most predictive of LOS. This work culminated in a model that predicts LOS within 2 days for over 60% of patients. To do so, I took the following steps:

1) Cleaned the data and imputed missing values using random forest classification/regression
2) Performed exploratory data analysis/visualization, and hypothesis-testing with parametric and non-parametric methods
to identify features of interest for predictive modeling
3) Engineered features to account for multicollinearity and class imbalance of categorical features
4) Developed a negative binomial regression model using k-fold cross validation


With nearly 800 patients and almost 200 features, I began with extensive exploratory data analysis, with the goal of identifying important
features for the model.


There turned out to be an interesting difference in LOS among the 4 race groups in the study. The most obvious trend was that White patients had
a substantially longer LOS than Black/African American, Hispanic or Asian patients, as demonstrated by the rightward-shifted cumulative probability plot:

![image](https://user-images.githubusercontent.com/89553765/212777042-c0d5aedd-050d-4ca0-8540-e81786c0240b.png)

Unsurprisingly, there was a clear positive correlation between the number of complications during surgery and LOS. There appeared to be a linear
relationship here, with each additional complication increasing median LOS by ~3 days:

![image](https://user-images.githubusercontent.com/89553765/212780762-15d01331-42db-465a-8fd9-adcddd2b1fe1.png)

Of course, there was considerable interplay between many of the variables. One particularly noteworthy illustration of this is the connection
between age, functional health status, levels of albumin in the blood, and length of stay:

![image](https://user-images.githubusercontent.com/89553765/212999846-c53445c0-967f-4e19-948e-e4de3e25f870.png)


Following EDA, features were selected for a negative binomial regression model, including race, number of complications, and albumin level. Additional features were included on the basis of correlation analyses, including the Pearson's r and Kendall's tau, and as controls on variables. Ultimately, 15 features accurately reproduced the main characteristics of the data's LOS distribution.

![image](https://user-images.githubusercontent.com/89553765/213261230-2c25575f-99e5-4fd4-9461-f7f68caf1c17.png)


Conclusions from the model:

- Higher levels of albumin, a transport protein in the blood, were associated with shorter length of stay: each unit increase 
shortens LOS by 0.8 days
- Higher international normalized ratio (INR), an indication of clotting ability, was associated with longer LOS: each unit increase 
extends LOS by 1.8 days
- Pre-existing conditions were associated with longer length of stay, including:  
      -Heart failure (1.65 days longer)  
      -Chronic obstructive pulmonary disease (1.2 days longer)
      
![image](https://user-images.githubusercontent.com/89553765/212435818-4a58de21-f0ec-41e8-b165-becff6990bc4.png)
    
- The model was considerably more accurate at predicting LOS for patients with LOS less than 12 days, who make up 88% of the 
dataset. 
    The error in predicting longer LOS is due to the undersampling of long LOS times in the dataset, and would be reduced with more data.
    
![image](https://user-images.githubusercontent.com/89553765/198186332-6b6d062c-f143-4530-a0c5-44f0b34eff98.png)



Implications and future directions:

- Albumin is a protein made by the liver that has two important functions in the body. Firstly, it acts as a carrier
in the blood for many small molecules (e.g. ions, hormones). Secondly, it prevents blood from leaking out of the blood
vessels. Given the substantial effect of albumin on LOS, knowing what factors might influence albumin levels could well
improve surgical outcomes. 


 Clearly, albumin decreases with age. Furthermore, functional dependence is associated with lower albumin levels (an older age). It may 
 therefore be advisable to increase albumin levels in these groups through preoperative protein supplementation.

- INR is a measure of blood clotting efficiency. The present results indicate that increased INR (slower clotting) is associated
with longer LOS. Vitamin K has been shown to enhance clotting ability and lower INR, and the implication of the present analysis 
is that prescription of vitamin K preoperatively might decrease LOS, perhaps substantially, and is indicated for those with 
hight preoperative INR.

