#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from patsy import dmatrix,dmatrices
import statsmodels.api as sm
from statsmodels.formula.api import glm
import missingno as msno
from MissForestExtra import MissForestExtra as mfe
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from statsmodels.discrete.discrete_model import NegativeBinomial
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from functools import reduce


# In[3]:


colorectal_1 = pd.read_excel('/Users/jnl741/Downloads/Colorectal 10.28.16_final.xlsx')


# In[4]:


colorectal_2 = pd.read_excel('/Users/jnl741/Downloads/Colon 10.29.21-6.24.22_final.xlsx')


# In[5]:


"""Find common and distinct columns."""


missing_cols_colorectal_1 = []
missing_cols_colorectal_2 = []
common_cols = []
for col in colorectal_1.columns:
    if col in colorectal_2.columns:
        common_cols.append(col)
    else:
        missing_cols_colorectal_2.append(col)
for col in colorectal_2.columns:
    if col not in common_cols:
        missing_cols_colorectal_1.append(col)
 


# In[6]:


missing_cols_colorectal_1


# In[7]:


missing_cols_colorectal_2


# In[8]:


"""Looks like these features are common to both datasets, but inconsistently named: 

'GS (<75) Home Origin Status - Support'
'Colectomy Operative Approach'
'# of Postop Vein Thrombosis Requiring Therapy'
'Case Acuity'

Let's correct those inconsistencies."""

colorectal_1 = colorectal_1.rename(columns = {'GS (<75) Home Origin Status - Support':'GS (>75) Home Origin Status - Support',
                               'Colectomy Operative Approach':'Operative Approach',
                               '# of Postop Vein Thrombosis Requiring Therapy':'# of Postop Venous Thrombosis Requiring Therapy', 
                               'Case Status': 'Case Acuity'})


# In[9]:


missing_cols_colorectal_1 = []
missing_cols_colorectal_2 = []
common_cols = []
for col in colorectal_1.columns:
    if col in colorectal_2.columns:
        common_cols.append(col)
    else:
        missing_cols_colorectal_2.append(col)
for col in colorectal_2.columns:
    if col not in common_cols:
        missing_cols_colorectal_1.append(col)


# In[10]:


"""Merge the two datasets"""
merged_colorectal = pd.concat([colorectal_1, colorectal_2],axis = 0, ignore_index = True)
merged_colorectal = merged_colorectal[common_cols]


# In[11]:


"""Check for duplicated records"""
merged_colorectal.duplicated().sum()


# In[12]:


"""Replace # in column names with 'Num'"""
merged_colorectal.columns = merged_colorectal.columns.str.replace("#","Num")


# In[13]:


pd.options.display.max_rows = 1000
merged_colorectal.isnull().sum()


# In[14]:


"""Hospital_Length_of_Stay (LOS) is the column we're interested in modeling. Let's therefore remove any rows with 
missing LOS data. """

merged_colorectal['Hospital Length of Stay'].isnull().sum()


# In[15]:


merged_colorectal = merged_colorectal[merged_colorectal['Hospital Length of Stay'].notnull()].reset_index()


# In[16]:


"""Rename lengthy column names"""

merged_colorectal = merged_colorectal.rename(columns = {"Age at Time of Surgery": 'Age',
                                            "Sepsis (SIRS/Sepsis/Septic shock) (48h)":'Sepsis',
                                            'Num of Postop Blood Transfusions (72h of surgery start time)':'Num of Postop Blood Transfusions',
                                            'Hemoglobin A1c (HbA1c)':'Hemoglobin A1c',
                                            'Preop RBC Transfusions (72h)': 'Preop RBC Transfusions'})


# In[17]:


"""Examining the distribution of hospital length of stay."""

bins = int(np.sqrt(merged_colorectal.shape[0]))

sns.displot(merged_colorectal['Hospital Length of Stay'], bins = bins)
sns.despine()
plt.title('Length of Hospital Stay')
plt.xlabel('Days')
plt.show()


# We have a distribution of counts, with a right skew, which is best fit by either a Poisson or negative binomial distribution. 

# In[25]:


"""There is a patient with 0 days LOS. Given the nature of this procedure, and the fact that only one patient
has such a record, this seems unlikely. Let's investigate this record before deciding what to do with this record."""

merged_colorectal[merged_colorectal['Hospital Length of Stay'] == 0][['Hospital Admission Date',
                                                                     'Acute Hospital Discharge Date']]


# While the dates of admission and discharge are the same, this patient is elderly (87 years old), and would almost
# certainly have stayed in the hospital for at least one day. We will remove this record from the dataset.

# In[172]:


merged_colorectal = merged_colorectal.drop(labels = 644)


# In[31]:


"""Generate simulated Poisson and negative binomial distributions based on parameters derived from our data.
The Poisson distribution has one parameter which describes both its mean and its variance. The binomial
distribution is overdispersed, such that the variance is a quadratic function of the mean and a dispersion parameter. 
Determinging which distribution best fits our data will aid our choice of model."""


# Simulated Poisson. The Poisson distribution has one parameter which describes both its mean and its variance
Poisson_sim = np.random.poisson(merged_colorectal['Hospital Length of Stay'].mean(),800)


# Simulate negative binomial distribution
mean = merged_colorectal['Hospital Length of Stay'].mean()
var = merged_colorectal['Hospital Length of Stay'].var()

n = mean**2 / (var - mean)
p = mean / var

NB_sim = np.random.negative_binomial(n,p,1000)

# plot distributions
fig, axes = plt.subplots(1,3, 
                         squeeze = False, 
                         figsize = (12,4))
sns.distplot(Poisson_sim, ax = axes[0][0])
axes[0][0].set_title('Poisson')
sns.distplot(merged_colorectal['Hospital Length of Stay'], ax = axes[0][1])
axes[0][1].set_title('Data')
sns.distplot(NB_sim, ax = axes[0][2])
axes[0][2].set_title('Negative Binomial')
sns.despine()
plt.show();


# Both the Poisson and the negative binomial distributions peak around 5, with the majority of values falling between 1 and ~15, and have a rightward skew. However, the negative binomial distribution looks to be a better overall fit to the data, capturing more completely the tail than does the Poisson distribution. One problem with both fits is that they allow 0 values, but the minimum LOS is 1 day. 

# In[32]:


"""To further substantiate our choice of model, let's examine our target variable, length of stay (LOS), for over-
dispersion. To do that, we will fit a negative binomial distribution to our data without any predictors."""

x = np.ones_like(merged_colorectal['Hospital Length of Stay'])
nb_model = sm.NegativeBinomial(merged_colorectal['Hospital Length of Stay'],x).fit()
print(nb_model.summary())


# Alpha is the overdispersion parameter, and is 0.36 in our case. While small, the p-value indicates that it is statistically significant. We will therefore conclude that our data is more appropriately fit with a negative binomial distribution than a Poisson distribution.

# In[91]:


get_ipython().run_cell_magic('capture', '', '\n"""There appear to be four types of features in the dataset:\n\n1) Demographic data: race, sex, age\n2) Information about the surgery: open vs laproscopic, etc\n3) Comorbidities, or diseases/ health conditions\n4) Complications arising from the surgery\n\nLet\'s perform some cleaning and EDA on each category."""')


# In[92]:


get_ipython().run_cell_magic('capture', '', '"""Cleaning and EDA on surgery type. \'In/Out-Patient_Status and CPT_Description\' appear to be relevant columns."""')


# In[93]:


merged_colorectal['In/Out-Patient Status'].value_counts(dropna = False)


# Due to the substantial class imbalance, this feature will likely not be useful for modeling. 

# In[94]:


"""CPT description"""

merged_colorectal['CPT Description'].value_counts(dropna = False)


# In[95]:


"""Open procedures involve making a single, large incision, and are the traditional surgical approach. Laparoscopic
surgery, in contrast, is a minimally invasive approach. It has been shown that recovery times are generally shorter
for laparoscopic procedures. 
Let's consolidate the column along an open vs. laparoscipic axis."""

merged_colorectal['Open or Laparoscopic'] =  np.where(merged_colorectal['CPT Description'].str.startswith('Laparoscopy'),
                                                     'Laparoscopic',
                                                     'Open')


# In[96]:


merged_colorectal['Open or Laparoscopic'].value_counts(dropna = False)


# In[97]:


sns.boxplot(x = 'Open or Laparoscopic',
            y = 'Hospital Length of Stay',
            data = merged_colorectal)


# In[98]:


"""Mann-Whitney U test comparison of median length of stay following open vs. laparoscopic colectomies."""

scipy.stats.mannwhitneyu(merged_colorectal[merged_colorectal['Open or Laparoscopic'] == 'Laparoscopic']['Hospital Length of Stay'],
                    merged_colorectal[merged_colorectal['Open or Laparoscopic'] == 'Open']['Hospital Length of Stay'])


# It appears as though patients undergoing laparoscopic colectomies have a statistically significant 
# shorter length of stay.

# In[99]:


"""What does the distribution of surgery duration look like? """

sns.displot(x = 'Duration of Surgical Procedure (in minutes)',
            data = merged_colorectal)
sns.despine()


# In[100]:


"""Is there an obvious relationship between duration of surgery and hospital length of stay?"""

sns.scatterplot(x = 'Duration of Surgical Procedure (in minutes)',
                 y = 'Hospital Length of Stay',
                 data = merged_colorectal)
sns.despine()


# In[101]:


""" Let's quantify the correlation between duration of surgery and length of stay. Given the discrete, essentially 
ordinal nature of the length of stay variable, Kendall's tau, a rank-based test of correlation, is the appropriate 
choice."""

ktau = merged_colorectal['Duration of Surgical Procedure (in minutes)'].corr(merged_colorectal['Hospital Length of Stay'],
                                     method = 'kendall')
print(f" Kendall's tau correlation coefficient = {ktau}")


# There is not a strong association between surgery duration and length of stay in our dataset. 

# In[102]:


"""What is the breakdown of case acuity, that is, elective, urgent, or emergent, in our dataset? """
merged_colorectal['Case Acuity'].value_counts(dropna = False)


# In[103]:


merged_colorectal['Emergency Case'].value_counts(dropna = False)


# In[104]:


"""To restrict analysis to elective procedures, drop emergent cases."""

merged_colorectal[merged_colorectal['Emergency Case'].isnull()]['Elective Surgery'].value_counts(dropna = False)
merged_colorectal[merged_colorectal['Emergency Case'].isnull()]['Case Acuity'].value_counts(dropna = False)
merged_colorectal = merged_colorectal.drop(merged_colorectal[merged_colorectal['Case Acuity'] == 'Emergent'].index)

# Replace null values with 'Elective'
merged_colorectal = merged_colorectal.fillna(value = {'Case Acuity':'Elective'})
merged_colorectal['Case Acuity'].value_counts(dropna = False)


# It looks like there are 773 elective cases, 15 urgent cases, and 8 emergent cases (which we dropped). Owing to this substantial class imbalance, it is unlikely that we can make any statistically significant inferences about differences in length of stay between these groups. We will therefore exclude this variable from further analysis.

# In[105]:


"""To where were patients discharged, home, a nursing facility, etc.?"""

merged_colorectal['Hospital Discharge Destination'].value_counts(dropna = False)


# The overwhelming majority of patients, 692/776, were discharged to their home/permanent residence, with the remaining
# patients going to a smattering of other destinations. Owing to the substantial class imbalance, we will exlude this 
# variable from further analysis.

# In[106]:


""" Functional health status describes whether a patient is functionally independent, or requires assistance to perform
the daily activities of life. It is therefore a readout of overall health, and may be predictive of length of stay."""

# Fix the typo in the original column name
merged_colorectal.rename(columns = {'Functional Heath Status': 'Functional Health Status'}, inplace = True)


# In[107]:


merged_colorectal['Functional Health Status'].value_counts(dropna = False)


# In[108]:


props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
sns.boxplot(x = 'Functional Health Status', 
            y = 'Hospital Length of Stay', 
            data = merged_colorectal,color = 'white',
            **props, 
            palette = 'Set2',
           order = ['Independent','Partially Dependent','Totally Dependent','Unknown'])
sns.stripplot(x = 'Functional Health Status', 
              y = 'Hospital Length of Stay', 
              data = merged_colorectal, 
              palette = 'Set2',
              linewidth = 1,
              jitter = 0.1,
             order = ['Independent','Partially Dependent','Totally Dependent','Unknown'])
plt.xticks(rotation = 60)
sns.despine()
plt.show()


# In[109]:


#Consolidate health status into functionally independent or dependent values

merged_colorectal['Functionally Dpdnt or Ind'] = np.where(merged_colorectal['Functional Health Status'].isin(['Partially Dependent','Totally Dependent']),
                                                                                                                 'Dependent',
                                                                                                                 'Independent')


# In[110]:


merged_colorectal['Functionally Dpdnt or Ind'].value_counts(dropna = False)


# In[111]:


props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
sns.boxplot(x = 'Functionally Dpdnt or Ind', 
            y = 'Hospital Length of Stay', 
            data = merged_colorectal,color = 'white',
            **props, 
            palette = 'Set2',
            order = ['Independent','Dependent'])
sns.stripplot(x = 'Functionally Dpdnt or Ind', 
              y = 'Hospital Length of Stay', 
              data = merged_colorectal, 
              palette = 'Set2',
              linewidth = 1,
              jitter = 0.1,
              order = ['Independent','Dependent'])
plt.xticks(rotation = 60)
sns.despine()
plt.show()


# In[112]:


scipy.stats.mannwhitneyu(merged_colorectal[merged_colorectal['Functionally Dpdnt or Ind']=='Dependent']['Hospital Length of Stay'], 
                        merged_colorectal[merged_colorectal['Functionally Dpdnt or Ind']=='Independent']['Hospital Length of Stay'])


# Functional dependence is associated with a longer length of stay

# In[113]:


"""On which day of the week were surgeries performed? This has shown to be a factor in determining length of stay."""

# Get the data type and format of the 'Operation_Date' column'
merged_colorectal['Operation Date'].head()


# In[114]:


"""Create operation month and day-of-week columns"""

#Convert operation date column to datetime type
merged_colorectal['Operation Date'] = pd.to_datetime(merged_colorectal['Operation Date'])

#Extract month
merged_colorectal['Operation Month'] = merged_colorectal['Operation Date'].dt.month

# Create a dictionary to map numeric months to proper nouns (January, February, etc.)
months_dict = {i:j for i,j in enumerate(list(calendar.month_name)) if i != 0}
merged_colorectal['Operation Month'] = merged_colorectal['Operation Month'].map(months_dict)


# Repeat the above procedure for day of operation
merged_colorectal['Operation dow'] = merged_colorectal['Operation Date'].dt.dayofweek

days_dict = {i:j for i,j in enumerate(list(calendar.day_name))}
merged_colorectal['Operation dow'] = merged_colorectal['Operation dow'].map(days_dict)


# In[115]:


dow = merged_colorectal.groupby(by = 'Operation dow')['Hospital Length of Stay'].agg([np.median,np.size])
dow = dow.reindex(['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday'])
dow


# In[116]:


props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
sns.boxplot(x = 'Operation dow', 
            y = 'Hospital Length of Stay', 
            data = merged_colorectal,color = 'white',
            **props, 
            palette = 'Set2',
            order = [day for day in days_dict.values()])
sns.stripplot(x = 'Operation dow', 
              y = 'Hospital Length of Stay', 
              data = merged_colorectal, 
              palette = 'Set2',
              linewidth = 1,
              order = [day for day in days_dict.values()],
              jitter = 0.1)
plt.xticks(rotation = 60)
sns.despine()
plt.show()


# There is no systematic variation in length of stay based on the day of the week of the procedure.

# In[117]:


props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
sns.boxplot(x = 'Operation Month', 
            y = 'Hospital Length of Stay', 
            data = merged_colorectal,color = 'white',
            **props, 
            palette = 'Set2',
            order = [month for month in months_dict.values()])
sns.stripplot(x = 'Operation Month', 
              y = 'Hospital Length of Stay', 
              data = merged_colorectal, 
              palette = 'Set2',
              linewidth = 1,
              order = [month for month in months_dict.values()],
              jitter = 0.15)
plt.xticks(rotation = 60)
sns.despine()
plt.show()


# There is no systematic variation in length of stay based on the month of the operation.

# In[118]:


get_ipython().run_cell_magic('capture', '', '"""Cleaning and EDA on demographic features"""')


# In[119]:


"""Is there a difference in length of stay between male and female patients?"""

props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
sns.boxplot(x = 'Sex', 
            y = 'Hospital Length of Stay', 
            data = merged_colorectal,color = 'white',
            **props, 
            palette = 'Set2')
sns.stripplot(x = 'Sex', 
              y = 'Hospital Length of Stay', 
              data = merged_colorectal, 
              palette = 'Set2',
              linewidth = 1,
              jitter = 0.15)
plt.xticks(rotation = 60)
sns.despine()
plt.show()


# There does not appear to be a difference in length of stay between the sexes

# In[120]:


"""What is the racial makeup of the patient population?"""

merged_colorectal['Race'].value_counts(dropna = False)


# In[121]:


"""Is unknown/unreported race data present elsewhere?"""

merged_colorectal['Hispanic Ethnicity'].value_counts()


# In[122]:


merged_colorectal[(merged_colorectal['Race']=='Unknown/Not Reported') & (merged_colorectal['Hispanic Ethnicity'] == 'Yes')]


# In[123]:


get_ipython().run_cell_magic('capture', '', '\n"""There are 141 instances of patients who labeled themselves as \'unknown race\' but of hispanic \nethnicity. Let\'s treat these as a separate, \'Hispanic\', racial group."""\n\nmerged_colorectal[\'Race\'] = np.where((merged_colorectal[\'Race\'] == \'Unknown/Not Reported\') & (merged_colorectal[\'Hispanic Ethnicity\'] == \'Yes\'),\n                                     \'Hispanic\',\n                                     merged_colorectal[\'Race\'])')


# In[124]:


merged_colorectal['Race'].value_counts(dropna = False)


# In[125]:


"""Do some racial/ethnic groups have longer LOS than others? """

sns.kdeplot(data = merged_colorectal[~merged_colorectal['Race'].isin(['Unknown/Not Reported','American Indian or Alaska Native'])], 
                           x = "Hospital Length of Stay", 
                           hue = "Race",
            cumulative = True, 
            common_norm = False, 
            common_grid = True,
);


# Self-identified white patients appear to have longer LOS than other groups whereas elf-identified asian patients appear to have the shortest.

# In[126]:


"""Is there a relationship between preferred language and length of stay?"""
merged_colorectal['Preferred Language'].value_counts(dropna = False)


# There is not enough data to warrant a comparison

# In[127]:


sns.boxplot(x = 'Preferred Language',
            y = 'Hospital Length of Stay',
            data = merged_colorectal)


# In[128]:


"""What is the makeup by age of the patient population?"""

sns.histplot(x = merged_colorectal['Age'], 
            bins = int(np.sqrt(merged_colorectal.shape[0])))
sns.despine()
plt.title('Age')
plt.xlabel('Years')

plt.show()


# In[129]:


"""Is there a relationship between length of stay and age at time of surgery?"""

sns.scatterplot(x = merged_colorectal['Age'],
            y = merged_colorectal['Hospital Length of Stay'])
sns.despine()
plt.show()


# In[130]:


""" Kendall's tau calculation to quantify correlation between age and LOS."""

ktau = merged_colorectal['Age'].corr(merged_colorectal['Hospital Length of Stay'],
                                     method = 'kendall')
print(f"Kendall's tau correlation coefficient = {ktau}")


# It appears as though age is weakly positively correlated with length of stay. 

# In[131]:


"""Is there a difference in median age between racial/ethnic groups? Age may serve as a confounding factor in this case,
and add an element of multicollinearity to our eventual regression model."""

merged_colorectal.groupby(by = 'Race')['Age'].agg([np.mean,np.std,np.size])


# In[132]:


props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
sns.boxplot(x = 'Race', y = 'Age',data = merged_colorectal,color = 'white',**props, palette = 'Set2')
sns.stripplot(x = 'Race', y = 'Age',data = merged_colorectal, palette = 'Set2',linewidth = 1)
plt.xticks(rotation = 60)
sns.despine()
plt.show()


# It appears as though white patients tend to be older than patients from other race groups, with a difference in median age of 4-5 years.

# In[133]:


"""Does body mass index, or BMI, correlate with length of stay? Increased BMI, especially at the level of obesity,
is associated with poorer health outcomes, generally, and has been linked to longer length of stay."""

sns.scatterplot(x = merged_colorectal['BMI'],
            y = merged_colorectal['Hospital Length of Stay'])
sns.despine()
plt.show()

ktau = merged_colorectal['BMI'].corr(merged_colorectal['Hospital Length of Stay'],
                                     method = 'kendall')

print(f"Kendall's tau correlation coefficient = {ktau}")


# There is no obvious correlation between BMI and length of stay.

# In[134]:


"""Does insurance payor status effect length of stay?"""

merged_colorectal['Primary Payor Status'].value_counts(dropna = False)


# There are 262 missing values, but let's see if there's anything interesting in the data

# In[135]:


props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
sns.boxplot(x = 'Primary Payor Status', 
            y = 'Hospital Length of Stay',
            data = merged_colorectal,color = 'white',
            **props, 
            palette = 'Set2')
sns.stripplot(x = 'Primary Payor Status', 
              y = 'Hospital Length of Stay',
              data = merged_colorectal, 
              palette = 'Set2',
              linewidth = 1)
plt.xticks(rotation = 60)
sns.despine()
plt.show()


# There is no obvious difference in median length of stay between the various insurance classes for which sufficient data is present.There is an obviously more substantial variance in the Medicare group

# In[136]:


""" Comorbidities are severe medical conditions that negatively impact health and, potentially, length of stay.""" 

#Columns containing information related to co-morbidities.
comorb_cols = ['Ventilator Dependent',
 'History of Severe COPD',
 'Ascites', 
 'Heart Failure',
 'Sepsis',
 'Hypertension requiring medication',
 'Disseminated Cancer',
 'Immunosuppressive Therapy',
 'Malnourishment',
 'Bleeding Disorder',
 'Preop RBC Transfusions',
 'Preop COVID-19 Diagnosis', 
 'Diabetes Mellitus',
 'Current Smoker within 1 year',
 'Dyspnea']

# Display missing values
merged_colorectal[comorb_cols].isnull().sum()


# In[137]:


comorb_cols.remove('Preop COVID-19 Diagnosis')


# In[138]:


fig, axs = plt.subplots(nrows=7, ncols=2,figsize = (8,30))
props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
for col, ax in zip(comorb_cols, axs.ravel()):
    sns.stripplot(x = col, y = 'Hospital Length of Stay', 
                  data = merged_colorectal,
                  ax = ax,
                  linewidth = 1,
                  jitter = 0.1,
                  palette = 'Set2')
    sns.boxplot(x = col, y = 'Hospital Length of Stay', 
                  data = merged_colorectal,
                  ax = ax,
                  palette = 'Set2',
                  **props)
    sns.despine()


# In[139]:


"""Because 'SIRS' and 'Sepsis' subcategories of the Sepsis variable have similar distributions, let's consolidate them 
in order to make a statistically robust comparison between presence and absence."""

merged_colorectal['Has Sepsis'] = np.where(merged_colorectal['Sepsis'] == 'None',
                                          'No',
                                          'Yes')

#Update list of comorbidity columns
comorb_cols.append('Has Sepsis')


# In[140]:


fig, axs = plt.subplots(nrows=8, ncols=2,figsize = (8,30))
props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
for col, ax in zip(comorb_cols, axs.ravel()):
    sns.stripplot(x = col, y = 'Hospital Length of Stay', 
                  data = merged_colorectal,
                  ax = ax,
                  linewidth = 1,
                  jitter = 0.1,
                  palette = 'Set2')
    sns.boxplot(x = col, y = 'Hospital Length of Stay', 
                  data = merged_colorectal,
                  ax = ax,
                  palette = 'Set2',
                  **props)
    sns.despine()


# As with complications, the comorbidity categories are suffer from class imbalances. This is a problem because
# it gives the model less information to learn from, and will leave comparisons underpowered. Including these variables
# in the model could also lead to overfitting. 
# To get around this, we will calculate the Charlson index for each patient. The Charlson index is a weighted sum of
# comorbidities, and is commonly used in the medical community.

# In[141]:


charlson_cols = ['Heart Failure',
                 'GS (>75) History of Dementia or Cognitive Impairment',
                 'History of Severe COPD',
                 'Diabetes Mellitus',
                 'Disseminated Cancer']


# In[142]:


def Charlson(heart_failure, dementia, COPD, diabetes, cancer):
    score = 0
    if heart_failure == 'Yes':
        score += 1
    elif dementia == 'Yes':
        score += 1
    elif COPD == 'Yes':
        score += 1
    elif diabetes == 'Non-Insulin' or diabetes == 'Insulin':
        score += 1
    elif cancer == 'Yes':
        score += 6
    return score


# In[143]:


merged_colorectal['Charlson'] = merged_colorectal.apply(lambda row: Charlson(row['Heart Failure'],
                                                                             row['GS (>75) History of Dementia or Cognitive Impairment'],
                                                                             row['History of Severe COPD'],
                                                                             row['Diabetes Mellitus'],
                                                                             row['Disseminated Cancer']), axis = 1)


# In[144]:


merged_colorectal[['Heart Failure',
                 'GS (>75) History of Dementia or Cognitive Impairment',
                 'History of Severe COPD',
                 'Diabetes Mellitus',
                 'Disseminated Cancer',
                  'Charlson']]


# In[145]:


sns.boxplot(x = 'Charlson', y = 'Hospital Length of Stay', data = merged_colorectal)


# Charlson score does not have an effect on long of stay in our dataset. We will not include it in our model.

# In[146]:


"""Prior to surgery, blood is drawn to test critical markers of organ function and overall health. How do lab test 
values relate to length of stay?"""

# Columns containing lab test data
lab_tests = ['Serum Sodium', 'BUN','Serum Creatinine', 'Albumin', 'Total Bilirubin', 'AST/SGOT', 
             'Alkaline Phosphatase', 'WBC', 'Hematocrit', 'Hemoglobin A1c','Platelet Count', 'INR', 'PTT']


# In[147]:


merged_colorectal[lab_tests].dtypes


# In[148]:


# Check for missing values
merged_colorectal[lab_tests].isnull().sum()


# In[149]:


"""How do lab values correlate with LOS?"""

fig, axs = plt.subplots(nrows=5, ncols=3,figsize = (12,24))

for col, ax in zip(lab_tests, axs.ravel()):
    sns.scatterplot(x = col, y = 'Hospital Length of Stay', data = merged_colorectal,ax = ax)
    sns.despine()


# In[150]:


"""Determine the correlation coefficients between each lab value and length of stay."""

merged_colorectal[lab_tests].corrwith(merged_colorectal['Hospital Length of Stay'],
                                     method = 'kendall')


# Conventionally, values close to and above +-0.3 are evidence of correlation. Accordingly, albumin, hematocrit, and INR
# are the likeliest predictors of length of stay. 
# 
#     Albumin is a marker of overall health, and higher levels have been linked to shorter length of stay following surgery.
# 
#     Hematocrit is the percentage by volume of blood cells, and effects oxygenation of the tissues. 
# 
#     INR, or international normalized ratio, indicates how long it takes for blood to clot. The higher the value, the longer the clotting time. 
# 
# Given what these three values indicate, their correlation with length of stay makes sense.

# In[151]:


sns.pairplot(data = merged_colorectal[lab_tests],corner = True)


# In[152]:


lab_test_corrs = merged_colorectal[lab_tests].corr(method = 'pearson')

#create a mask to zero out cells that evaluate to 1, or True
mask = np.triu(np.ones_like(lab_test_corrs, dtype = bool))

#create a colormap, blue for negative correlations, red for positive
cmap = sns.diverging_palette(230, 20, as_cmap=True)

fig, axes = plt.subplots(figsize = (11,9))
sns.heatmap(lab_test_corrs,
            mask = mask, 
            vmax = 0.5,
            vmin = -0.5,
            center = 0, 
            square = True, 
            linewidths = 0.5, 
            cmap = cmap,
            annot = True,
            cbar_kws = {'shrink':0.5})


# In[153]:


"""Complications that arise as a result of surgery are an important factor in determining length of stay. The 
following columns relate to complications."""

complications_cols = ['Num of Postop Superficial Incisional SSI',
                     'Num of Postop Deep Incisional SSI',
                     'Num of Postop Organ/Space SSI',
                     'Num of Postop Wound Disruption', 
                     'Num of Postop Pneumonia',
                     'Num of Postop Unplanned Intubation',
                     'Num of Postop Pulmonary Embolism',
                     'Num of Postop On Ventilator > 48 hours',
                     'Num of Stroke/Cerebral Vascular Acccident (CVA)',
                     'Num of Cardiac Arrest Requiring CPR', 
                     'Num of Myocardial Infarction',
                     'Num of Postop Blood Transfusions (72h of surgery start time)',
                     'Total Blood Transfused (in units)', 
                     'Num of Postop C. diff',
                     'Num of Postop Sepsis',
                     'Num of Postop Septic Shock',
                     'Num of Postop Other Occurrences', 
                     'New Postop COVID-19 Diagnosis',
                     'Still in Hospital >30 Days',
                     'Postop Death w/in 30 days of Procedure',
                     'Num of Readmissions w/in 30 days',
                     'Num of Unplanned Readmissions',
                     'Num of Readmissions likely related to principal procedure',
                     'Num of Readmissions likely unrelated to principal procedure',
                     'Num of First Unplanned Return Procedures',
                     'Num of Second Unplanned Return Procedures']

# Remove columns which leak information, or do not relate to the original procedure.

complications_cols = ['Num of Postop Superficial Incisional SSI',
                     'Num of Postop Deep Incisional SSI',
                     'Num of Postop Organ/Space SSI',
                     'Num of Postop Wound Disruption', 
                     'Num of Postop Pneumonia',
                     'Num of Postop Unplanned Intubation',
                     'Num of Postop Pulmonary Embolism',
                     'Num of Postop On Ventilator > 48 hours',
                     'Num of Stroke/Cerebral Vascular Acccident (CVA)',
                     'Num of Cardiac Arrest Requiring CPR', 
                     'Num of Myocardial Infarction',
                     'Num of Postop Blood Transfusions',
                     'Total Blood Transfused (in units)', 
                     'Num of Postop C. diff',
                     'Num of Postop Sepsis',
                     'Num of Postop Septic Shock',
                     'Num of Postop Other Occurrences', 
                     'New Postop COVID-19 Diagnosis',
                     #'Still in Hospital >30 Days',
                     #'Postop Death w/in 30 days of Procedure',
                     #'Num of Readmissions w/in 30 days',
                     #'Num of Unplanned Readmissions',
                     #'Num of Readmissions likely related to principal procedure',
                     #'Num of Readmissions likely unrelated to principal procedure',
                     #'Num of First Unplanned Return Procedures',
                     #'Num of Second Unplanned Return Procedures'
                     ]


# In[154]:


merged_colorectal[complications_cols].isnull().sum()


# In[155]:


# Let's drop 'Total Blood Transfused (in units)' and 'New Postop COVID-19 Diagnosis' from the complications list

complications_cols = [col for col in complications_cols if col not in ['Total Blood Transfused (in units)','New Postop COVID-19 Diagnosis']]


# In[156]:


complications_cols


# In[157]:


#Plot complications vs length of stay

fig, axs = plt.subplots(nrows=6, ncols=3,figsize = (12,25))
props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
for col, ax in zip(complications_cols, axs.ravel()):
    sns.stripplot(x = col, y = 'Hospital Length of Stay', 
                  data = merged_colorectal,
                  ax = ax,
                  linewidth = 1,
                  jitter = 0.1,
                  palette = 'Set2')
    sns.boxplot(x = col, y = 'Hospital Length of Stay', 
                  data = merged_colorectal,
                  ax = ax,
                  palette = 'Set2',
                  **props)
    sns.despine()


# In[158]:


"""Owing to the large class imbalance for most complications columns, we will sum complications and look at the
aggregate effect on length of stay. This can be justified by the fact that in all cases, based on the boxplots above,
having a complication increases length of stay."""

merged_colorectal['Summed Complications'] = merged_colorectal[complications_cols].sum(axis = 1)

merged_colorectal['Summed Complications'].value_counts().sort_index()


# In[159]:


"""Bin complications into new ordinal column to look at aggregate effect of complications on length of stay."""

def bin_complications(num_complications):
    if num_complications == 0:
        Binned_complications = 'None'
    elif num_complications == 1:
        Binned_complications = 'One'
    elif num_complications == 2:
        Binned_complications = 'Two'
    elif num_complications >= 3:
        Binned_complications = 'Three_or_more'
    
    return Binned_complications


merged_colorectal['Binned Complications'] = merged_colorectal['Summed Complications'].apply(bin_complications)


# In[161]:


merged_colorectal['Binned Complications'].value_counts()


# In[162]:


props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
sns.boxplot(x = merged_colorectal['Binned Complications'], 
            y = 'Hospital Length of Stay', 
            data = merged_colorectal, 
            color = 'white',
            **props, palette = 'Set2',
            order = ['None','One','Two','Three_or_more'])
sns.stripplot(x = merged_colorectal['Binned Complications'], 
            y = 'Hospital Length of Stay',
              data = merged_colorectal, 
              palette = 'Set2',
              linewidth = 1,
              order = ['None','One','Two','Three_or_more'])
#plt.xticks(rotation = 60)
sns.despine()
plt.show()


# Length of stay increases monotonically with the number of complications. We will therefore include the 'binned_complications' variable in our model.

# In[166]:


"""Let's assess the extent of missing data in the columns of interest."""

model_cols = ['Age',
              'Race',
              'Open or Laparoscopic',
              'BMI',
              'Case Acuity',
              'Functionally Dpdnt or Ind',
              'Duration of Surgical Procedure (in minutes)']
model_cols.extend(lab_tests)
model_cols.extend(comorb_cols)
model_cols.extend(complications_cols)


# In[167]:


# Bar plot to display the amount of missing data in each column
msno.bar(merged_colorectal[model_cols])


# The following columns are missing the vast majority of their data and may reduce the accuracy of the imputation:
# 
# 'Hemoglobin A1c', 
# 'Total Blood Transfused', 
# 'New Postop COVID-19 Diagnosis',
# 'Preop COVID-19 Diagnosis'

# In[168]:


"""Heatmap to display pairwise correlations of present/missing data:
     1 indicates missing data in one column perfectly predicts the missing data in the other column of the pair
    -1 indicates missing data in one column does not at all predict the missing data in the other column of the pair"""

msno.heatmap(merged_colorectal[model_cols])


# Where high positive correlations exist they are between missing lab values and missing comorbidity/complications values, and not between such values and demographic information,i.e race and age. This indicates that there is a group of patients for whom the record is simply incomplete, and not that there is some bias in the data collection process. We will therefore assume data is missing at random, and proceed to imputation.
# 
# We will model the data first without imputation, simply dropping rows with missing values, and then proceed to modeling the imputed dataset. This approach will help us determine the validity of the imputation approach.

# In[169]:


get_ipython().run_cell_magic('capture', '', '\n"""Now that data cleaning and EDA are complete, it is time to develop a predictive model of length of stay. One \napproach is to build a generalized linear model for count data, which basically extends the logic of least squares\nregression to non-normally distributed data. Typically Poisson and negative binomial regression are utilized for \nmodeling count data. The key difference between these methods is that Poisson regression assumes that the distribution \nbeing modeled has variance equal to the mean, whereas negative binomial regression has no such assumption. Negative \nbinomial regression allows for variance to vary as a quadratic function of the mean. """')


# In[170]:


df_for_model = merged_colorectal[['Age',
                                 'Albumin',
                                 'INR',
                                 'Hematocrit',
                                 'Binned Complications',                                  
                                 'Race',
                                 'Platelet Count',
                                 'BUN',
                                 'Open or Laparoscopic',
                                 'Functionally Dpdnt or Ind',
                                 'Hospital Length of Stay',
                                 'Has Sepsis',
                                 'Immunosuppressive Therapy',
                                 'Preop RBC Transfusions',
                                 'Heart Failure',
                                 'Hypertension requiring medication',
                                 'Disseminated Cancer',
                                 'Bleeding Disorder',
                                 'Diabetes Mellitus',
                                 'Current Smoker within 1 year',
                                 'Ventilator Dependent',
                                 'Charlson',
                                 'History of Severe COPD',
                                 'Operation dow']]


# In[171]:


## Drop null values
df_for_model = df_for_model.dropna(subset = ['Age',
                                 'Albumin',
                                 'INR',
                                 'Hematocrit',
                                 'Binned Complications',                                  
                                 'Race',
                                 'Platelet Count',
                                 'BUN',
                                 'Open or Laparoscopic',
                                 'Functionally Dpdnt or Ind',
                                 'Hospital Length of Stay',
                                 'Has Sepsis',
                                 'Immunosuppressive Therapy',
                                 'Preop RBC Transfusions',
                                 'Heart Failure',
                                 'Hypertension requiring medication',
                                 'Disseminated Cancer',
                                 'Bleeding Disorder',
                                 'Diabetes Mellitus',
                                 'Current Smoker within 1 year',
                                 'Ventilator Dependent',
                                 'Charlson',
                                 'History of Severe COPD',
                                 'Operation dow'])


# In[172]:


#Format column names for Patsy
df_for_model.columns = df_for_model.columns.str.replace(' ', '_')


# In[173]:


get_ipython().run_cell_magic('capture', '', '\n"""Let\'s define a formula for the statsmodels GLM package. We will use the Patsy package to define a design matrix."""\n\n# formula, in statsmodels format, using Patsy \nexpr = """ Hospital_Length_of_Stay ~ Albumin + INR + Hematocrit + C(Race, Treatment(1)) + C(Binned_Complications) +\nC(Open_or_Laparoscopic) + C(Functionally_Dpdnt_or_Ind) + Age + C(Heart_Failure) + C(Has_Sepsis) + \nC(History_of_Severe_COPD)"""')


# In[174]:


# Fit and print the results of a glm model with the above model matrix configuration. We will first fit a Poisson
# regression model.

Poisson_reg = glm(expr,
            data = df_for_model, 
            family = sm.families.Poisson()).fit()

print(Poisson_reg.summary2())


# In[175]:


# Fit and print the results of a glm model with the above model matrix configuration

NB2_reg = glm(expr,  
            data = df_for_model,
            # Fit data with a negative binomial regression, using the log link function, and dispersion
            # parameter of 1.57, the median of the bootstrapped distribution
            family = sm.families.NegativeBinomial(alpha = 0.3636)).fit()

print(NB2_reg.summary2())


# In[176]:


"""Diagnostic plots. 
To compare the performance and suitability of the Poisson and negative binomial regression for modeling length of 
stay, let's look at two metrics:

1) Deviance is the difference in log likelihoods between the proposed and saturated models, and indicates
how good of a fit the proposed model is to the data. Plotting deviance residuals as a function of fitted values 
therefore reveals the contribution of each fitted value to the model deviance. 

2) A Pearson residual is the difference between predicted and observed value, standardized by the standard deviation
of the observed value. The presence of a large number of values beyond +-2 standard deviations therefore suggests
poor model fit."""


fig,axes = plt.subplots(2,2,figsize = (8,8))
sns.scatterplot(x = Poisson_reg.fittedvalues,y = Poisson_reg.resid_deviance,ax = axes[0][0])
axes[0][0].set_xlabel('Fitted Values')
axes[0][0].set_ylabel('Deviance Residuals')
axes[0][0].set_title('Poisson Regression Model')
axes[0][0].set_ylim(bottom = -4, top = 8)
sns.scatterplot(x = NB2_reg.fittedvalues,y = NB2_reg.resid_deviance, ax = axes[0][1])
axes[0][1].set_xlabel('Fitted Values')
axes[0][1].set_ylabel('Deviance Residuals')
axes[0][1].set_title('Negative Binomial Regression (NB2) Model')
axes[0][1].set_ylim(bottom = -4, top = 8)
sns.scatterplot(x = Poisson_reg.fittedvalues,y = Poisson_reg.resid_pearson,ax = axes[1][0])
axes[1][0].set_xlabel('Fitted Values')
axes[1][0].set_ylabel('Pearson Residuals')
axes[1][0].set_title('Poisson Regression Model')
axes[1][0].set_ylim(bottom = -4, top = 8)
sns.scatterplot(x = NB2_reg.fittedvalues,y = NB2_reg.resid_pearson, ax = axes[1][1])
axes[1][1].set_xlabel('Fitted Values')
axes[1][1].set_ylabel('Pearson Residuals')
axes[1][1].set_title('Negative Binomial Regression (NB2) Model')
axes[1][1].set_ylim(bottom = -4, top = 8)

sns.despine()

fig.tight_layout()

plt.show()


# The negative binomial regression model leads to smaller deviance and Pearson residuals than the Poisson regression model. 

# In[177]:


"""Perform a chi-square test comparing the the NB2 model with a dispersion parameter, and the Poisson model with mean
equal to the variance. The null hypothesis is that the value of the dispersion parameter should be 1, in other words,
that the negative binomial regression is not a better fit than Poisson regression."""

def likelihood_ratio_test(log_likelihood_null, log_likelihood_alt, df):

    LR_statistic = -2 * (log_likelihood_null - log_likelihood_alt)

    p_val = scipy.stats.chi2.sf(LR_statistic, df)

    print(f"p-value = {p_val}")
    return(LR_statistic, p_val)


# In[178]:


likelihood_ratio_test(Poisson_reg.llf, NB2_reg.llf, df = 1)


# Given the p-value, we should reject the null hypothesis that negative binomial regression, with its additonal dispersion parameter, does not provide a better fit to the data than does Poisson regression.
# 

# In[179]:


# plot distributions
fig, axes = plt.subplots(1,2, 
                         squeeze = False, 
                         figsize = (8,4))
sns.distplot(NB2_reg.predict(), ax = axes[0][0], color = 'darkred')
axes[0][1].set_title('Model')
sns.distplot(df_for_model['Hospital_Length_of_Stay'], ax = axes[0][1], color = 'skyblue')
axes[0][0].set_xlabel('Length of Stay (days)',fontsize = 11)
axes[0][1].set_xlabel('Length of Stay (days)', fontsize = 11)
axes[0][0].set_title('Data')
sns.despine()
plt.tight_layout()
plt.show()


# The negative binomial regression models appears to recapitulate the main features of the data, but underestimates
# the density of the tail.

# In[180]:


"""We will now develop and validate a negative binomial regression model. To validate the model, we will use K-fold
cross-validation to assess our model's ability to predict length of stay given unseen data."""

# Root mean squared error is a common metric for evaluating model performance. It is defined as the square root of the
# mean squared error, with the error being the difference between acual and predicted values. Let's define a function
# to calculate RMSE.

def RMSE(array1, array2):
    """Calculate RMSE """
    squared_error = np.sum(((array1 - array2) **2))
    RMSE = np.sqrt(squared_error / len(array1))
    return RMSE


# In[181]:


# Reset the index

df_for_model = df_for_model.reset_index()


# In[182]:


"""Let's perform k-fold cross-validation on both the Poisson and negative binomial regression models. This will allow
us to directly assess the accuracy of each model when making predictions of length of stay based on unseen data."""


# Create kfold column and intiate it with a dummy value
df_for_model["kfold"] = -1



# create k-fold instance with 10 splits on shuffled dataset
kf = KFold(n_splits = 10, shuffle = True)
# for loop to create the k-folds
for fold, (train_indicies, test_indicies) in enumerate(kf.split(X = df_for_model)):
    df_for_model.loc[test_indicies, "kfold"] = fold


NB2_summaries = {}
NB2_test_actual = []
NB2_test_predictions = []
NB2_training_predictions = []
NB2_test_RMSEs = []
NB2_train_RMSEs = []

Poisson_summaries = {}
Poisson_test_predictions = []
Poisson_training_predictions = []
Poisson_test_RMSEs = []
Poisson_train_RMSEs = []

for i in range(10):
    y_train, x_train = dmatrices(expr, 
                                df_for_model[df_for_model['kfold'] != i],
                                return_type = 'dataframe')
    y_test, x_test = dmatrices(expr, 
                               df_for_model[df_for_model['kfold'] == i], 
                               return_type = 'dataframe')
        
    x_train, x_test = x_train.align(x_test, join = 'outer', axis = 1, fill_value = 0)
    
    # Fit negative binomial regression model
    NB2_model = sm.GLM(y_train, 
                       x_train, 
                       family = sm.families.NegativeBinomial(alpha = 0.36)).fit() 
    NB2_summaries[i] = NB2_model.summary2()
    
    #Generate predictions on test set
    NB2_test_predicted = NB2_model.get_prediction(x_test)
    NB2_test_predictions.append(NB2_test_predicted.summary_frame()['mean'])
    
    #Generate predictions on train set
    NB2_training_predicted = NB2_model.get_prediction(x_train)
    NB2_training_predictions.append(NB2_training_predicted)
    
    #Calculate RMSE for test fold
    NB2_test_RMSE = RMSE(NB2_test_predicted.summary_frame()['mean'],
                         y_test['Hospital_Length_of_Stay'])
    NB2_test_RMSEs.append(NB2_test_RMSE)    

    #Calculate RMSE for training folds
    NB2_train_RMSE = RMSE(NB2_training_predicted.summary_frame()['mean'],
                          y_train['Hospital_Length_of_Stay'])
    NB2_train_RMSEs.append(NB2_train_RMSE)
    
    # Append predictions actual LOS values
    NB2_test_actual.append(y_test)
    
    
    
    
    # Fit Poisson regression model
    Poisson_model = sm.GLM(y_train, 
                           x_train, 
                           family = sm.families.Poisson()).fit()
     
    Poisson_summaries[i] = Poisson_model.summary2()
    
    #Generate predictions on test folds
    Poisson_test_predicted = Poisson_model.get_prediction(x_test)
    Poisson_test_predictions.append(Poisson_test_predicted)
   
    #Generate predictions on train folds
    Poisson_training_predicted = Poisson_model.get_prediction(x_train)
    Poisson_training_predictions.append(Poisson_training_predicted)
    
    #Calculate RMSE for test fold
    Poisson_test_RMSE = RMSE(Poisson_test_predicted.summary_frame()['mean'],
                             y_test['Hospital_Length_of_Stay'])
    Poisson_test_RMSEs.append(Poisson_test_RMSE)    
    
    #Calculate RMSE for training folds
    Poisson_train_RMSE = RMSE(Poisson_training_predicted.summary_frame()['mean'],
                              y_train['Hospital_Length_of_Stay'])
    Poisson_train_RMSEs.append(Poisson_train_RMSE)


# In[183]:


"""Assemble dataframe of predictions and actual LOS values for plotting"""

NB2_preds = reduce(lambda x,y: pd.concat([x,y]),NB2_test_predictions)
NB2_actual = reduce(lambda x,y: pd.concat([x,y]),NB2_test_actual)
NB2_preds_actual = pd.concat([NB2_preds, NB2_actual], axis = 1)
NB2_preds_actual


# In[184]:


NB2_preds_actual['RMSE'] = 1
for i in NB2_preds_actual['Hospital_Length_of_Stay'].unique():
    NB2_preds_actual['RMSE'] = np.where(NB2_preds_actual['Hospital_Length_of_Stay'] == i,
                                        RMSE(NB2_preds_actual[NB2_preds_actual['Hospital_Length_of_Stay'] ==i]['Hospital_Length_of_Stay'],
                                             NB2_preds_actual[NB2_preds_actual['Hospital_Length_of_Stay'] ==i]['mean']),
                                        NB2_preds_actual['RMSE'])


# In[185]:


NB2_preds_actual.head()


# In[186]:


fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (8,5))
sns.barplot(x = NB2_preds_actual['Hospital_Length_of_Stay'].astype(int),
           y = 'RMSE',
           data = NB2_preds_actual,
           color = 'skyblue')
#axs.set_xticks([i for i in range(1,int(merged_colorectal['Hospital Length of Stay'].max()))])
axs.set_xlabel('LOS (days)',fontdict = {'fontsize': 12}, fontweight = 'bold')
axs.set_ylabel('Prediction error (RMSE)',fontdict = {'fontsize': 12}, fontweight = 'bold', color = 'skyblue')
axs.axhline(y = 6.1, linestyle = '--', color = 'grey', linewidth = 1)
sns.despine()
axs.set_title('Prediction Error as Function of Length of Stay',
          fontdict = {'fontsize':15})
axs.text(x = 34, y = 5.5, s = 'Error using median LOS', color = 'grey', fontdict = {'fontsize': 12})
plt.xticks(rotation = 45)

#Create overlapping set of axes with shared x axis
ax2 = axs.twinx()

#plot counts of data LOS
sns.barplot(x = NB2_preds_actual['Hospital_Length_of_Stay'].astype(int),
            y = NB2_preds_actual['Hospital_Length_of_Stay'].astype(int),
            estimator = lambda x: len(x) / len(merged_colorectal) * 100,
            ax = ax2,
            alpha = 0.5,
            color = 'darkred')
ax2.set_ylabel('Data (% counts)', fontweight = 'bold',fontdict = {'fontsize': 12}, color = 'darkred')

plt.show();


# In[234]:


"""Calculate the mean test and training RMSEs for both models"""

#Negative binomial regression model 
NB2_mean_test_RMSE = (np.sum(NB2_test_RMSEs) / len(NB2_test_RMSEs))
NB2_mean_train_RMSE = (np.sum(NB2_train_RMSEs)/len(NB2_train_RMSEs))

print(f" The average RMSE for NB2 model predictions on test folds = {NB2_mean_test_RMSE}")
print(f" The average RMSE for NB2 model predictions on training folds  = {NB2_mean_train_RMSE}")


#Poisson regression model
Poisson_mean_test_RMSE = (np.sum(Poisson_test_RMSEs) / len(Poisson_test_RMSEs))
Poisson_mean_train_RMSE = (np.sum(Poisson_train_RMSEs)/len(Poisson_train_RMSEs))

print(f" The average RMSE for Poisson model predictions on test folds = {Poisson_mean_test_RMSE}")
print(f" The average RMSE for Poisson model predictions on training folds  = {Poisson_mean_train_RMSE}")


# In[260]:


"""Let's plot the train and test RMSEs against one another. This plot will tell us about the variance among both
sets of RMSE, and allow us to diagnose systematic under- or overfitting."""

fig, axes = plt.subplots(1,2, figsize = (10,5), squeeze = False)
sns.scatterplot(x = NB2_test_RMSEs, y = NB2_train_RMSEs, ax = axes[0][0])
x = np.linspace(1,10,num = 300)
axes[0][0].plot(x,x)
axes[0][0].set_xlabel("Test RMSEs")
axes[0][0].set_ylabel("Train RMSEs")
axes[0][0].set_title('Negative Binomial Regression')
sns.scatterplot(x = Poisson_test_RMSEs, y = Poisson_train_RMSEs, ax = axes[0][1])
x = np.linspace(1,10,num = 300)
axes[0][1].plot(x,x)
axes[0][1].set_xlabel("Test RMSEs")
axes[0][1].set_ylabel("Train RMSEs")
axes[0][1].set_title('Poisson Regression')
sns.despine()

plt.show()


# Systematic under/overfitting is not a problem for either model

# In[261]:


"""Does the model perform consistently across the range of LOS values? The distribution of LOS is heavily skewed,
with relatively few counts in each count bin of the tail compared to count bins around the median. This
asymmetry may lead to comparatively poorer predictions for larger LOS values in the training and test sets."""

# Separate the actual values of the last test fold into below and above-median values
below_median_data = y_test[y_test['Hospital_Length_of_Stay'] < merged_colorectal['Hospital Length of Stay'].median()]
above_median_data = y_test[y_test['Hospital_Length_of_Stay'] > merged_colorectal['Hospital Length of Stay'].median()]


NB2_rmse_below_median = RMSE(NB2_test_predicted.summary_frame().loc[below_median_data.index]['mean'],
                         below_median_data['Hospital_Length_of_Stay'])

NB2_rmse_above_median = RMSE(above_median_data['Hospital_Length_of_Stay'], 
                         NB2_test_predicted.summary_frame().loc[above_median_data.index]['mean'])


print(f" RMSE for below-median LOS with NB2 regression = {NB2_rmse_below_median}")
print(f" RMSE for above-median LOS with NB2 regression = {NB2_rmse_above_median}")

# Mean standard error for below and above median predictions from Poisson model
Poisson_below_median_SE = np.mean(Poisson_test_predicted.summary_frame().loc[below_median_data.index]['mean_se'])
Poisson_above_median_SE = np.mean(Poisson_test_predicted.summary_frame().loc[above_median_data.index]['mean_se'])

# Mean standard error for below and above median predictions from NB2 model
NB2_below_median_SE = np.mean(NB2_test_predicted.summary_frame().loc[below_median_data.index]['mean_se'])
NB2_above_median_SE = np.mean(NB2_test_predicted.summary_frame().loc[above_median_data.index]['mean_se'])

print(f" Average standard error of prediction for below-median LOS with Poisson regression = {Poisson_below_median_SE}")
print(f" Average standard error of prediction for above-median LOS with Poisson regression = {Poisson_above_median_SE}")

print(f" Average standard error of prediction for below-median LOS with NB2 regression = {NB2_below_median_SE}")
print(f" Average standard error of prediction for above-median LOS with NB2 regression = {NB2_above_median_SE}")


# The model performs significantly more poorly when predicting length of stay values greater than the median, as indicated by both the larger RMSE and standard error for above-median predictions.

# In[262]:


"""Does the model provide better predictions than the median value of LOS?"""
Median_LOS = merged_colorectal['Hospital Length of Stay'].median()

RMSE_using_median = RMSE(np.ones_like(y_train['Hospital_Length_of_Stay']) * Median_LOS,
                         y_train['Hospital_Length_of_Stay'])

print(f" RMSE using median  = {RMSE_using_median}")


# Our negative binomial regression model provides substantially more accurate predictions of length of stay than does the median LOS, with a difference in RMSE of >1.5 days.

# In[263]:


"""Let's run the model with all missing values imputed. To do so, we will use the MissForest imputation algorithm from
the missingpy package. MissForest uses the random forest algorithm to regress/classify each column with the other
columns in the dataset. """

# 
merged_colorectal_imputed = merged_colorectal[model_cols].copy()
merged_colorectal_imputed.drop(columns = ['Sepsis'], inplace = True)


# In[264]:


merged_colorectal_imputed.columns


# In[265]:


# Drop all duplicated columns

merged_colorectal_imputed = merged_colorectal_imputed.loc[:, ~merged_colorectal_imputed.columns.duplicated()].copy()


# In[266]:


# Transform categorical variables into dummy variables

merged_colorectal_dummies = pd.get_dummies(data = merged_colorectal_imputed,
                                           columns = merged_colorectal_imputed.select_dtypes('object').columns)


# In[267]:


merged_colorectal_dummies.shape


# In[268]:


#Add null values from original columns to dummified dataframe

for col in merged_colorectal_imputed.select_dtypes('object').columns:
    merged_colorectal_dummies.loc[merged_colorectal_imputed[col].isnull(), 
                              merged_colorectal_dummies.columns.str.startswith(col + '_')] = np.nan

merged_colorectal_dummies


# In[298]:


# Apply MissForest imputer on dataset

imputer = MissForest()
imputed_colorectal = imputer.fit_transform(merged_colorectal_dummies)


# In[299]:


# Convert imputed array into pandas dataframe

imputed_colorectal = pd.DataFrame(imputed_colorectal,
                                  columns = merged_colorectal_dummies.columns)


# In[300]:


"""In order to use the imputed dataframe with Patsy and Statsmodels, we need to restore the dataframe to the original
categorical columns. This function does that."""

def undo_dummies(df, prefix_sep = "_"):
    # create dictionary based on column names with key  = the string to left of '_' separator : value = boolean of
    # whether the separator is present or not
    cols2collapse = {item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns}
    
    #initiate list to store undummified series
    series_list = []
    
    # loop through dictionary and filter df by columns with '_' separator
    for col, needs_to_collapse in cols2collapse.items():
        #if column has '_' separator (is dummified)
        if needs_to_collapse:
            #create series 
            undummified = (
                # select all columns with column string present
                df.filter(like = col)
                # use idxmax to find index of largest value (which should be 1)
                .idxmax(axis = 1)
                # split column name on '_' separator, and use the string(s) to the right as variable names
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis = 1)
    return undummified_df


# In[301]:


# Reconstruct categorical variables in imputed dataframe
imputed_colorectal = undo_dummies(imputed_colorectal)

imputed_colorectal.shape


# In[302]:


imputed_colorectal.isnull().sum()


# In[273]:


"""Let's compare the distributions of the original data to the imputed data."""

imputed_colorectal.columns = imputed_colorectal.columns.str.replace('_', ' ')

fig,axs = plt.subplots(nrows = 7, ncols = 2, figsize = (8,24))
for col, ax in zip(lab_tests, axs.ravel()):
    sns.kdeplot(x = col, 
                  data = merged_colorectal,
                  ax = ax,
                label = 'Original')
    sns.kdeplot(x = col, 
                  data = imputed_colorectal,
                  ax = ax,
                label = 'Imputed')
    ax.legend()
sns.despine()
sns.set_context(context = None)
plt.show()


# In[274]:


# Concatenate imputed dataframe and LOS column

imputed_colorectal = pd.concat([imputed_colorectal, 
                                merged_colorectal['Hospital Length of Stay']],
                                axis = 1)


# In[275]:


# Bin complications

imputed_colorectal['Summed Complications'] = np.round(imputed_colorectal[complications_cols].sum(axis = 1))
imputed_colorectal['Binned Complications'] = imputed_colorectal['Summed Complications'].apply(bin_complications)


# In[276]:


props = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
sns.boxplot(x = imputed_colorectal['Binned Complications'], 
            y = 'Hospital Length of Stay', 
            data = merged_colorectal, 
            color = 'white',
            **props, palette = 'Set2',
            order = ['None','One','Two','Three_or_more'])
sns.stripplot(x = imputed_colorectal['Binned Complications'], 
            y = 'Hospital Length of Stay',
              data = merged_colorectal, 
              palette = 'Set2',
              linewidth = 1,
              order = ['None','One','Two','Three_or_more'])
#plt.xticks(rotation = 60)
sns.despine()
plt.show()


# In[277]:


# Replace spaces in column names with '_' for compatibility with Patsy
imputed_colorectal.columns = imputed_colorectal.columns.str.replace(' ', '_')


# In[278]:


"""Let's fit a negative binomial regression model to the imputed dataset."""

# formula, in statsmodels format, using Patsy 
expr = """ Hospital_Length_of_Stay ~ Albumin + INR  + Hematocrit + C(Race, Treatment(1)) + C(Binned_Complications) +
C(Open_or_Laparoscopic) + C(Functionally_Indpdnt_or_Dpdnt) + Age + C(Heart_Failure)  + C(History_of_Severe_COPD) + 
C(Has_Sepsis) """


NB2_reg = glm(expr,  
            data = imputed_colorectal,
            # Fit data with a negative binomial regression, using the log link function, and dispersion
            # parameter of 0.36
            family = sm.families.NegativeBinomial(alpha = 0.36)).fit()

print(NB2_reg.summary2())


# In[279]:


df_for_model = imputed_colorectal


# In[280]:


# Creating kfold column and intiate it with a dummy value
df_for_model["kfold"] = -1

# for loop to create the k-folds
kf = KFold(n_splits = 10, shuffle = True)
for fold, (train_indicies, test_indicies) in enumerate(kf.split(X = df_for_model)):
    df_for_model.loc[test_indicies, "kfold"] = fold

NB2_summaries = {}
NB2_test_predictions = []
NB2_training_predictions = []
NB2_test_RMSEs = []
NB2_train_RMSEs = []

Poisson_summaries = {}
Poisson_test_predictions = []
Poisson_training_predictions = []
Poisson_test_RMSEs = []
Poisson_train_RMSEs = []

for i in range(10):
    y_train, x_train = dmatrices(expr, 
                                df_for_model[df_for_model['kfold'] != i],
                                return_type = 'dataframe')
    y_test, x_test = dmatrices(expr, 
                               df_for_model[df_for_model['kfold'] == i], 
                               return_type = 'dataframe')
        
    x_train, x_test = x_train.align(x_test, join = 'outer', axis = 1, fill_value = 0)
    
    # Fit negative binomial regression model
    NB2_model = sm.GLM(y_train, 
                       x_train, 
                       family = sm.families.NegativeBinomial(alpha = 0.3636)).fit() 
    NB2_summaries[i] = NB2_model.summary2()
    
    #Generate predictions on test set
    NB2_test_predicted = NB2_model.get_prediction(x_test)
    NB2_test_predictions.append(NB2_test_predicted)
    
    #Generate predictions on train set
    NB2_training_predicted = NB2_model.get_prediction(x_train)
    NB2_training_predictions.append(NB2_training_predicted)
    
    #Calculate RMSE for test fold
    NB2_test_RMSE = RMSE(NB2_test_predicted.summary_frame()['mean'],
                         y_test['Hospital_Length_of_Stay'])
    NB2_test_RMSEs.append(NB2_test_RMSE)    

    #Calculate RMSE for training folds
    NB2_train_RMSE = RMSE(NB2_training_predicted.summary_frame()['mean'],
                          y_train['Hospital_Length_of_Stay'])
    NB2_train_RMSEs.append(NB2_train_RMSE)


# In[281]:


#Negative binomial regression model 
NB2_mean_test_RMSE = (np.sum(NB2_test_RMSEs) / len(NB2_test_RMSEs))
NB2_mean_train_RMSE = (np.sum(NB2_train_RMSEs)/len(NB2_train_RMSEs))

print(f" The average RMSE for NB2 model predictions on test folds = {NB2_mean_test_RMSE}")
print(f" The average RMSE for NB2 model predictions on training folds  = {NB2_mean_train_RMSE}")


# Data imputation with MissingForest allowed use of data from an additional 100 patients, and reduced RMSE on the test
# folds.

# In[282]:


# Define function to output plot of the model coefficients

def coefplot(results, coef_labels = False, vars_to_drop = False):
    '''
    Takes in results of regression model and returns a plot of 
    the coefficients with 95% confidence intervals.
    
    Removes intercept, so if uncentered will return error.
    '''
    # Create dataframe of results summary 
    coef_df = pd.DataFrame(results.summary().tables[1].data)
    
    # Add column names
    coef_df.columns = coef_df.iloc[0]

    # Drop the extra row with column labels
    coef_df = coef_df.drop(0)

    # Set index to variable names 
    coef_df = coef_df.set_index(coef_df.columns[0])
    if coef_labels:
        coef_df = coef_df.rename(index = coef_labels)
    if vars_to_drop:
        coef_df = coef_df.drop(vars_to_drop, axis = 0)
    
    # Change datatype from object to float
    coef_df = coef_df.astype(float)

    # Get errors; (coef - lower bound of conf interval)
    errors = coef_df['coef'] - coef_df['[0.025']
    
    # Append errors column to dataframe
    coef_df['errors'] = errors

    # Drop the constant for plotting
    coef_df = coef_df.drop(['Intercept'])

    # Sort values by coef ascending
    coef_df = coef_df.sort_values(by=['coef'])

    ### Plot Coefficients ###

    # x-labels
    variables = list(coef_df.index.values)
    
    # Add variables column to dataframe
    coef_df['variables'] = variables
    
    # Set sns plot style back to 'poster'
    # This will make bars wide on plot
    sns.set_context("poster")

    # Define figure, axes, and plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Error bars for 95% confidence interval
    # Can increase capsize to add whiskers
    coef_df.plot(x  ='variables', y = 'coef', kind = 'bar',
                 ax = ax, color='none', fontsize = 12, 
                 ecolor = 'steelblue',capsize = 0,
                 yerr = 'errors', legend = False)
    
    # Set title & labels
    plt.title('Feature Coefficients w/ 95% Confidence Intervals',fontsize=15)
    ax.set_ylabel('Coefficients',fontsize=12)
    ax.set_xlabel('',fontsize = 22)
    
    # Coefficients
    ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
               marker='o', s=80, 
               y=coef_df['coef'], color='black')
    
    # Line to define zero on the y-axis
    ax.axhline(y = 0, linestyle = '--', color = 'red', linewidth = 1)
    
    
    return plt.show()


# In[153]:


new_vars = {'C(Functionally_Indpdnt_or_Dpdnt)[T.Independent]':'Functionally dpdnt or indpdnt',
                             'C(Race, Treatment(1))[T.Black or African American]':'Black/African American',
                             'C(Race, Treatment(1))[T.White]':'White',
                             'C(Binned_Complications)[T.One]': '1 complication',
                             'C(Binned_Complications)[T.Two]':'2 complications',
                             'C(Binned_Complications)[T.Three_or_more]':'3 complications',
                             'C(Open_or_Laparoscopic)[T.Open]':'Open surgery',
                             'C(Heart_Failure)[T.Yes]':'History of heart failure',
                             'C(Has_Sepsis)[T.Yes]':'Post-op sepsis',
                             'C(History_of_Severe_COPD)[T.Yes]':'History of COPD',
                             'C(Race, Treatment(1))[T.American Indian or Alaska Native]':'American Indian or Alaskan Native',
                             'C(Race, Treatment(1))[T.Hispanic]':'Hispanic'}


coefplot(results = NB2_reg, coef_labels = new_vars, 
         vars_to_drop = ['American Indian or Alaskan Native',
                        'C(Race, Treatment(1))[T.Unknown/Not Reported]']
        )


# In[188]:


"""Function to remove outliers for plotting, etc"""
def find_outliers(df, col, num_std_dev):
    threshold = np.mean(df[col]) + num_std_dev * df[col].std()
    
    return df[df[col] < threshold]                                    


# In[189]:


albumin_no_outliers = find_outliers(merged_colorectal, 'Albumin',3)


# In[191]:


"""Explore relationship between albumin, age and functional health status. Given the role of Albumin
in this """

fig, axs = plt.subplots()
sns.scatterplot(y = albumin_no_outliers['Albumin'],
               x = albumin_no_outliers['Age'],
               hue = albumin_no_outliers['Hospital Length of Stay'],
               size = albumin_no_outliers['Functionally Dpdnt or Ind'],
               ax = axs)
axs.set_xlabel('Age', fontweight = 'bold')
axs.set_ylabel('Albumin', fontweight = 'bold')
axs.legend(bbox_to_anchor = (1,1))
axs.set_title('Albumin and LOS/Health', fontweight = 'bold')
sns.despine()


# In[ ]:




