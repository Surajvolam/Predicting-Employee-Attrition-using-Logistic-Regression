#  Read data from the dataset and describe the dataframe.
import pandas as PaN

# dataset is imported into a pandas dataframe
DtaFrme = PaN.read_csv('Employee_Attrition.csv')

# Display column names and data types.
print(DtaFrme.info())

# Display summary statistics for numerical columns
print(DtaFrme.describe())
# Clean and process missing data.
# Check for missing values
print(DtaFrme.isnull().sum())
#Research Question: Can we predict which employees are at risk of leaving based on their job-related factors and demographics?

#Hypothesis: Employees who report low job satisfaction, have poor work-life balance, and receive low performance ratings are more likely to leave the company.
# Replace missing values with the median for numerical columns
DtaFrme['TotalWorkingYears'].fillna(DtaFrme['TotalWorkingYears'].median(), inplace=True)
DtaFrme['NumCompaniesWorked'].fillna(DtaFrme['NumCompaniesWorked'].median(), inplace=True)
DtaFrme['TrainingTimesLastYear'].fillna(DtaFrme['TrainingTimesLastYear'].median(), inplace=True)

# Replace missing values with the mode for categorical columns
DtaFrme['BusinessTravel'].fillna(DtaFrme['BusinessTravel'].mode()[0], inplace=True)
DtaFrme['Department'].fillna(DtaFrme['Department'].mode()[0], inplace=True)
DtaFrme['EducationField'].fillna(DtaFrme['EducationField'].mode()[0], inplace=True)

# Drop irrelevant columns
DtaFrme.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
# Clean and process missing data.
# Check for missing values
print(DtaFrme.isnull().sum())
#Research Question: Can we predict which employees are at risk of leaving based on their job-related factors and demographics?

#Hypothesis: Employees who report low job satisfaction, have poor work-life balance, and receive low performance ratings are more likely to leave the company.
# Replace missing values with the median for numerical columns
DtaFrme['TotalWorkingYears'].fillna(DtaFrme['TotalWorkingYears'].median(), inplace=True)
DtaFrme['NumCompaniesWorked'].fillna(DtaFrme['NumCompaniesWorked'].median(), inplace=True)
DtaFrme['TrainingTimesLastYear'].fillna(DtaFrme['TrainingTimesLastYear'].median(), inplace=True)

# Replace missing values with the mode for categorical columns
DtaFrme['BusinessTravel'].fillna(DtaFrme['BusinessTravel'].mode()[0], inplace=True)
DtaFrme['Department'].fillna(DtaFrme['Department'].mode()[0], inplace=True)
DtaFrme['EducationField'].fillna(DtaFrme['EducationField'].mode()[0], inplace=True)

# Drop irrelevant columns
DtaFrme.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
import matplotlib.pyplot as PyPT
import seaborn as SbRn

# Set the style of the plots
SbRn.set(style="ticks", color_codes=True)

# Plot the distribution of age
SbRn.displot(DtaFrme['Age'], bins=10, kde=False, rug=True)

# Plot the distribution of monthly income
SbRn.displot(DtaFrme['MonthlyIncome'], bins=10, kde=False, rug=True)

# Plot the distribution of job satisfaction
SbRn.displot(DtaFrme['JobSatisfaction'], bins=10, kde=False, rug=True)

# Plot the distribution of years at company
SbRn.displot(DtaFrme['YearsAtCompany'], bins=10, kde=False, rug=True)

# Plot the distribution of total working years
SbRn.displot(DtaFrme['TotalWorkingYears'], bins=10, kde=False, rug=True)

# Plot the distribution of job role
SbRn.countplot(x='JobRole', data=DtaFrme)

# Plot the distribution of department
SbRn.countplot(x='Department', data=DtaFrme)

# Plot the distribution of gender
SbRn.countplot(x='Gender', data=DtaFrme)

# Plot the distribution of education field
SbRn.countplot(x='EducationField', data=DtaFrme)

# Plot the distribution of business travel
SbRn.countplot(x='BusinessTravel', data=DtaFrme)

# Plot the distribution of attrition
SbRn.countplot(x='Attrition', data=DtaFrme)
# Correlation matrix is plotted
Crrtn_Mtrix = DtaFrme.corr()

# Display the correlation matrix
print(Crrtn_Mtrix)

# Correlation matrix as a heatmap is plotted
SbRn.heatmap(Crrtn_Mtrix, annot=True)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the Ftures and Trgt variables
Ftures = DtaFrme.drop('Attrition', axis=1)
Trgt = DtaFrme['Attrition']

# Convert categorical variables to numerical using one-hot encoding
Ftures = PaN.get_dummies(Ftures)

# Split the dataset into training and testing sets
X_Trn, X_Tst, y_Trn, y_Tst = train_test_split(Ftures, Trgt, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
RndmFrst = RandomForestClassifier(n_estimators=100, random_state=42)
RndmFrst.fit(X_Trn, y_Trn)

# Predict the Trgt variable for the testing set
Y_Prdctn = RndmFrst.predict(X_Tst)

# Evaluate the peRndmFrstormance of the model
print('Accuracy:', accuracy_score(y_Tst, Y_Prdctn))
print('Precision:', precision_score(y_Tst, Y_Prdctn, pos_label='Yes'))
print('Recall:', recall_score(y_Tst, Y_Prdctn, pos_label='Yes'))
print('F1 Score:', f1_score(y_Tst, Y_Prdctn, pos_label='Yes'))
