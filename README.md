
# Ex.No: 13 REVOLUTIONIZING HEART DISEASE DETECTION USING MACHINE LEARNING
### DATE: 24.10.2024
### REGISTER NUMBER : 212222220049
## AIM
This project aims to leverage machine learning techniques, that implements Diverse machine learning algorithms to determine the best accuracy model to develop a model for early heart disease detection.

## ALGORITHM
```
-Start Program -Data Collection and Loading
-Data Preprocessing
-Feature Engineering
-Data Splitting
-Predictive Modeling with Diverse Algorithms
-Model Evaluation and Selection -
-Interpretability and Feature Importance
-Deploy Model in Real-World Application
-Generate Prediction and Improve Patient Outcomes
-End Program
```
### PROGRAM
```
*Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

*Importing Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

*Suppressing warnings
Important Code segments
import warnings
warnings.filterwarnings(action="ignore")
# Load Dataset
df = pd.read_csv("/content/heart_disease_data (1) (1).csv")
df.head()

*Data Overview
df.info()
df.isnull().sum()
print(f"Shape of dataset: {df.shape}")
print(df.describe())

*Data Visualization: Distribution of Heart Attacks
sns.countplot(data=df, x="target") # 'target' represents heart disease presence
plt.title("Distribution of Heart Attack")
plt.show()

*Splitting Features and Target Variable
X = df.drop("target", axis=1) # Features
y = df["target"] # Target Variable
Important Code segments

*Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

*Model Definitions
models = {
 "Logistic Regression": LogisticRegression(),
 "Decision Tree": DecisionTreeClassifier(),
 "Random Forest": RandomForestClassifier(),
 "SVM": SVC(),
 "KNN": KNeighborsClassifier(),
 "Gradient Boosting": GradientBoostingClassifier(),
 "XGBoost": xgb.XGBClassifier(),
 "AdaBoost": AdaBoostClassifier(),
 "Naive Bayes": GaussianNB(),
 "MLP Neural Network": MLPClassifier()
}
Important Code segments

*Training and Evaluating Models
for name, model in models.items():
 print(f"Training {name}...")
 model.fit(X_train, y_train) # Training the model
 y_pred = model.predict(X_test) # Making predictions
 accuracy = accuracy_score(y_test, y_pred) # Calculating accuracy
 print(f"{name} Accuracy: {np.round(accuracy, 2)}") # Displaying accuracy
 print(classification_report(y_test, y_pred)) # Displaying classification report
```
### OUTPUT 1

![Screenshot 2024-10-17 113247](https://github.com/user-attachments/assets/ea973986-5092-4af1-9d02-dd4a560d6eaa)
![Screenshot 2024-10-17 113314](https://github.com/user-attachments/assets/1564d7f2-fa11-4e1b-ba17-823ab83b5f67)
![Screenshot 2024-10-17 113343](https://github.com/user-attachments/assets/12571363-fd9d-4aff-a53e-09dcdf32b59c)

### OUTPUT 2

![Screenshot 2024-10-17 113906](https://github.com/user-attachments/assets/bf6ff96f-5780-49f1-a176-507b8dcff917)
![Screenshot 2024-10-17 113919](https://github.com/user-attachments/assets/cd19adc5-93b9-4fec-ac07-b13684d7e12f)
![Screenshot 2024-10-17 113930](https://github.com/user-attachments/assets/712b9442-2f2e-47f3-906c-138df6efd0b8)


### RESULTS 

The heart disease prediction project utilizes diverse algorithms to deliver accurate risk assessments, enabling healthcare professionals to make informed decisions and enhance early detection strategies. By integrating clinical and demographic data, it facilitates personalized patient care and improves health outcomes. Its scalable approach allows adaptation to other health conditions, promoting a proactive stance in preventive healthcare and benefiting public health.We have got two types of output in which output-1 is executed with predefined dataset and all ml algorithms provide average accuracy of 0.84% .In ouput-2 in which we get instant data provide diffrent accuracy for different algorithms in which, Naive Bayes Accuracy and Logistic Regression Accuracy has the highest accuracy of 0.91%.
Therefore, we conclude that Naive Bayes,Logistic Regression,SVM,AdaBoost,MLP Neural Network produce more accurate prdictions.

