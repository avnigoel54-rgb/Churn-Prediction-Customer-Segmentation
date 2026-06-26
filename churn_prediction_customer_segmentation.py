
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel("Telco_customer_churn.xlsx")
df

df.shape
df.info()

df['Churn Label'].value_counts()

plt.figure(figsize=(8,5))
sns.histplot(df['Tenure Months'],kde=True,bins=30)
plt.xlabel('Tenure Months')
plt.ylabel('Frequency')
plt.title('Tenure Months Distribution')
plt.show()

print(df['Tenure Months'].max(), df['Tenure Months'].min())

plt.figure(figsize=(8,5))
sns.boxplot(data=df,y='Tenure Months',x='Churn Label')
plt.xlabel('Churn Label')
plt.ylabel('Tenure Months')
plt.title('Tenure Months Distribution vs churn')
plt.show()

df['Churn Label'].unique()

df[df['Churn Label']=='Yes']['Tenure Months'].mean()

plt.figure(figsize=(8,5))
sns.histplot(df['Monthly Charges'],kde=True,bins=30)
plt.xlabel('Monthly charges')
plt.ylabel('frequency')
plt.title('Monthly charges Distribution')
plt.show()

print(df['Monthly Charges'].max(), df['Monthly Charges'].min())

plt.figure(figsize=(8,5))
sns.boxplot(data=df,y='Monthly Charges',x='Churn Label')
plt.xlabel('Churn Label')
plt.ylabel('Monthly Charges')
plt.title('Monthly charges Distribution vs churn')
plt.show()

df[df['Churn Label']=='Yes']['Monthly Charges'].mean()

df[df['Churn Label']=='Yes']['Monthly Charges'].quantile([0.25,0.5,0.75])

df['Monthly Charges'].describe()

df['Contract'].unique()

plt.figure(figsize=(8,5))
sns.countplot(data=df,x='Contract',hue='Churn Label')
plt.xlabel('Contract')
plt.ylabel('count')
plt.title('contract Distribution')
plt.show()

df.info()

df['Internet Service'].unique()

plt.figure(figsize=(8,5))
sns.countplot(data=df,x='Internet Service',hue='Churn Label')
plt.xlabel('Internet Service')
plt.ylabel('count')
plt.title('Internet Service Distribution')
plt.show()

df['Payment Method'].unique()

plt.figure(figsize=(10,6))
sns.countplot(data=df,x='Payment Method',hue='Churn Label')
plt.xlabel('Payment Method')
plt.ylabel('count')
plt.title('Payment Method Distribution')
plt.show()

df['Tech Support'].unique()

plt.figure(figsize=(8,5))
sns.countplot(data=df,x='Tech Support',hue='Churn Label')
plt.xlabel('Tech Support')
plt.ylabel('count')
plt.title('Tech Support Distribution')
plt.show()

avg_tenure=df.groupby('Churn Label')['Tenure Months'].mean()
avg_tenure

df.info()

numeric_cols=['Tenure Months','Monthly Charges','Churn Value','Churn Score','CLTV']
corr_mx=df[numeric_cols].corr()
corr_mx

pd.crosstab(df['Contract'],df['Churn Label'],normalize='index')

"""DATA CLEANING"""

df['Total Charges']

df['Total Charges']=pd.to_numeric(df['Total Charges'],errors='coerce')
df['Total Charges']

df['Total Charges'].dtype



df['Total Charges'].isnull().sum()

df[df['Total Charges'].isnull()]['Tenure Months']

df['Total Charges']=df['Total Charges'].fillna(0)

drop_columns=['CustomerID','Count','Country','State','Zip Code','Lat Long','Latitude','Longitude','Churn Label','Churn Score','CLTV','Churn Reason']
df=df.drop(columns=drop_columns)

df.shape

df_encoded=pd.get_dummies(df,drop_first=True)
df_encoded.head()

df_encoded.shape

df=df.drop(columns=['City'])
df

df.info()

df_encoded=pd.get_dummies(df,drop_first=True)
df_encoded.shape

df_encoded

X=df_encoded.drop(columns=['Churn Value'])
y=df_encoded['Churn Value']
X.shape

y.shape

X

y

"""MACHINE LEARNING IMPLEMENTATION"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

from sklearn.ensemble import RandomForestClassifier

rf_model=RandomForestClassifier(n_estimators=100,random_state=42)

rf_model.fit(X_train,y_train)

y_pred=rf_model.predict(X_test)

y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion_mx=confusion_matrix(y_test,y_pred)
confusion_mx

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

"""approach 1 - handle class imbalance
(acc to data 1869 were churn reason ie only 1869 leaves and 5000 apx stays thats wy model predicts that more people dont leave and recall is less. this is imbalance)
so now we shift more focus towards leaving so that balance out.
"""

rf_model_balanced=RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')

rf_model_balanced.fit(X_train,y_train)

y_pred_balanced=rf_model_balanced.predict(X_test)
y_pred_balanced

accuracy_balanced=accuracy_score(y_test,y_pred_balanced)
accuracy_balanced

confusion_mx_balanced=confusion_matrix(y_test,y_pred_balanced)
confusion_mx_balanced

print(classification_report(y_test,y_pred_balanced))

"""approach 2 - hyperparameter tuning
change n_estimator & max_depth parameters in random forest
"""

rf_model_balanced_tuned=RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced',max_depth=10)
rf_model_balanced_tuned.fit(X_train,y_train)

y_pred_tuned=rf_model_balanced_tuned.predict(X_test)
y_pred_tuned

print(classification_report(y_test,y_pred_tuned))

rf_model_balanced_tuned2=RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced',max_depth=10)
rf_model_balanced_tuned2.fit(X_train,y_train)

y_pred_tuned2=rf_model_balanced_tuned2.predict(X_test)
y_pred_tuned2

print(classification_report(y_test,y_pred_tuned2))

"""combination of n_estimators and max_depth"""

n_estimators=[100,200,300,400,500]
from sklearn.metrics import recall_score,f1_score
max_depth=[5,7,10,15,20]
result=[]
for n in n_estimators:
    for m in max_depth:
        rf_model_balance_tune=RandomForestClassifier(n_estimators=n,random_state=42,class_weight='balanced',max_depth=m)
        rf_model_balance_tune.fit(X_train,y_train)
        y_pred_tune=rf_model_balance_tune.predict(X_test)
        accuracy_tune=accuracy_score(y_test,y_pred_tune)
        recall_tune=recall_score(y_test,y_pred_tune)
        f1_tune=f1_score(y_test,y_pred_tune)
        result.append({'Trees':n, 'Depth': m,'Accuracy': accuracy_tune, 'recall':recall_tune,'f1':f1_tune})
result_df=pd.DataFrame(result)
result_df=result_df.sort_values(by=['recall','Accuracy'],ascending=False)
result_df

"""approach 3 - feature importance analysis"""

feature_imp = pd.DataFrame({
    'Features': X.columns,
    'Importance' : rf_model_balanced_tuned.feature_importances_
})
feature_imp = feature_imp.sort_values(by='Importance',ascending=False)
feature_imp

X_selected=X.drop(columns=['Phone Service_Yes','Multiple Lines_No phone service'])
X_selected.shape

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.20, random_state=42)

rf_model_balanced_tuned_sel=RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced',max_depth=10)
rf_model_balanced_tuned_sel.fit(X_train_sel,y_train_sel)
y_pred_tuned_sel=rf_model_balanced_tuned_sel.predict(X_test_sel)
y_pred_tuned_sel

print(classification_report(y_test_sel,y_pred_tuned_sel))

"""Cross validation
ie shuffle data for train & test and form combinations and then find their respective accuracy & recall and then their mean
done to check if overall split is right or not
"""

from sklearn.model_selection import cross_val_score
final_rf=RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced',max_depth=10)
cv_accuracy=cross_val_score(final_rf,X,y,cv=5,scoring='accuracy')
print(cv_accuracy,cv_accuracy.mean())
cv_recall=cross_val_score(final_rf,X,y,cv=5,scoring='recall')
print(cv_recall,cv_recall.mean())

"""Roc auc curve"""


from sklearn.metrics import roc_curve,roc_auc_score
y_probability=rf_model_balanced_tuned.predict_proba(X)
print(y_probability)
y_prob=rf_model_balanced_tuned.predict_proba(X_test)

churn_probability=y_probability[:,1]
churn_prob=y_prob[:,1]
fpr,tpr,threshold=roc_curve(y_test,churn_prob)
auc_score=roc_auc_score(y_test,churn_prob)
auc_score

"""CUSTOMER SEGMENTATION"""

segmen_df=pd.DataFrame({'Tenure months':X['Tenure Months'], 'Monthly charges':X['Monthly Charges'], 'total charges':X['Total Charges'],'Churn probability':churn_probability})
segmen_df

"""implementation of k-means"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaled_data=scaler.fit_transform(segmen_df)
scaled_data[:5]

from sklearn.cluster import KMeans

wcss=[]
for k in range(1,16):
  kmeans=KMeans(n_clusters=k, random_state=42)
  kmeans.fit(scaled_data)
  wcss.append(kmeans.inertia_)
  plt.figure(figsize=(8,6))
plt.plot(range(1,16),wcss,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow method')
plt.show()


kmeans=KMeans(n_clusters=3, random_state=42)

clusters=kmeans.fit_predict(scaled_data)

segmen_df['Clusters']=clusters
segmen_df

segmen_df.groupby('Clusters').mean()

clusters_summary={
    0:'Budget loyal customers',
    1:'High risk new customers',
    2:'Loyal premium customers'
}

segmen_df['Cluster Segment']=segmen_df['Clusters'].map(clusters_summary)
segmen_df

sns.scatterplot(data=segmen_df,x='Tenure months',y='Churn probability',hue='Cluster Segment')

sns.scatterplot(data=segmen_df,x='Monthly charges',y='Churn probability',hue='Cluster Segment')

sns.scatterplot(data=segmen_df,x='total charges',y='Churn probability',hue='Cluster Segment')