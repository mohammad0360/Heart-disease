
# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# %%
train = pd.read_csv('/Users/asmabaccouche/Heart disease project/Data/Modelling data/Train/train3.csv')
test = pd.read_csv('/Users/asmabaccouche/Heart disease project/Data/Modelling data/Test/test3.csv')

features = train.columns.tolist()
features.remove('cardio')

X_train = train[features]
y_train = train[['cardio']]

X_test = test[features]
y_test = test[['cardio']]
# %%
sc = StandardScaler()
columns_to_scale=['age_years','height','weight','ap_hi','ap_lo']
cols = [col for col in train.columns.tolist() if col in columns_to_scale]
X_train[columns_to_scale]=sc.fit_transform(X_train[cols])
X_test[columns_to_scale]=sc.fit_transform(X_test[cols])
# %%
columns_to_encode=['cholesterol', 'gluc', 'alco', 'active']
cols = [col for col in train.columns.tolist() if col in columns_to_encode]
X_train = pd.get_dummies(X_train, columns = cols)
X_test = pd.get_dummies(X_test, columns = cols)
# %%
def classify(classifier):
    classifier.fit(X_train, y_train)
    y_pred  =  classifier.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    p = precision_score(y_test, y_pred)
    r = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True)
    plt.show()
    return ac, p, r, f1
# %%
classifier1 = GaussianNB()
classifier2 = KNN(n_neighbors = 30)
classifier3 = xgb.XGBClassifier(objective="binary:logistic")
classifier4 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.7, max_features=4, max_depth=4, random_state=0)
classifier5 = svm.SVC(kernel="linear", C=0.01)
classifier6 = RandomForestClassifier(max_depth=2, random_state=0)
classifier7 = LogisticRegression(random_state=0)
classifier8 = DecisionTreeClassifier(random_state=0, max_depth=2)
# %%
classifiers = [classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, classifier7, classifier8]
model_names = ['Naive Bayes', 'KNN', 'XGBoost', 'Gradient Boosting', 'SVM', 'Random Forest', 'Logistic Regression', 'Decision Tree']
col=[]
acs=[]
ps=[]
rs=[]
f1s=[]
for i in range(len(classifiers)):
    classifier = classifiers[i]
    model_name = model_names[i]
    ac, p, r, f1 = classify(classifier)
    col.append(model_name)
    acs.append(ac)
    ps.append(p)
    rs.append(r)
    f1s.append(f1)

df_results=pd.DataFrame()
df_results['model']=col
df_results['accuracy']=acs
df_results['precision']=ps
df_results['recall']=rs
df_results['f1-score']=f1s

df_results.sort_values('accuracy').reset_index(drop=True,inplace=True)
df_results.to_csv('/Users/asmabaccouche/Heart disease project/Documents/Model reports/results_3.csv', index=False)
# %%
