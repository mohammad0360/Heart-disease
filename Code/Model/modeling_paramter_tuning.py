
# %%
from termios import VERASE
import pandas as pd
from sklearn.model_selection import GridSearchCV
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
import numpy as np

# %%
global_path = '/Users/asmabaccouche/Heart disease/Heart-disease'

for nb in range(3):
    train = pd.read_csv(global_path+'/Data/Modelling data/Train/train'+str(nb+1)+'.csv')
    test = pd.read_csv(global_path+'/Data/Modelling data/Test/test'+str(nb+1)+'.csv')

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
    def classify(classifier, param):
        grid = GridSearchCV(classifier, param_grid=param, scoring = 'accuracy', cv=10, verbose=1, n_jobs=1)
        grid.fit(X_train, y_train)
        best_classifier = grid.best_estimator_
        y_pred  =  best_classifier.predict(X_test)
        ac = accuracy_score(y_test, y_pred)
        p = precision_score(y_test, y_pred)
        r = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        #sns.heatmap(cm, annot=True)
        #plt.show()
        return ac, p, r, f1
    # %%
    classifier1 = GaussianNB()
    classifier2 = KNN()
    classifier3 = xgb.XGBClassifier()
    classifier4 = GradientBoostingClassifier()
    classifier5 = svm.SVC()
    classifier6 = RandomForestClassifier()
    classifier7 = LogisticRegression(random_state=0)
    classifier8 = DecisionTreeClassifier(random_state=0)
    # %%
    classifiers = [classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, classifier7, classifier8]
    model_names = ['Naive Bayes', 'KNN', 'XGBoost', 'Gradient Boosting', 'SVM', 'Random Forest', 'Logistic Regression', 'Decision Tree']
    params = [
    {'var_smoothing': np.logspace(0,-9, num=100)},
    {'leaf_size': list(range(1,50)), 'n_neighbors':list(range(1,30))},
    {'learning_rate': [0.01, 0.05, 0.1, 0.2], 'n_estimators': [1000],'random_state':[0], 'max_depth' : [5, 8, 15, None], 
    'min_samples_split' : [1,2,5,10], 'min_samples_leaf': [1,2,5,10], 'max_features': ['log2', 'sqrt', 'auto', 'None']},
    {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]},
    {'C': [1,10,100], 'kernel': ['linear']},
    {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
    'max_features': ['auto', 'sqrt'], 'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]},
    {'penalty' : ['l1', 'l2'], 'C' : np.logspace(-4, 4, 20), 'solver' : ['liblinear']},
    {"criterion":['gini', 'entropy'], "max_depth":range(1,10), "min_samples_split":range(1,10),"min_samples_leaf":range(1,5)}]
    col=[]
    acs=[]
    ps=[]
    rs=[]
    f1s=[]
    for i in range(len(classifiers)):
        classifier = classifiers[i]
        model_name = model_names[i]
        param = params[i]
        ac, p, r, f1 = classify(classifier, param)
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
    df_results.to_csv(global_path+'/Documents/Model reports/results'+str(nb+1)+'.csv', index=False)
# %%
