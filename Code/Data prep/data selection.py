
#%%
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

#%%
# Load data
global_path = '/Users/asmabaccouche/Heart disease/Heart-disease'
data_path = global_path+'/Data/Preprocessed data/'
df = pd.read_csv(data_path+'data_clean3.csv')
# %%
# Feature Selection with Univariate Statistical Tests
features = df.columns.tolist()
features.remove('cardio')

X = df[features].values
Y = df['cardio'].values
# Feature Extraction with RFE
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
ranking = fit.ranking_.tolist()
features_ranking = [(features[i], ranking[i]) for i in range(0, len(features))]
features_ranking.sort(key = lambda x: x[1])

best_features1 = [k for (k, v) in features_ranking if v==1]

# %%
# Feature Importance with Extra Trees Classifier
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
importance = model.feature_importances_
features_importance = [(features[i], importance[i]) for i in range(0, len(features))]
features_importance.sort(key = lambda x: x[1], reverse=True)

best_features2 = [k for (k, v) in features_importance if v>0.01]

# %%
# Removing correlated features
def correlation(dataset, threshold):
    col_corr = set()  
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]                  
                col_corr.add(colname)
    return col_corr      

corr_features = correlation(df[features], 0.7)

best_features3 = [f for f in features if f not in corr_features]
# %%
best_features = [f for f in features if f in best_features3 and f in best_features1+best_features2]
data = df[best_features+['cardio']]
data.to_csv(global_path+'/Data/Preprocessed data/data_selected.csv', index=False)

train, test= train_test_split(data, test_size=0.2, random_state=42)
train.to_csv(global_path+'/Data/Modelling data/Train/train3.csv', index=False)
test.to_csv(global_path+'/Data/Modelling data/Test/test3.csv', index=False)

# train, test= train_test_split(df, test_size=0.2, random_state=42)
# train.to_csv(global_path+'/Data/Modelling data/Train/train3_wo.csv', index=False)
# test.to_csv(global_path+'/Data/Modelling data/Test/test3_wo.csv', index=False)
# %%
