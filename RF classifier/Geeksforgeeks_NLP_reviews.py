#%%
# import numpy as 

import pandas as pd

# read in dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')
print(dataset.shape)

# libraries to clean data
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# data preprocessing
corpus = []
for i in range(0, 1000):

	review = re.sub('[^a-zA-Z]', ' ', dataset['Review'].iloc[i])
	review = review.lower()
	review = review.split()

	ps = PorterStemmer()
	review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]

	# convert back to string
	review = " ".join(review)
	corpus.append(review)

# NLP CountVecotrizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# RF training model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators = 501, random_state=137, 
	                           n_jobs=-1, criterion = 'entropy')

# base model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc_base = roc_auc_score(y_test, y_pred)
probs_base = model.predict_proba(X_test)
fpr_base, tpr_base, thresholds = roc_curve(y_test,probs_base[:, 1])
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(f'base model: {cm}')
print(f"auc score: {acc_base:.4f}")

# Create the random grid
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [False, True]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# random search training
rf = RandomForestClassifier(random_state=137)
rf_random = RandomizedSearchCV(estimator=rf, 
	                           param_distributions=random_grid,
	                           n_iter=100, cv=3, verbose=2, 
	                           random_state=42, n_jobs=-1)

rf_random.fit(X_train, y_train)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('grid searched best parmeters:\n')
best_random = rf_random.best_estimator_
print(best_random)

# use best estimater to estimate performance
random_accuracy = best_random.predict(X_test)
cm = confusion_matrix(y_test, random_accuracy)

acc_random = roc_auc_score(y_test, random_accuracy)
probs_search = best_random.predict_proba(X_test)
fpr_random, tpr_random, thresholds = roc_curve(y_test,probs_search[:, 1])
print('results from best parameters:\n')
print(f'base model: {cm}')
print(f"auc score: {acc_base:.4f}")


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 10, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [3],
    'min_samples_split': [8, 10],
    'n_estimators': [200]
}
# Create a based model
rf = RandomForestClassifier(random_state=137)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
	                       cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_test, y_test)
print('best grid parameter with CV: \n')
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
grid_accuracy = best_grid.predict(X_test)
cm = confusion_matrix(y_test, grid_accuracy)

acc_grid = roc_auc_score(y_test, grid_accuracy)
probs_grid = best_grid.predict_proba(X_test)
fpr_grid, tpr_grid, thresholds = roc_curve(y_test,probs_grid[:, 1])
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(f'base model: {cm}')
print(f"auc score: {acc_grid:.4f}")

#AOC curve display
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], '--')
ax.plot(fpr_base, tpr_base, 'c+', label='base')
ax.plot(fpr_random, tpr_random, 'r*', label='random')
ax.plot(fpr_grid, tpr_grid, 'b--', label='grid with CV')
ax.text(0.7, 0.5, "Base AUC: " + str(round(acc_base,2)))
ax.text(0.7, 0.45, "Random AUC: " + str(round(acc_random,2)))
ax.text(0.7, 0.4, "Grid AUC: " + str(round(acc_grid,2)))
plt.legend()
plt.show('hold')

#%%
tree = best_grid.estimators_[0]

#%%
from sklearn.tree.export import export_text
r = export_text(tree, feature_name=)

#%%
