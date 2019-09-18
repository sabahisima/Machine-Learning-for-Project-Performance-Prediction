import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.stats import skew
from sklearn.model_selection import train_test_split
# from .preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
# from sklearn.linear_model import LassoCV
# from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import cross_val_score

# pd.set_option('display.height', 1000000000)
# pd.set_option('display.max_rows', 50000000)
# pd.set_option('display.max_columns', 500000000)
# pd.set_option('display.width', 1000000000)

a = 0
b = 0
c = 0

#df_data = pd.read_csv('Complete Data 8-5-2018-Regression.csv', usecols=list(range(0, 19)))
df_data = pd.read_csv('Data-Regression.csv', usecols=list(range(0, 11)))
descriptive = df_data.describe()
descriptive.to_excel('summary.xlsx')
df_kurtosis = df_data.kurtosis()
df_kurtosis.to_excel('data-kurtosis.xlsx')
df_kurt = df_data.kurt()
df_kurt.to_excel('dara_kurt.xlsx')
df_skewness = df_data.skew()
df_skewness.to_excel('data_skew.xlsx')
#print(df_data.describe())

# print(df_data['RT'].kurt()) ##kurtosis - how skewed is your data

# print(df_data.corr()) ##correlation matrix

# df_data[['feature1', 'feature2', 'feature3', 'feature4']].corr()  ##allows you to create a correlation matrix from a subset of df_data

# df_data.categoricalfeature.value_counts() ##counts the number of records within each category for a given feature

#print(df_data.info())
y = df_data.Perf
X = df_data.drop('Perf', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.27, random_state=1)





# ax1.set_title('Before Scaling')
# sns.kdeplot(X_train['RT'], ax=ax1)
# sns.kdeplot(X_train['Reac'], ax=ax1)
# sns.kdeplot(X_train['App'], ax=ax1)
# sns.kdeplot(X_train['Soc'], ax=ax1)
# sns.kdeplot(X_train['Comp'], ax=ax1)
# sns.kdeplot(X_train['GPA'], ax=ax1)
# sns.kdeplot(X_train['Gender'], ax=ax1)
# sns.kdeplot(X_train['BIOL'], ax=ax1)
# sns.kdeplot(X_train['CHEM'], ax=ax1)
# sns.kdeplot(X_train['CM'], ax=ax1) ## for some reasons normall plot gets weird from this point
# sns.kdeplot(X_train['CST'], ax=ax1)
# sns.kdeplot(X_train['GCS'], ax=ax1)
# sns.kdeplot(X_train['AET'], ax=ax1)
# sns.kdeplot(X_train['White'], ax=ax1)
# sns.kdeplot(X_train['African-American'], ax=ax1)
# sns.kdeplot(X_train['Hispanic-Latino'], ax=ax1)
# sns.kdeplot(X_train['Multiracial'], ax=ax1)
# ax2.set_title('After MinMax Scaler')
# sns.kdeplot(X_train_sc['RT'], ax=ax2)
# sns.kdeplot(X_train_sc['Inn'], ax=ax2)
# sns.kdeplot(X_train_sc['Reac'], ax=ax2)
# sns.kdeplot(X_train_sc['App'], ax=ax2)
# sns.kdeplot(X_train_sc['Soc'], ax=ax2)
# sns.kdeplot(X_train_sc['Comp'], ax=ax2)
# sns.kdeplot(X_train_sc['GPA'], ax=ax2)
# sns.kdeplot(X_train_sc['Gender'], ax=ax2)
# sns.kdeplot(X_train_sc['BIOL'], ax=ax2)
# sns.kdeplot(X_train_sc['CHEM'], ax=ax2)
# sns.kdeplot(X_train_sc['CM'], ax=ax2)
# sns.kdeplot(X_train_sc['CST'], ax=ax2)
# sns.kdeplot(X_train_sc['GCS'], ax=ax2)
# sns.kdeplot(X_train_sc['AET'], ax=ax2)
# sns.kdeplot(X_train_sc['White'], ax=ax2)
# sns.kdeplot(X_train_sc['African-American'], ax=ax2)
# sns.kdeplot(X_train_sc['Hispanic-Latino'], ax=ax2)
# sns.kdeplot(X_train_sc['Multiracial'], ax=ax2)
# plt.show()

#print(X_train)
## ########################################lasso rigression#################################
alphas = [0.00045, 0.0001, 0.01, 0.1, 5, 10, 20,150, 200]
scores = []
for a in alphas:
 lasso = Lasso(alpha = a)
 lasso.fit(X_val, y_val)
 scores.append(lasso.score(X_val, y_val))


plt.plot(alphas, scores, '-o')
plt.xlabel('alpha, a')
plt.ylabel('lasso scores')
plt.xticks(alphas)
plt.show()
print('lasso scores are: ', str(scores))
#best results obtain when alpha is 0.1
lasso = Lasso(alpha=0.00045)
lasso.fit(X_train, y_train)
lasso_pred_test = lasso.predict(X_test)
lasso_pred_train= lasso.predict(X_train)

#lasso_scores = cross_val_score(lasso, X, y, cv=3, scoring='mean_squared_error')
#print('CV mse for lasso is:', lasso_scores)
#print(lasso.score(X_test, y_test))
print('mse test for lasso is: '+ str(mean_squared_error(y_test, lasso_pred_test)))
print('mse train for lasso with complete data is: '+ str(mean_squared_error(y_train, lasso_pred_train)))
print('R2 for lasso is ', str(r2_score(y_test, lasso_pred_test)))
#print(r2_score(y_train, lasso_pred_train))
lasso.coef_
df_lasso = pd.DataFrame(lasso.coef_)
#print(lasso_pred_test)
#print(y_test)


names = pd.DataFrame(list(X.columns))
df_lasso = pd.concat([df_lasso, names], axis=1)
df_lasso.columns = ['Lasso Coefficient', 'Feature']
print(df_lasso)
a = 0
b = 0
y_testlist = y_test.values.tolist()
lasso_pred_testlist = np.array(lasso_pred_test).tolist()
for i in range(0,len(lasso_pred_testlist)):
    a += (lasso_pred_testlist[i]-np.mean(lasso_pred_testlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(lasso_pred_testlist)-1)*(np.std(lasso_pred_testlist)*(np.std(y_testlist)))
print('correlation for lasso is: ', (a)/b)
a = 0
b = 0

#plt.plot(y_testlist, 'b')
#plt.plot(lasso_pred_testlist, 'r')
#plt.title('Lasso')
#plt.xlabel('Test samples')
#plt.ylabel('Performance value')
#plt.show()


## ########################################lasso regression CV#################################
#lasso_cv = LassoCV(cv=5, random_state=0).fit(X, y)
# print(lasso_cv.score(X, y))


## ########################################Ridge Regression#################################

alphas = [0.00045, 0.0001, 0.01, 0.1, 5, 10, 20,150, 200]
scores = []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_val, y_val)
    scores.append(ridge.score(X_val, y_val))
print('ridge scores: ', str(scores))

plt.plot(alphas, scores, '-o')
plt.xlabel('alpha, a')
plt.ylabel('ridge scores')
plt.xticks(alphas)
plt.show()
ridge = Ridge(alpha=0.00045)
ridge.fit(X_train, y_train)
ridge_preds_test = ridge.predict(X_test)
ridge_preds_train= ridge.predict(X_train)
print('r2 for ridge is', str(ridge.score(X_test, y_test)))
#print(ridge.score(X_train, y_train))
#ridge_scores = cross_val_score(ridge, X, y, cv=3, scoring='mean_squared_error')
#print('CV mse for ridge is: ', ridge_scores)
print('test mse for ridge is: '+str(mean_squared_error(y_test, ridge_preds_test)))
print('train mse for ridge is: '+str(mean_squared_error(y_train, ridge_preds_train)))
df_ridge = pd.DataFrame(ridge.coef_)
names = pd.DataFrame(list(X.columns))
df_ridge = pd.concat([df_ridge, names], axis=1)
df_ridge.columns = ['Coefficient', 'Feature']
print(df_ridge)

a = 0
b = 0
y_testlist = y_test.values.tolist()
ridge_pred_testlist = np.array(ridge_preds_test).tolist()
for i in range(0,len(ridge_pred_testlist)):
    a += (ridge_pred_testlist[i]-np.mean(ridge_pred_testlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(ridge_pred_testlist)-1)*(np.std(ridge_pred_testlist)*(np.std(y_testlist)))
print('correlation for ridge is: ', a/b)
a = 0
b = 0

plt.plot(y_testlist, 'b')
plt.plot(ridge_pred_testlist, 'r')
plt.title('Ridge')
plt.xlabel('Test samples')
plt.ylabel('Performance value')
#plt.show()

## ########################################SVR#################################
#####parameter Tuning###################
svr = SVR()

param_grid = {
    #"kernel": ['linear'],
    #"kernel": ['rbf'],
    #"kernel": ['sigmoid'],
    "kernel": ['poly'],
    "gamma": [1e-1, 1e-2, 1e-3, 1e-4],
    "C": [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
#CV_svr = GridSearchCV(estimator=svr, param_grid=param_grid, cv=10)
#CV_svr.fit(X_val, y_val)
#print(CV_svr.best_params_)

###################################
##############cs = [1e3, 1e4, 1e5]
# for c in cs:
# SVR_PT = SVR(kernel='linear', C=c).fit(X_train_sc, y_train)
# pred_Linear_PT = SVR_PT.predict(X_train_sc)
# print("score for SVR linear is: " + str(r2_score(y_train, pred_Linear_PT)))



SVR_linear=SVR(kernel='linear', C=0.1)
SVR_linear.fit(X_train,y_train)
pred_Linear=SVR_linear.predict(X_test)
print('test mse for svr linear is: '+str(mean_squared_error(y_test, pred_Linear)))
print('train mse for svr linear is: '+str(mean_squared_error(y_train, SVR_linear.predict(X_train))))
#print(SVR_linear.score(X_test,y_test))
print("score for test SVR linear is: "+ str(r2_score(y_test,pred_Linear)))
#print("score for train SVR linear is: "+ str(SVR_linear.score(X_train,y_train)))


pred_Linearlist = np.array(pred_Linear).tolist()
for i in range(0,len(pred_Linearlist)):
    a += (pred_Linearlist[i]-np.mean(pred_Linearlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(pred_Linearlist)-1)*(np.std(pred_Linearlist)*(np.std(y_testlist)))
print('correlation for svr linear is: ', a/b)
a = 0
b = 0

plt.plot(y_testlist, 'b')
plt.plot(pred_Linearlist, 'r')
plt.title('SVM-Linear')
plt.xlabel('Test samples')
plt.ylabel('Performance value')
#plt.show()
#feature importance

##################################################################
SVR_rbf=SVR(kernel='rbf', C=50, gamma=0.0001)
SVR_rbf.fit(X_train,y_train)
pred_rbf_test=SVR_rbf.predict(X_test)
pred_rbf_train=SVR_rbf.predict(X_train)
#print(SVR_rbf.score(X_test,y_test))
#print(SVR_rbf.score(X_train,y_train))
print("r2 for SVR rbf is: "+ str(r2_score(y_test,pred_rbf_test)))
#print("score for SVR rbf is: "+ str(r2_score(y_train,pred_rbf_train)))
print('test mse for svr rbf is: '+ str(mean_squared_error(y_test, pred_rbf_test)))
print('train mse for svr rbf is: '+ str(mean_squared_error(y_train, pred_rbf_train)))

pred_rbf_testlist = np.array(pred_rbf_test).tolist()
for i in range(0,len(pred_rbf_testlist)):
    a += (pred_rbf_testlist[i]-np.mean(pred_rbf_testlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(pred_rbf_testlist)-1)*(np.std(pred_rbf_testlist)*(np.std(y_testlist)))
print('correlation for svr rbf is: ', (a)/b)
a = 0
b = 0

plt.plot(y_testlist, 'b')
plt.plot(pred_rbf_testlist, 'r')
plt.title('SVM-RBF')
plt.xlabel('Test samples')
plt.ylabel('Performance value')
#plt.show()

SVR_poly=SVR(kernel='poly', C=50, gamma= 0.001)
SVR_poly.fit(X_train,y_train)
pred_poly=SVR_poly.predict(X_test)
#print(SVR_poly.score(X_test,y_test))
print("r2 for SVR poly is: "+ str(r2_score(y_test,pred_poly)))
#print("score for SVR poly is: "+ str(r2_score(y_train,SVR_poly.predict(X_train))))
print('test mse for svr poly is: ', str(mean_squared_error(y_test, pred_poly)))
print('train mse for svr poly is: ', str(mean_squared_error(y_train, SVR_poly.predict(X_train))))

pred_polylist = np.array(pred_poly).tolist()
for i in range(0,len(pred_polylist)):
    a += (pred_polylist[i]-np.mean(pred_polylist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(pred_polylist)-1)*(np.std(pred_polylist)*(np.std(y_testlist)))
print('correlation for svr poly is: ', (a)/b)
a = 0
b = 0

plt.plot(y_testlist, 'b')
plt.plot(pred_polylist, 'r')
plt.title('SVM-Poly')
plt.xlabel('Test samples')
plt.ylabel('Performance value')
#plt.show()

SVR_sigmoid=SVR(kernel='sigmoid', C=25, gamma= 0.001)
SVR_sigmoid.fit(X_train,y_train)
pred_sigmoid=SVR_sigmoid.predict(X_test)
#print(SVR_sigmoid.score(X_test,y_test))
#print(SVR_sigmoid.score(X_train,y_train))
print("r2 for sigmoid  is: "+ str(r2_score(y_test,pred_sigmoid)))
print('test mse for svr sigmoid is: ', str(mean_squared_error(y_test, pred_sigmoid)))
print('train mse for svr sigmoid is: ', str(mean_squared_error(y_train, SVR_sigmoid.predict(X_train))))

pred_sigmoidlist = np.array(pred_sigmoid).tolist()
for i in range(0,len(pred_sigmoidlist)):
    a += (pred_sigmoidlist[i]-np.mean(pred_sigmoidlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(pred_sigmoidlist)-1)*(np.std(pred_sigmoidlist)*(np.std(y_testlist)))
print('correlation for svr sigmoid is: ', (a)/b)
a = 0
b = 0

plt.plot(y_testlist, 'b')
plt.plot(pred_sigmoidlist, 'r')
plt.title('SVM-Sigmoid')
plt.xlabel('Test samples')
plt.ylabel('Performance value')
#plt.show()
## ########################################Randon forest#################################

regressor_RF = RandomForestRegressor(n_estimators=100, random_state=1, min_samples_leaf=1)
regressor_RF.fit(X_train, y_train)
pred_RF_test = regressor_RF.predict(X_test)
pred_RF_train = regressor_RF.predict(X_train)
RF_importance = pd.DataFrame(regressor_RF.feature_importances_)
names = pd.DataFrame(list(X.columns))
RF_importance = pd.concat([RF_importance, names], axis=1)
RF_importance.columns = ['importance', 'Feature']
print(RF_importance)
print('r2 for RF is',r2_score(y_test, pred_RF_test))
#print(r2_score(y_train, pred_RF_train))
print('test mse for random forest is: '+str(mean_squared_error(y_test, pred_RF_test)))
print('train mse for random forest is: '+str(mean_squared_error(y_train, pred_RF_train)))

pred_RF_testlist = np.array(pred_RF_test).tolist()
for i in range(0,len(pred_RF_testlist)):
    a += (pred_RF_testlist[i]-np.mean(pred_RF_testlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(pred_RF_testlist)-1)*(np.std(pred_RF_testlist)*(np.std(y_testlist)))
print('correlation for random forest is: ', (a)/b)
a = 0
b = 0

plt.plot(y_testlist, 'b')
plt.plot(pred_RF_testlist, 'r')
plt.title('Random forest')
plt.xlabel('Test samples')
plt.ylabel('Performance value')
#plt.show()

###parameter tuning for random forest regressor#####
sample_n_estimator = [5, 10, 25, 50, 100, 200]
sample_leaf_options = [1,5,10,50,100,200,500]
RF_R2 = []
leaf_size_index = []
estimator_index = []
for leaf_size in sample_leaf_options:
    for estimator in sample_n_estimator:
            regressor_RF_tuned = RandomForestRegressor(n_estimators = estimator, oob_score = False, n_jobs = 1,random_state =1,
                                            max_features = "auto", min_samples_leaf = leaf_size)
            regressor_RF_tuned.fit(X_val, y_val)
            pred_RF_tuned=regressor_RF_tuned.predict(X_val)
            RF_R2.append(r2_score(y_val, pred_RF_tuned))
            leaf_size_index.append(leaf_size)
            estimator_index.append(estimator)

print('the highest r2 for RF with is: ', str(max(RF_R2)), 'the index is: ',
        str(RF_R2.index(max(RF_R2))), ' min_sample leaf: ', str(leaf_size_index[RF_R2.index(max(RF_R2))]), ' n_estimator: ',
        str(estimator_index[RF_R2.index(max(RF_R2))]) )
            #print('n_estimator:'+ str(estimator)+
                  #'   min_sample_leaf'+str(leaf_size)+ '   r-squared is:'+str(r2_score(y_val, pred_RF_tuned)))

########## NN #################
#parameter tuning####
hidden_layer_sizes = [1, 2, 3, 4, 5]
activation = ['relu', 'identity', 'logistic', 'tanh']
solver = ['lbfgs', 'sgd', 'adam']
alpha = [0.0001, 0.001, 0.01, 0.1]
learning_rate = ['constant', 'invscaling', 'adaptive']
nn_r2 = []
ac_index = []
s_index = []
a_index = []
l_index = []
for h in hidden_layer_sizes:
    for ac in activation:
        for s in solver:
            for a in alpha:
               for l in learning_rate:
                    nn = MLPRegressor(hidden_layer_sizes = (h,), activation = ac, max_iter = 10000, solver = s, alpha=a, learning_rate = l, random_state= 42)
                    nn.fit(X_val, y_val)
                    nn_r2.append(r2_score(y_val, nn.predict(X_val)))
                    l_index.append(l)
                    #print(nn_r2)
                    #print('r2 for nn with hidden layer size of ', str(h), ' activation of ', str(ac),
                    #  ' solver of ', str(s), ' learning rate of ', str(l), ' and alpha of ', str(a), ' is: ',
                     #   str(r2_score(y_val, nn.predict(X_val))))
                    l_index.append(l)
                    a_index.append(a)
                    s_index.append(s)
                    ac_index.append(ac)
    #print("iteration {} done".format(a))
    print('the highest r2 for nn with', str(h), ' hidden layers is', str(max(nn_r2)), 'the index is: ', str(nn_r2.index(max(nn_r2))),
          'learning rate: ', str(l_index[nn_r2.index(max(nn_r2))]), ' alpha: ',
          str(a_index[nn_r2.index(max(nn_r2))]), 'solver: ', str(s_index[nn_r2.index(max(nn_r2))]),
          'activation: ', str(ac_index[nn_r2.index(max(nn_r2))]))
    nn_r2 = []
    ac_index = []
    s_index = []
    a_index = []
    l_index = []
nn1 = MLPRegressor(hidden_layer_sizes = (1), activation='logistic', solver = 'lbfgs',
                  learning_rate = 'constant', alpha = 0.01, random_state= 1)
#scores = cross_val_score(nn, X, y, cv=5, scoring= 'mean_squared_error')
#print('SHIT:', scores)
nn1.fit(X_train, y_train)
#print("NN R2 result: ", str(nn.score(X_train, y_train)))
print('mse of nn1 for train set is: '+ str(mean_squared_error(y_train, nn1.predict(X_train))))
print('r2 for nn1 is:', str(r2_score(y_test, nn1.predict(X_test))))
print('mse of nn1 for test set is: '+ str(mean_squared_error(y_test, nn1.predict(X_test))))

a = 0
b = 0
nn_test = nn.predict(X_test)
nn_testlist = np.array(nn_test).tolist()
for i in range(0,len(nn_testlist)):
    a += (nn_testlist[i]-np.mean(nn_testlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(nn_testlist)-1)*(np.std(nn_testlist)*(np.std(y_testlist)))
print('correlation for nn1 is: ', (a)/b)
a = 0
b = 0

plt.plot(y_testlist, 'b')
plt.plot(nn_testlist, 'r')
plt.title('Neural network')
plt.xlabel('Test samples')
plt.ylabel('Performance value')
#plt.show()



nn2 = MLPRegressor(hidden_layer_sizes = (2), activation='logistic', solver = 'lbfgs',
                  learning_rate = 'invscaling', alpha = 0.1, random_state= 1)
#scores = cross_val_score(nn, X, y, cv=5, scoring= 'mean_squared_error')
#print('SHIT:', scores)
nn2.fit(X_train, y_train)
#print("NN R2 result: ", str(nn.score(X_train, y_train)))
print('mse of nn2 for train set is: '+ str(mean_squared_error(y_train, nn2.predict(X_train))))
print('r2 for nn2 is:', str(r2_score(y_test, nn2.predict(X_test))))
print('mse of nn2 for test set is: '+ str(mean_squared_error(y_test, nn2.predict(X_test))))

a = 0
b = 0
nn2_test = nn2.predict(X_test)
nn2_testlist = np.array(nn2_test).tolist()
for i in range(0,len(nn2_testlist)):
    a += (nn2_testlist[i]-np.mean(nn2_testlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(nn2_testlist)-1)*(np.std(nn2_testlist)*(np.std(y_testlist)))
print('correlation for nn2 is: ', (a)/b)
a = 0
b = 0

nn3 = MLPRegressor(hidden_layer_sizes = (3), activation='tanh', solver = 'lbfgs',
                  learning_rate = 'invscaling', alpha = 0.1, random_state= 1)
#scores = cross_val_score(nn, X, y, cv=5, scoring= 'mean_squared_error')
#print('SHIT:', scores)
nn3.fit(X_train, y_train)
#print("NN R2 result: ", str(nn.score(X_train, y_train)))
print('mse of nn3 for train set is: '+ str(mean_squared_error(y_train, nn3.predict(X_train))))
print('r2 for nn3 is:', str(r2_score(y_test, nn3.predict(X_test))))
print('mse of nn3 for test set is: '+ str(mean_squared_error(y_test, nn3.predict(X_test))))

a = 0
b = 0
nn3_test = nn3.predict(X_test)
nn3_testlist = np.array(nn3_test).tolist()
for i in range(0,len(nn3_testlist)):
    a += (nn3_testlist[i]-np.mean(nn3_testlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(nn3_testlist)-1)*(np.std(nn3_testlist)*(np.std(y_testlist)))
print('correlation for nn3 is: ', (a)/b)
a = 0
b = 0

nn4 = MLPRegressor(hidden_layer_sizes = (4), activation='tanh', solver = 'lbfgs',
                  learning_rate = 'invscaling', alpha = 0.1, random_state= 1)
#scores = cross_val_score(nn, X, y, cv=5, scoring= 'mean_squared_error')
#print('SHIT:', scores)
nn4.fit(X_train, y_train)
#print("NN R2 result: ", str(nn.score(X_train, y_train)))
print('mse of nn4 for train set is: '+ str(mean_squared_error(y_train, nn4.predict(X_train))))
print('r2 for nn4 is:', str(r2_score(y_test, nn4.predict(X_test))))
print('mse of nn4 for test set is: '+ str(mean_squared_error(y_test, nn4.predict(X_test))))

a = 0
b = 0
nn4_test = nn4.predict(X_test)
nn4_testlist = np.array(nn4_test).tolist()
for i in range(0,len(nn4_testlist)):
    a += (nn4_testlist[i]-np.mean(nn4_testlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(nn4_testlist)-1)*(np.std(nn4_testlist)*(np.std(y_testlist)))
print('correlation for nn4 is: ', (a)/b)
a = 0
b = 0

nn5 = MLPRegressor(hidden_layer_sizes = (5), activation='logistic', solver = 'lbfgs',
                  learning_rate = 'invscaling', alpha = 0.1, random_state= 1)
#scores = cross_val_score(nn, X, y, cv=5, scoring= 'mean_squared_error')
#print('SHIT:', scores)
nn5.fit(X_train, y_train)
#print("NN R2 result: ", str(nn.score(X_train, y_train)))
print('mse of nn5 for train set is: '+ str(mean_squared_error(y_train, nn5.predict(X_train))))
print('r2 for nn5 is:', str(r2_score(y_test, nn5.predict(X_test))))
print('mse of nn5 for test set is: '+ str(mean_squared_error(y_test, nn5.predict(X_test))))

a = 0
b = 0
nn5_test = nn5.predict(X_test)
nn5_testlist = np.array(nn5_test).tolist()
for i in range(0,len(nn5_testlist)):
    a += (nn5_testlist[i]-np.mean(nn5_testlist))*(y_testlist[i]-np.mean(y_testlist))
    b = (len(nn5_testlist)-1)*(np.std(nn5_testlist)*(np.std(y_testlist)))
print('correlation for nn5 is: ', (a)/b)
a = 0
b = 0