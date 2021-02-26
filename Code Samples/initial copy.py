import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

final_features_normalized = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/final_features/final_features_normalized.csv')
final_features_standardized = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/final_features/final_features_standardized.csv')


ffn2 = final_features_normalized.copy()
ffs2 = final_features_standardized.copy()

ffn2.set_index('customer_id',inplace=True)
ffs2.set_index('customer_id',inplace=True)

def variance(x):
    return(ffn2[x].std())

def highlynull(x):
    return(ffn2[x].isnull().sum()/len(ffn2[x]))

def blankrecord(x):
    return(ffn3.iloc[x].isnull().sum()/len(ffn3.iloc[x]))


ffn3 = ffn2.astype('float64')
ffn3.fillna(float(0),inplace=True)

ffs3 = ffs2.astype('float64')
ffs3.fillna(float(0),inplace=True)

checkhighlynull = [x for x in ffn2.columns if highlynull(x)>0.0]
checklowvariance = [x for x in ffn2.columns if variance(x)<0.05]
checkblankrecords = [x for x in range(0,len(ffn3)) if blankrecord(x) > 0.5]

#
ffn3.iloc[4].isnull().sum()

#generic kmeans 
from sklearn.cluster import KMeans
kmn = KMeans(n_clusters=4)
kmn.fit(ffn3)

predictions = kmn.predict(ffn3)
preddf = pd.DataFrame(predictions,columns=['prediction'])
# preddf.prediction.value_counts()
centers = kmn.cluster_centers_
plt.scatter(ffn3.iloc[:, 0], ffn3.iloc[:, 1], c=predictions, s=2, cmap='viridis')
plt.xlim(-.02,.02)
plt.ylim(-.02,.05)
plt.title('K Means Clustering with 4 Clusters') 
plt.scatter(centers[:, 0], centers[:, 1], s=50, alpha=0.5,c='black')
plt.show()


plt.scatter()


#elbow method, seeking optimal number of clusters using inertia
from sklearn import metrics
from scipy.spatial.distance import cdist

distortions = []
inertias = []
sils = []
mapping1 = {}
mapping2 = {}
K = range(2,31)
X = ffn3

for k in K: 
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k).fit(X) 
    labels = kmeanModel.labels_
    kmeanModel.fit(X)    
      
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / X.shape[0]) 
    inertias.append(kmeanModel.inertia_) 

    sils.append(metrics.silhouette_score(X,labels,metric='euclidean'))
  
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / X.shape[0] 
    mapping2[k] = kmeanModel.inertia_ 

for key,val in mapping1.items(): 
    print(str(key)+' : '+str(val)) 

len(distortions)
plt.plot(K, distortions, 'bx-')
plt.xlim(2,31)
plt.xticks(range(2,31))
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show()

plt.plot(K, inertias, 'bx-')
plt.xlim(2,31)
plt.xticks(range(2,31)) 
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()

plt.plot(K, sils, 'bx-')
plt.xlim(2,10)
plt.xticks(range(2,10))
plt.xlabel('Values of K') 
plt.ylabel('Silhouette') 
plt.title('The Elbow Method using Silhouette') 
plt.show()

#the silhoutte chart shows the optimal cluster is 4 (highest positive silhouette value wins)
#checking the distributions amongst k=[1,5]



#tsne
from sklearn.manifold import TSNE

tsne = TSNE(learning_rate=200)

transformed = tsne.fit_transform(pdf)

x_axis = transformed[:, 0]
y_axis = transformed[:, 1]

plt.scatter(x_axis, y_axis)
plt.show()


#spectral clustering
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=4, affinity='nearest_neighbors',assign_labels='kmeans')
labels = model.fit_predict(ffn3)
plt.scatter(ffn3[:, 0], ffn3[:, 1], c=labels,s=50, cmap='viridis')


#Attempting PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
principalComponents = pca.fit_transform(ffs3)

pca_names = [f'pc{x}' for x in range(0,50)]
principalDf = pd.DataFrame(data = principalComponents,columns=pca_names)

pdf = principalDf.copy()

plt.plot(pca.explained_variance_ratio_.cumsum())

pdf2 = principalDf[principalDf.columns[0:5]]
pdf2['cluster'] = predictions
pdf2['anomaly'] = isoforestdf.anomaly

pdf2.head()

import seaborn as sns
sns.pairplot(pdf2,hue='anomaly',palette='seismic',diag_kind='hist')


from sklearn.ensemble import IsolationForest

isof = IsolationForest(max_samples=12000,n_estimators=205)
isof.fit(ffn3)

customers = list(ffn3.index.values)

isoforestdf = pd.DataFrame(customers,columns=['customer_id'])
isoforestdf['scores'] = isof.decision_function(ffn3)
isoforestdf['anomaly'] = isof.predict(ffn3)

isoforestdf.anomaly.value_counts()

anomaly_ids = [isoforestdf.iloc[x].customer_id for x in isoforestdf.index if isoforestdf.iloc[x].anomaly == -1]

anomalydf = ffn3.loc[anomaly_ids]

#determining percentage of applicants that are expected to be fraudulent
leases = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Data/CSC/performance/leases_05_12_20.csv')
customers = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Data/CSC/performance/customers_05_12_20.csv')
lease_payments = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Data/CSC/performance/lease_payments_05_12_20.csv')

bcycle = pd.DataFrame(lease_payments.groupby(['lease_id']).sum().billing_cycle)

#these are the customers who did not make it past their initial payment
bc2 = bcycle[bcycle.billing_cycle ==1]
len(bc2)/len(bcycle)
#portion of assumed fraudulent applicants is .2% and the IForest found .7% on its first pass

#using random forest to predict the anomaly values from the isolation forest, then determining top features

ffn3['anomaly'] = list(isoforestdf.anomaly)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc

rfor = RandomForestClassifier()

X = ffn3.drop(['anomaly'],axis=1)
y = ffn3['anomaly']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

rf = RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train,y_train)

predictions = rf.predict(X_test)
pred_probs = rf.predict_proba(X_test)

confusion_matrix(y_test,predictions)
roc_value = roc_auc_score(predictions,y_test)
roc_curve(y_test,predictions)

preds = pred_probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

