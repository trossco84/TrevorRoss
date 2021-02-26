import pandas as pd 
import numpy as np

#Reducing Highly Correlated Variables, Threshold set to 92.5%

#reducing correlation in the numerical features, using the normalized as the base set
num_norm = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/cleaned_normalized_numerical_features.csv')
num_norm.set_index('customer_id',inplace=True)

num_norm_corr = num_norm.corr().abs()

upper_tri = num_norm_corr.where(np.triu(np.ones(num_norm_corr.shape),k=1).astype(np.bool))

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.925)]

num_norm_reduced = num_norm.drop(to_drop,axis=1)


#reducing correlation in standardized numerical
num_stand = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/cleaned_standardized_numerical_features.csv')
num_stand.set_index('customer_id',inplace=True)

num_stand_corr = num_stand.corr().abs()

upper_tri2 = num_stand_corr.where(np.triu(np.ones(num_stand_corr.shape),k=1).astype(np.bool))

to_drop2 = [column for column in upper_tri2.columns if any(upper_tri2[column] > 0.925)]

num_stand_reduced = num_stand.drop(to_drop2,axis=1)

#reducing correlation in the categorical variables
cat_feats = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/cleaned_categorical_features.csv')
cat_feats.set_index('customer_id',inplace=True)

cf2 = pd.get_dummies(cat_feats,drop_first=True)

cf2_corr = cf2.corr().abs()

upper_tri3 = cf2_corr.where(np.triu(np.ones(cf2_corr.shape),k=1).astype(np.bool))

to_drop3 = [column for column in upper_tri3.columns if any(upper_tri3[column] > 0.925)]

cf2_reduced = cf2.drop(to_drop3,axis=1)

#importing binary features so as to combine all feature sets
bin_feats = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/cleaned_binary_features.csv')
bin_feats.set_index('customer_id',inplace=True)

#creating final sets of features
joindfs = [cf2_reduced,bin_feats]
final_features_normalized = num_norm_reduced.join(joindfs)
final_features_standardized = num_stand_reduced.join(joindfs)

final_features_normalized.reset_index(inplace=True)
final_features_standardized.reset_index(inplace=True)

final_features_normalized.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/final_features/final_features_normalized.csv',index=False)
final_features_standardized.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/final_features/final_features_standardized.csv',index=False)
