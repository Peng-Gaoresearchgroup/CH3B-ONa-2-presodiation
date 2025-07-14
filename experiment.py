import joblib
import pandas as pd
features=pd.read_csv('./outputs/pca_features.csv')
k=joblib.load('./model/kmeans_model.pkl')
k.fit(data=features)
df=k.get_t_sne(data=features)
df.to_csv('./outputs/kmeans_t_sne.csv',index=False)
