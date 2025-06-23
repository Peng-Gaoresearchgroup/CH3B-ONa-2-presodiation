import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")

import yaml
from utils import SafeDict,smiles2descirptors,get_representative_mol,get_sctter_info_for_origin,get_rank_data,rank
from model import hc, kmeans,pca,Pareto
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
#-----------------------------------------------------
# Load conf
#-----------------------------------------------------
with open("./conf/conf.yaml") as f:
    conf = SafeDict(yaml.safe_load(f))

if conf.rank==False:
    #-----------------------------------------------------
    # Get original desritors
    #-----------------------------------------------------
    if conf.calculate_features == True:
        print('Start')
        df=pd.read_csv('./data/data.csv')
        if conf.load_model == False:
            descriptors= df['canonicalsmiles'].apply(lambda x: smiles2descirptors(smiles=x,desdic=conf.hc,seed=conf.seed))
            joblib.dump(descriptors,'./model/descriptors.pkl')
        else:
            descriptors=joblib.load('./model/descriptors.pkl')

        descriptors.dropna(how='any', inplace=True)
        descriptors=pd.DataFrame(descriptors)
        descriptors=descriptors.apply(lambda x: round(x, 8))
        descriptors.to_csv('./outputs/data_descriptors.csv',index=False)
    else:
        df=pd.read_csv('./data/data.csv')
        descriptors=pd.read_csv('./outputs/data_descriptors.csv')
        
    #-----------------------------------------------------
    # Get nromalized desritors
    #-----------------------------------------------------
    scaler = MinMaxScaler()
    normal_des = pd.DataFrame(scaler.fit_transform(descriptors), columns=descriptors.columns)
    # normal_des.apply(lambda x: round(x, 8))
    normal_des.to_csv('./outputs/data_normalized_descriptors.csv',index=False)
    #print(normal_des)
    # data = pd.concat([df, normal_des], axis=1)

    #-----------------------------------------------------
    # PCA dimensionality reduction for descritors
    #-----------------------------------------------------
    pca_ = pca.PCA(n_components=conf.PCA.feature_number)
    pca2_ =pca.PCA(n_components=2)
    features = pca_.fit_transform(X=normal_des)
    # features.apply(lambda x: round(x, 8))
    #print(features)
    features.to_csv('./outputs/pca_features.csv',index=False)
    features_2d=pca2_.fit_transform(X=normal_des)
    # print(features[0:5])
    # print(pca_.components)
    # print(pca_.explained_variance)

    #-----------------------------------------------------
    # Clustering
    #-----------------------------------------------------
    if conf.load_model == False:
        hc_=hc.hierarchical_clustering(distance=conf.hc.distance,partition_line=conf.hc.partition_line)
        joblib.dump(hc_,'./model/hc_model.pkl')
    else:
        hc_=joblib.load('./model/hc_model.pkl')
    hc_._fit(X=features)
    hc_.plot_dendrogram(
        dpi=conf.hc.dendro.dpi,
        figsize=tuple((conf.hc.dendro.fig_size_x,conf.hc.dendro.fig_size_y)),
        fontname=conf.hc.dendro.font,
        fontsize=conf.hc.dendro.font_size,
        treelw=conf.hc.dendro.treelw,
        borderlw=conf.hc.dendro.borderlw,
        save=conf.hc.dendro.save)
    mapping = hc_.group_mapping(save=conf.hc.save_infor)

    if conf.load_model == False:
        k=kmeans.kmeans(n_clusters=conf.kmeans.n_cluster,max_iter=conf.kmeans.max_iter)
        joblib.dump(k,'./model/kmeans_model.pkl')
    else:
        k=joblib.load('./model/kmeans_model.pkl')
    k.fit(data=features)
    fit_info=k.get_fit_info(data=features,save=conf.kmeans.fit_info_save)
    represent=get_representative_mol(fit_info,save=conf.kmeans.represent_save)
    k.plot_scatters(data=features_2d,save=conf.kmeans.plot_save)
    info=k.get_plot_information(data=features_2d,save=conf.kmeans.plot_info_save)
    result=get_sctter_info_for_origin(info,save=conf.kmeans.plot_info_save)

#-----------------------------------------------------
# Rank
#-----------------------------------------------------
if conf.rank== True:
    rank_df=pd.read_csv('./data/recommend_clusters.csv')
    # rank_absolute=get_rank_data(rank_df,'./data/data.csv')
    # rank_absolute.to_csv('./outputs/rank_absolute.csv',index=False)

    rank_absolute=pd.read_csv('./outputs/rank_absolute.csv')
    p=Pareto.Pareto(df=rank_absolute)
    front=p.pareto_front()
    # rank_mormal=rank(rank_absolute)
    front.to_csv('./outputs/rank_pareto_front.csv',index=False)
