import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")
import yaml
import utils
from model import hc, kmeans,pca,Pareto
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib,random,argparse,sys
import numpy as np

def load_conf():
    with open("./conf/conf.yaml") as f:
        conf = utils.SafeDict(yaml.safe_load(f))
    return conf

def wash_data():
    import subprocess
    subprocess.run("python wash.py", shell=True, check=True)

def calculate_descriptors(conf,data):
    df=data
    # df=df[]
    descriptors= df['canonicalsmiles'].apply(lambda x: utils.smiles2descirptors(smiles=x,desdic=conf.hc,seed=conf.seed))
    descriptors.dropna(how='any', inplace=True)
    descriptors=pd.DataFrame(descriptors)
    descriptors=descriptors.apply(lambda x: round(x, 8))
    # descriptors.to_csv('./outputs/data_descriptors.csv',index=False)
    return descriptors

def normalized_descriptors(descriptors):
    scaler = MinMaxScaler()
    normal_des = pd.DataFrame(scaler.fit_transform(descriptors), columns=descriptors.columns)
    normal_des.apply(lambda x: round(x, 8))
    return normal_des

def get_partition_k_from_hc(hc_model,range:list):
    partition_line=[]
    ks=[]
    for i in np.arange(range[0], range[1], range[2]):
        hc_model.partition_line=i
        hc_model._fit()
        k=hc_model.get_n_cluster()
        partition_line.append(i)
        ks.append(k)
    return pd.DataFrame({'partition_line':partition_line,'k':ks})

def get_best_k_by_ch(km:kmeans.kmeans,partition_k):
    def f(k):
        km.n_clusters=k
        km.fit()
        return km.get_dunn_index(),km.get_ch_score(),km.get_silhouette_scores(),km.get_davies_bouldin_index(),km.get_xie_beni_index()
    partition_k[['dunn_index', 'ch_score', 'silhouette_scores','davies_bouldin_index','xie_beni_index']] = partition_k['k'].apply(lambda k: pd.Series(f(k)))
    return partition_k

def get_demension_explained_variance(data,pca_model):
    demension=[]
    accumalative_explained_variance_ratio=[]
    for i in range(1,data.shape[1]):
        pca_model.n_components=i
        pca_model.fit()
        demension.append(i)
        accumalative_explained_variance_ratio.append(pca_model.accumalative_explained_variance_ratio)
    return pd.DataFrame({'demension':demension,'accumalative_explained_variance_ratio':accumalative_explained_variance_ratio})

def choose_recommend_mols(df,lower_limit,upper_limit):
    # represent mols
    recommed_repre=df[(df['AnodeLimit(V)']>=float(lower_limit)) & (df['AnodeLimit(V)']<=float(upper_limit))]['Reactant'].to_list()
    def find_mol(str):
        import re
        match=re.match(r'^mol(\d+)_',str)
        return int(match.group(1))
    recommed_repre=[find_mol(i) for i in recommed_repre]

    # represent mols -> corresponding culsters
    df=pd.read_csv('./outputs/kmeans_represent.csv')
    recommed_clusters=df[df['Molecule'].isin(recommed_repre)]['Cluster'].to_list()

    # clusters -> recommded mols
    df=pd.read_csv('./outputs/kmeans_info.csv')
    recommed_mols=df[df['Cluster'].isin(recommed_clusters)]
    df=pd.read_csv('./data/data.csv')
    recommed_mols['canonicalsmiles']=recommed_mols['Molecule'].apply(lambda x: df[df['idx']==x]['canonicalsmiles'].values[0])    
    return recommed_mols

def rank_after_pareto(front,conf):
    pareto_dft=pd.read_csv(conf.pareto_dft_path)
    front['anode_limit']=front['Molecule'].apply(lambda x: pareto_dft[pareto_dft['Reactant']==f'mol{x}_']['AnodeLimit(V)'].values[0])
    front['normal_anode_limit']=front['anode_limit'].apply(lambda x: 1 if 2.8<=x<=3.8 else 0)
    front[['normal_scscore','normal_spacial_score','normal_capacity']]=front[['scscore','spacial_score','capacity']].apply(lambda x: (x-x.min())/(x.max()-x.min()))
    front['total_score']=-front['normal_scscore']-front['normal_spacial_score']+front['normal_capacity']+front['normal_anode_limit']
    front=front.sort_values(by='total_score',ascending=False)
    rank=front[['Molecule','Cluster','canonicalsmiles','scscore','spacial_score','capacity','anode_limit','normal_scscore','normal_spacial_score','normal_capacity','normal_anode_limit','total_score']]
    return rank

def quick_test():
    conf=load_conf()
    print('Loading dataset...')
    recommend_mols=pd.read_csv('./data/data.csv')
    print(f'Dataset_size:{len(recommend_mols)}')
    print(f'Dataset_snapshot:\n{recommend_mols.head(5)}')
    print('\n')
    # clustering
    print('Initializing a hierarchical_clustering model...')
    hc_model=joblib.load('./model/quick_test/hc_model.pkl')
    print('Model hyperparameters:')
    print(f'method:{hc_model.method}\ndistance:{hc_model.distance}\npartition_line:{hc_model.partition_line}')
    print('\n')

    print('Initializing a hierarchical_clustering model...')
    km=joblib.load('./model/quick_test/kmeans_model.pkl')
    print('Model hyperparameters:')
    print(f'n_clusters:{km.n_clusters}\nmax_iter:{km.max_iter}')
    print('\n')

    # choose recommmedation molecules
    print('Calculating potential molecules with suitable anode limits recommended by kmeans clustering...')
    recommend_mols=joblib.load('./model/quick_test/recommend_mols.pkl')
    print(f'Recommended molecules numbers: {len(recommend_mols)}')
    print('Recommended molecules snapshot:')
    print(recommend_mols.head(5))
    print('\n')

    # Pareto fliter
    print('Initializing a Pareto filter model...')
    p=joblib.load('./model/quick_test/pareto_model.pkl')
    print('\n')
    print('Calculating potential molecules with high capacity and simple structure...')
    front=joblib.load('./model/quick_test/pareto_front.pkl')
    print('Pareto optimal molecules:')
    print(front)

    # Rank
    print('\nRank snapshot:')
    rank=rank_after_pareto(front=front,conf=conf)
    rank.to_csv('./outputs/rank.csv',index=False)
    print(rank)
    print(f"Optimal molecule:{rank['canonicalsmiles'][0]}")
    print(f"Anode limit(V vs Na+/Na):{rank['anode_limit'][0]}")
    print(f"Theoretical specific capacity(mAh/g):{rank['capacity'][0]}")
    print(f"SCScore:{rank['scscore'][0]}")
    print(f"Spacial score :{rank['spacial_score'][0]}")
    print(f"Total rank :{rank['total_score'][0]}")

def main():
    # load data and conf
    conf=load_conf()
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    mols=pd.read_csv(conf.data_path)

    # calculate descriptors
    des=calculate_descriptors(conf,mols)
    normal_des=normalized_descriptors(des)

    # pca
    pca_model=pca.PCA(data=normal_des,n_components=2)
    demension_explained_variance=get_demension_explained_variance(data=normal_des,pca_model=pca_model)
    best_demension=demension_explained_variance[demension_explained_variance['accumalative_explained_variance_ratio']>=0.95]['demension'].min()
    pca_model.n_components=best_demension
    features=pca_model.fit_transform()
    # features=normal_des
    
    # hc clusterting, recommend k for kmeans
    hc_model=hc.hierarchical_clustering(X=features,method=conf.hc.method,distance=conf.hc.distance,partition_line=0)
    partition_k=get_partition_k_from_hc(hc_model,range=[3,10.5,0.1])
    km=kmeans.kmeans(data=features,n_clusters=2,max_iter=conf.kmeans.max_iter)
    partition_k=get_best_k_by_ch(km,partition_k)
    # print(partition_k)

    # choose best k for kmeans
    best_partition=partition_k.loc[partition_k['silhouette_scores'].idxmax(), 'partition_line']
    best_k=partition_k.loc[partition_k['silhouette_scores'].idxmax(), 'k']

    # clustering
    hc_model.partition_line=best_partition
    joblib.dump(hc_model,'./model/quick_test/hc_model.pkl')
    hc_model._fit()
    hc_model.plot_dendrogram(save='./outputs/hc_dendro.png')
    hc_info=hc_model.group_mapping()

    km.n_clusters=best_k
    joblib.dump(km,'./model/quick_test/kmeans_model.pkl')
    km.fit()
    represent=km.get_representative_mol()
    represent['smiles'] = represent['Molecule'].apply(lambda x: mols.loc[mols['idx'] == x, 'canonicalsmiles'].values[0])
    kmeans_info=km.get_fit_info()
    t_sne=km.get_t_sne()
    heatmap=km.get_heatmap()

    # save files
    des=pd.concat([mols[['idx','canonicalsmiles']],des])\
        .to_csv('./outputs/descriptors.csv',index=False)
    normal_des=pd.concat([mols[['idx','canonicalsmiles']],normal_des])\
        .to_csv('./outputs/normalized_descripotrs.csv',index=False)
    demension_explained_variance.to_csv('./outputs/pca_demension_explained_variance.csv',index=False)
    features.to_csv('./outputs/features.csv',index=False)
    partition_k.to_csv('./outputs/partition_k_internal_validity.csv',index=False)
    kmeans_info.to_csv('./outputs/kmeans_info.csv',index=False)
    represent.to_csv('./outputs/kmeans_represent.csv',index=False)
    t_sne.to_csv('./outputs/kmeans_t_sne.csv',index=False)
    heatmap.to_csv('./outputs/kmeans_heatmap.csv',index=False)
    hc_info.to_csv('./outputs/hc_info.csv',index=False)

    
    if conf.dft_status==True:
        # choose recommmedation molecules
        dft=pd.read_csv(conf.dft_data_path)
        recommend_mols=choose_recommend_mols(df=dft,lower_limit=conf.anode_limit_lower,upper_limit=conf.anode_limit_upper)
        
        # Pareto fliter
        recommend_mols[['scscore','spacial_score']]= recommend_mols['canonicalsmiles'].apply(lambda x : pd.Series(utils.get_scscore(x,tar=['scscore','spatial'])))
        recommend_mols['capacity']=recommend_mols['canonicalsmiles'].apply(utils.get_specific_capacity)
        recommend_mols.to_csv('./outputs/recommend_mols.csv',index=False)
        joblib.dump(recommend_mols,'./model/quick_test/recommend_mols.pkl')
        p=Pareto.Pareto(df=recommend_mols)
        joblib.dump(p,'./model/quick_test/pareto_model.pkl')
        front=p.pareto_front()
        joblib.dump(front,'./model/quick_test/pareto_front.pkl')
        front.to_csv('./outputs/pareto_front.csv',index=False)
        
        # Rank
        rank=rank_after_pareto(front=front,conf=conf)
        rank.to_csv('./outputs/rank.csv',index=False)
    

if __name__=='__main__':
    test=int([i for i in sys.argv if i.startswith('quick_test=')][0].split('=')[1])
    if test==1:
        quick_test()
    elif test==0:
        main()