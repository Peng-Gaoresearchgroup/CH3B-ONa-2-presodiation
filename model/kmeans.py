from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import extract_xy

class kmeans:
    def __init__(self,data, n_clusters=3, init='k-means++', max_iter=300, random_state=42):
        """
        - n_clusters: int.
        - init: str, initializs method ('k-means++', 'random').
        - max_iter: int.
        - random_state: int.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.data=data

    def fit(self):
        """
        - data: DataFrame or ndarray.
        """
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.values
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model.fit(self.data)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.inertia_ = self.model.inertia_
    def predict(self, data):
        """
        return:
        - labels: ndarray
        """
        if self.model is None:
            raise ValueError("train model first.")
        if isinstance(data, pd.DataFrame):
            data = data.values
        return self.model.predict(data)

    def plot_scatters(self, data,save):
        """
        - data: DataFrame or ndarray
        """
        if self.labels_ is None:
            raise ValueError("train model first.")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.shape[1] != 2:
            raise ValueError("Demension error")

        plt.figure(figsize=(8, 6))
        for i in range(self.n_clusters):
            cluster_points = data[self.labels_ == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")
        
        plt.scatter(
            self.cluster_centers_[:, 0],
            self.cluster_centers_[:, 1],
            s=200,
            c='red',
            marker='+',
            label='Centroids'
        )
        plt.title("KMeans Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.savefig(save,dpi=300,format='png')
        # plt.show()

    def get_cluster_centers(self):
        """
        - cluster_centers: ndarray.
        """
        if self.cluster_centers_ is None:
            raise ValueError("train model first.")
        return self.cluster_centers_

    def get_plot_information(self, data,save):
        if self.labels_ is None:
            print("Model has not been fitted yet.")
            return None


        data = np.array(data)

        cluster_data = []
        for idx, label in enumerate(self.labels_):
            x,y=extract_xy(str(data[idx]))
            cluster_data.append({ "Molecule": idx,"Cluster": label, "x": x,'y':y})
        info=pd.DataFrame(cluster_data) 
        info.to_csv(save,index=False)
        grouped = info.groupby('Cluster')['Molecule'] \
               .agg(lambda x: ', '.join(x.astype(str))) \
               .sort_index()
        summary_str = '\n'.join(
        [f"Cluster{group}: {samples}" 
         for group, samples in grouped.items()]
        )
        print('kmeans_information:\n',summary_str)
        with open('./outputs/kmeans_infor.txt', 'w') as f:
            f.write(summary_str)
        return info

    def get_labels(self):
        """
        return:
        - labels: ndarray.
        """
        if self.labels_ is None:
            raise ValueError("train model first.")
        return self.labels_

    def get_inertia(self):
        """
        return:
        - inertia: float    
        """
        if self.inertia_ is None:
            raise ValueError("train model first.")
        return self.inertia_
    

    def get_fit_info(self):
        if self.labels_ is None:
            print("Model has not been fitted yet.")
            return None
        # np.set_printoptions(suppress=True)
        np.set_printoptions(precision=16)
        np.set_printoptions(linewidth=np.inf)
        self.data = np.array(self.data)
        cluster_data = []
        for idx, label in enumerate(self.labels_):
            # x,y=extract_xy(str(data[idx]))
            cluster_data.append({ "Molecule": idx,"Cluster": label,'data':self.data[idx],'Cluster_center':self.cluster_centers_[label],'Distance_to_center':np.linalg.norm(self.data[idx] - self.cluster_centers_[label])})
        info=pd.DataFrame(cluster_data)
        # info.to_csv(save,index=False)
        return info

    def get_representative_mol(self):
        df=self.get_fit_info()
        result = df.loc[df.groupby('Cluster')['Distance_to_center'].idxmin(), ['Molecule','Cluster','data','Cluster_center','Distance_to_center']]
        return result
    
    def get_t_sne(self, sample_ids=None, perplexity=30, random_state=0):

        if self.labels_ is None:
            raise ValueError("Fit first!")
        
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        tsne_result = tsne.fit_transform(self.data)

        n_samples = self.data.shape[0]
        if sample_ids is None:
            sample_ids = list(range(n_samples))

        df = pd.DataFrame({
            'id': sample_ids,
            'cluster': self.labels_,
            'X': tsne_result[:, 0],
            'Y': tsne_result[:, 1]
        })
        df = df.sort_values(by='cluster')
        slices = []
        for a_value in df['cluster'].unique():
            sliced_df = df[df['cluster'] == a_value].reset_index(drop=True)
            renamed = sliced_df.rename(columns=lambda col: f'{col}_{a_value}')
            max_len = df['cluster'].value_counts().max()
            renamed = renamed.reindex(range(max_len))

            slices.append(renamed)
        df = pd.concat(slices, axis=1)
        return df
    
    def get_silhouette_scores(self):
        from sklearn.metrics import silhouette_score
        return silhouette_score(self.data, self.labels_)
    
    def get_wcss(self):
        if self.inertia_ ==None:
            print('Model not fit yet')
        else:
            return self.inertia_
        
    def get_tss(self):
        mean = np.mean(self.data, axis=0)
        tss = np.sum(np.linalg.norm(self.data - mean, axis=1) ** 2)
        return tss
    
    def get_heatmap(self):
        n_clusters = len(np.unique(self.labels_))
        cluster_means = np.zeros((n_clusters, self.data.shape[1]))
        
        for i in range(n_clusters):
            cluster_data = self.data[self.labels_ == i]
            cluster_means[i] = np.mean(cluster_data, axis=0)

        df = pd.DataFrame(
            cluster_means,
            index=[f"Cluster {i}" for i in range(n_clusters)],
            columns=[f"PCA feature {j}" for j in range(self.data.shape[1])]
        )
        return df
    
    def get_ch_score(self):
        from sklearn.metrics import calinski_harabasz_score
        ch_score = calinski_harabasz_score(self.data, self.labels_)
        return ch_score
    
    def get_davies_bouldin_index(self):
        n_clusters = len(np.unique(self.labels_))
        intra_cluster_distances = np.zeros(n_clusters)
        inter_cluster_distances = np.zeros((n_clusters, n_clusters))
        
        for i in range(n_clusters):
            cluster_points = self.data[self.labels_ == i]
            centroid = np.mean(cluster_points, axis=0)
            intra_cluster_distances[i] = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
        
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                centroid_i = np.mean(self.data[self.labels_ == i], axis=0)
                centroid_j = np.mean(self.data[self.labels_ == j], axis=0)
                inter_cluster_distances[i, j] = np.linalg.norm(centroid_i - centroid_j)
                inter_cluster_distances[j, i] = inter_cluster_distances[i, j]
        
        db_index = 0
        for i in range(n_clusters):
            db_index += max([(intra_cluster_distances[i] + intra_cluster_distances[j]) / inter_cluster_distances[i, j] 
                            for j in range(n_clusters) if i != j])
        
        db_index /= n_clusters
        return db_index

    def get_dunn_index(self):
        from sklearn.metrics import pairwise_distances
        n_clusters = len(np.unique(self.labels_))
        min_inter_cluster_distance = np.inf
        max_intra_cluster_distance = 0
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_i_points = self.data[self.labels_ == i]
                cluster_j_points = self.data[self.labels_ == j]
                inter_cluster_distance = np.min(pairwise_distances(cluster_i_points, cluster_j_points))
                min_inter_cluster_distance = min(min_inter_cluster_distance, inter_cluster_distance)
        
        for i in range(n_clusters):
            cluster_points = self.data[self.labels_ == i]
            intra_cluster_distance = np.max(pairwise_distances(cluster_points, cluster_points))
            max_intra_cluster_distance = max(max_intra_cluster_distance, intra_cluster_distance)
        
        return min_inter_cluster_distance / max_intra_cluster_distance

    def get_xie_beni_index(self):
        n_samples = self.data.shape[0]
        n_clusters = len(np.unique(self.labels_))
        xie_beni_value = 0
        
        for i in range(n_clusters):
            cluster_points = self.data[self.labels_ == i]
            centroid = np.mean(cluster_points, axis=0)
            distance_to_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
            intra_cluster_dispersion = np.sum(distance_to_centroid**2) / len(cluster_points)
            
            for j in range(n_samples):
                if self.labels_[j] == i:
                    continue
                inter_cluster_distance = np.linalg.norm(self.data[j] - centroid)
                xie_beni_value += (intra_cluster_dispersion / (inter_cluster_distance**2))
        
        return xie_beni_value / n_samples
