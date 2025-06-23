# Title
### Introduction

### Contents
The project is as follows：
```
├── main.py 
├── utils.py
├── conf/
│   └── conf.yaml    # global configuration information, such as file storage paths and model hyperparameters
├── data/
│   ├── data.csv    # input samples for hierarchical clustering and KMeans clustering
│   └── recommend_cluster.csv    # 195 molecules recommended by KMeans
├── model/
│   ├── hc.py    # hierarchical clustering
│   ├── kmeans.py    # kmeans clustering
│   ├── Pareto.py    # Pareto optimazation
│   └── pca.py    # principal component analysis
├── outputs/    # generated after running main.py
│   ├── data_descriptors.csv    # results for descriptor of molecules in /data/data.csv
│   ├── data_normalized_descriptors.csv    # normalized result of data_descriptors.csv
│   ├── hc_dendro.png    # hierarchical clustering dendrogram, matplotlib
│   ├── hc_information.csv    # hierarchical clustering grouping information
│   ├── kmeans_fit_information.csv    # recording high-dimensional data for KMeans clustering
│   ├── kmeans_plot_information.csv    # coordinates in kmeans_scatter.png
│   ├── kmeans_representative_molecules.csv    # molecule closest to the center of the cluster
│   ├── kmeans_scatter.png    # KMeans clustering scatter plot, reduced to 2D with PCA, matplotlib
│   ├── pca_features.csv    # PCA downscaling of ./outputs/data_normalized_descriptors.csv for clustering
│   ├── rank_normalized.csv    # normalized rank of rank_absolute.csv, you can know the final rank
│   └── rank_Pareto_front.csv    # Pareto optimal molecules
└── requirements.txt
```
### System requirements

In order to run source code file in the Data folder, the following requirements need to be met:
- Windows, Mac, Linux
- Python and the required modules. See the [Instructions for use](#Instructions-for-use) for versions.

### Installation
You can download the zip package directly from this github site,or use git in the terminal：
```
git clone https://github.com/Peng-Gaoresearchgroup/CH3B-ONa-2-presodiation.git
```

### Instructions for use
- Environment
```
# create environment, conda is recommended
conda create -n yourname -c conda-forge rdkit=2024.9.4 python=3.11.8

# install python modules
pip install -r ./requirments.txt

# switch to it
conda activate yourname
```

- Quick test

```
sed -i 's/load_model : False/load_model : True/g' ./conf.yaml
sed -i 's/rank : True/rank : False/g' ./conf.yaml
python ./main.py
sed -i 's/rank : False/rank : True/g' ./conf.yaml
python ./main.py
```

- Reproduce the paper

Firstly, this project will pause to perform DFT calculations based on the results of the clustering, which will be used as input for the next ranking section. If you want to reproduce this project from scratch, follow the steps below.

1. Download original data from PubMed. See the paper for details. Rename it to "data.csv", put it into './data/'. The format references existing [data.csv](./data/data.csv)

2. In [conf](./conf/conf.yaml), modify these key and values:
    ```
    calculate_features : True
    load_model : False
    rank : False
    ``` 
    run [main.py](./main.py), clustering will be performed.

3. Analyze clustering results(see article), get recommended molecules, and their DFT calculated value, name it to "DFT_result.csv", put it into ./data/, make sure the column names match [existing files](./data/DFT_result.csv).

4. In [conf](./conf/conf.yaml), modify these key and values:
    ```
    rank : True
    ``` 
    run [main.py](./main.py) again, and the program will score the recommended molecules, generating [rank_absolute.csv](./outputs/rank_absolute.csv) and [rank_Pareto_front.csv](./outputs/rank_normalized.csv).

### Contributions
Y. Gao, Z. Zheng and G. Wu developed a workflow. G. Wu wrote the program.

### License
This project uses the [MIT LICENSE](LICENSE).

### Disclaimer
This code is intended for educational and research purposes only. Please ensure that you comply with relevant laws and regulations as well as the terms of service of the target website when using this code. The author is not responsible for any legal liabilities or other issues arising from the use of this code.

### Contact
If you have any questions, you can contact us at: yuegao@fudan.edu.cn
