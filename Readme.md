# CH3B-ONa-2-presodiation
### Introduction
This is the code for the "Boron-centered organic salts enabling Na-ion supply and interfacial protection for practical Na-ion batteries" for Z. Zheng et al.
### Contents
```
├── main.py 
├── utils.py
├── conf/
│   └── conf.yaml    # global configuration, such as file storage paths and model hyperparameters
├── data/
│   └── data.csv    # input samples for hierarchical clustering and KMeans clustering
├── model/
│   ├── hc.py    # hierarchical clustering
│   ├── kmeans.py    # kmeans clustering
│   ├── Pareto.py    # Pareto optimazation
│   └── pca.py    # principal component analysis
│   └── quick_test/ # saved model pkl for quick test. 
├── outputs/    # generated after running main.py
│   ├── *descriptors*.csv    # results for descriptor of molecules in /data/data.csv
│   ├── hc*    # hierarchical clustering data
│   ├── kmeans*.csv    # KMeans clustering data
│   ├── features.csv    # PCA downscaling of ./outputs/data_normalized_descriptors.csv for clustering
│   ├── rank.csv    # normalized rank of rank_absolute.csv, you can know the final rank
│   ├── recommend_mols.csv    # recommend molecules after clustering
│   └── pareto_front.csv    # Pareto optimal molecules
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
conda create -n envname -c conda-forge rdkit=2024.9.4 python=3.11.8

# install python modules
pip install -r ./requirments.txt

# switch to it
conda activate envname
```

- Quick test

```
python ./main.py quick_test=1
```

- Reproduce the paper

```
python ./main.py quick_test=0
```

### Contributions
Y. Gao, Z. Zheng and G. Wu developed a workflow. G. Wu wrote the program.

### License
This project uses the [MIT LICENSE](LICENSE).

### Disclaimer
This code is intended for educational and research purposes only. Please ensure that you comply with relevant laws and regulations as well as the terms of service of the target website when using this code. The author is not responsible for any legal liabilities or other issues arising from the use of this code.

### Contact
If you have any questions, you can contact us at: yuegao@fudan.edu.cn
