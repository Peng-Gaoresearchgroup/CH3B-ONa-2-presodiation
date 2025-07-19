import rdkit.Chem as Chem
from rdkit.Chem import AllChem,Descriptors
import pandas as pd

def is_valid(smiles):
    # print(f'Checking {smiles} is valid?')
    if '.' in smiles or '+' in smiles or '-' in smiles or smiles=='':
        return False
    else:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return True
            else:
                return False
        except:
            return False

def convert2canonical(smiles):
    # make sure smiles is valid
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def is_contain_special_groups(smiles):
    try:
        smarts_list = ['[$([NH,NH2]),$([SH])]','[$([OH][C]);!$([OH][C]=[O])]'] # Amino, imino, sulfhydryl, alcohol hydroxyl
        for smarts in smarts_list:
            pattern = Chem.MolFromSmarts(smarts)
            mol = Chem.MolFromSmiles(smiles)
            matches=mol.HasSubstructMatch(pattern)
            if matches:
                return True
        return False
    except:
        return True

def count_single_OH(smiles):
    smarts = '[$([OH]);!$([OH][C]=[O])]'
    pattern = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smiles)
    matches= mol.GetSubstructMatches(pattern)
    return len(matches)

def count_double_OH(smiles):
    smarts = '[$([OH]~[*]~[OH])]'
    pattern = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smiles)
    matches= mol.GetSubstructMatches(pattern)
    return len(matches)

def count_COOH(smiles):
    smarts = '[$([OH][C]=[O])]'
    pattern = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smiles)
    matches= mol.GetSubstructMatches(pattern)
    return len(matches)
       
def count_B(smiles):
    smarts = '[$([B])]'
    pattern = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smiles)
    matches= mol.GetSubstructMatches(pattern)
    return len(matches)

def replace_H2Na(smiles):
    try:
        replace_rule=['[O;H]','[O]-[Na]']
        mol = Chem.MolFromSmiles(smiles)
        smarts = Chem.MolFromSmarts(replace_rule[0])
        replacement_mol = Chem.MolFromSmiles(replace_rule[1])
        product = AllChem.ReplaceSubstructs(mol, smarts, replacement_mol, replaceAll=True)
        return Chem.MolToSmiles(product[0])
    except:
        return False


def get_capacity(smiles):
    def ck_Natype(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        na_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'Na']
        na_idx_map = {idx: i for i, idx in enumerate(na_indices)}

        pattern = Chem.MolFromSmarts("[Na]-O-B-O-[Na]")
        matches = mol.GetSubstructMatches(pattern)
        na_types = [1] * len(na_indices)
        
        if matches:
            na1, _, _, _, na2 = matches[0]
            if na1 in na_idx_map:
                na_types[na_idx_map[na1]] = 0
            if na2 in na_idx_map:
                na_types[na_idx_map[na2]] = 1

        return na_types
    na_types=ck_Natype(smiles)
    mol=Chem.MolFromSmiles(smiles)
    mol=Chem.AddHs(mol)
    mol_weight = Descriptors.MolWt(mol)
    Na_num = sum(1 for i in na_types if i ==1)
    # Na_num = len(na_types)
    specific_capacity=Na_num*96500/(3.6*mol_weight)
    return specific_capacity


def get_molwt(smiles):
    mol=Chem.MolFromSmiles(smiles)
    mol=Chem.AddHs(mol)
    mol_weight = Descriptors.MolWt(mol)
    return mol_weight

def main():
    df1=pd.read_csv('./data/PubChem_compound_smiles_substructure_[B][C].csv')
    df2=pd.read_csv('./data/PubChem_compound_smiles_substructure_[B][O].csv')
    li=df1['smiles'].to_list()+df2['smiles'].to_list()
    df=pd.DataFrame({'smiles':li})

    df['is_valid']=df['smiles'].apply(is_valid)
    df=df[df['is_valid']==True]

    df['canonicalsmiles']=df['smiles'].apply(convert2canonical)

    df['contain_special_group']=df['canonicalsmiles'].apply(is_contain_special_groups)
    df=df[df['contain_special_group']==False]



    df['num_OH']=df['canonicalsmiles'].apply(count_single_OH)
    df['num_COOH']=df['canonicalsmiles'].apply(count_COOH)
    df['num_double_OH']=df['canonicalsmiles'].apply(count_double_OH)
    df=df[(df['num_COOH']>0) | ((df['num_COOH']==0)&(df['num_double_OH']>0))]


    df['num_B']=df['canonicalsmiles'].apply(count_B)
    df=df[df['num_B']>0]

    df['canonicalsmiles']=df['canonicalsmiles'].apply(replace_H2Na)
    df=df[df['canonicalsmiles']!=False]

    df['capacity']=df['canonicalsmiles'].apply(get_capacity)
    df=df[df['capacity']>=120]
    print(len(df))

    df['molwt']=df['canonicalsmiles'].apply(get_molwt)
    df=df[(df['molwt'] <=200) & (df['molwt'] >= 100)]
    print(len(df))

    df = df.drop_duplicates(subset='canonicalsmiles')
    df = df.reset_index(drop=True)
    df['precursor']=df['smiles']
    df=df['idx','canonicalsmiles','precuror','capcaity','molwt']
    df['idx'] = df.index
    print(len(df))
    df.to_csv('./data/data.csv',index=False)
if __name__=='__main__':
    main()