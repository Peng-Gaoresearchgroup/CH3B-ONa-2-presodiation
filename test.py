import unittest
import utils
import pandas as pd
from model.Pareto import Pareto
class TestAddFunction(unittest.TestCase):
    def test(self):
        smiles='COc1ccc(B(O[Na])O[Na])c(O[Na])c1'
        print(utils.get_specific_capacity(smiles))
        print(utils.get_all_elements(df=pd.read_csv('./data/data.csv')))
        # print(utils.get_scscore('CCO',tar=['scscore','sascore','spatial']))
    def test2(self):
        p=Pareto(df=pd.read_csv('./outputs/rank_absolute.csv'))
        print(p.pareto_front())

if __name__ == '__main__':
    unittest.main()