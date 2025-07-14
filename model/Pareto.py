import pandas as pd
class Pareto:
    def __init__(self, df, target_cols=['scscore','spacial_score','capacity(mAh/g)'],info_cols=['idx','canonicalsmiles','cluster']):
        def origin2tar(li):
            li[2]=-li[2]
            # li[3]=-li[3]
            return li
        population = []
        # df = df.set_index('idx')
        for idx, row in df.iterrows():
            target_values = origin2tar([row[col] for col in target_cols])
            old_idx=row[info_cols[0]]
            smiles=row[info_cols[1]]
            cluster=row[info_cols[2]]
            population.append((idx, target_values,[old_idx,smiles,cluster]))

        self.population = population
        self.target_cols=target_cols
        self.info_cols=info_cols

    
    def dominate(self, a, b):
        a=a[1] # a=(idx,[target1,target2,target3],info_list)
        b=b[1] 
        compare= [None]*len(a)
        for i in range(len(a)):  
            if b[i] < a[i]:
                compare[i]='b'
            elif b[i] > a[i]:
                compare[i]='w'
            else:
                compare[i]='='
        if 'b' in compare and 'w' not in compare:
            return True
        else:
            return False

    def pareto_front(self):

        pareto_front = []
        for i in range(len(self.population)):
            is_dominated = False
            for j in range(len(self.population)):
                if i != j and self.dominate(self.population[i], self.population[j]):  # 检查 population[j] 是否支配 population[i]
                    is_dominated = True #如果支配了，[i]必定不在前沿，故break
                    break
            if not is_dominated:
                pareto_front.append(self.population[i])
        
        df = self.population_to_df(population=pareto_front)
        return df

    
    def population_to_df(self,population):
        def tar2origin(li):
            li[2]=-li[2]
            # li[3]=-li[3]
            return li

        records = []
        for idx, target_list, info in population:
            # row = {'idx': idx}
            row={}
            row.update({name: value for name, value in zip(self.target_cols, tar2origin(target_list))})
            row[self.info_cols[0]] = info[0]
            row[self.info_cols[1]] = info[1]
            row[self.info_cols[2]] = info[2]
            records.append(row)
        return pd.DataFrame(records)

