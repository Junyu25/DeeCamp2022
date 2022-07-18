

import cobra
import pandas as pd
from cobra.io import load_model
from itertools import combinations

model = load_model("iJO1366") #no need to import from matlab
targetNum = 20
KOlist = ('GLCabcpp', 'GLCptspp', 'HEX1', 'PGI', 'PFK', 'FBA', 'TPI', 'GAPD', 'PGK', 'PGM', 'ENO', 'PYK', 'LDH_D', 'PFL', 'ALCD2x', 'PTAr', 'ACKr', 'G6PDH2r', 'PGL', 'GND', 'RPI', 'RPE', 'TKT1', 'TALA', 'TKT2', 'FUM', 'FRD2', 'SUCOAS', 'AKGDH', 'ACONTa', 'ACONTb', 'ICDHyr', 'CS', 'MDH',  'MDH2', 'MDH3', 'ACALD')


targetList = KOlist[:targetNum]
combList = []
for i in range(1,targetNum+1):
    combList.extend(list(combinations(targetList, i)))

model.objective = 'EX_succ_e'
    
objList = []
n = 0
for comb in combList:
    with model:
        for rec in comb:
            model.reactions.get_by_id(rec).knock_out()
        objList.append(model.optimize().objective_value)
    n += 1
    if n%100 == 0:
        print(n)
    
df = pd.DataFrame()
df["comb"] = combList
df["obj"] = objList

df.to_csv("Ecoli_Suc_KO.csv")

