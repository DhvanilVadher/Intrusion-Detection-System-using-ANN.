import pandas as pd
f=pd.read_csv("./editcsv/FridayDDOS.csv")
keep_col = [' Average Packet Size','Active Mean',' Active Min',' Flow IAT Mean',' Flow Duration',' Fwd Packet Length Mean','Total Length of Fwd Packets',' Subflow Fwd Bytes',' Bwd IAT Mean',' Bwd Packet Length Std',' Bwd Packet Length Min',' Label']
new_f = f[keep_col]
new_f.to_csv("FridayCUT.csv", index=False)
