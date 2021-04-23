import pandas as pd
f=pd.read_csv("./AttacksDataSet.csv")
#f['label'] = 'attack'
keep_col = ['pkt_size_avg','active_mean','active_min','flow_iat_mean','flow_duration','fwd_pkt_len_mean','totlen_fwd_pkts','subflow_fwd_byts','bwd_iat_mean','bwd_pkt_len_std','bwd_pkt_len_min','Label']
new_f = f[keep_col]
f.to_csv("AttackDataSet1.csv", index=False)
