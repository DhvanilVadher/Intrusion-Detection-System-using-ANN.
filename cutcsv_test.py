import pandas as pd
f=pd.read_csv("asd.csv")
keep_col = ['pkt_size_avg','active_mean','active_min','flow_iat_mean','flow_duration','fwd_pkt_len_mean','totlen_fwd_pkts','subflow_fwd_byts','bwd_iat_mean','bwd_pkt_len_std','bwd_pkt_len_min']
new_f = f[keep_col]
new_f.to_csv("wednesdayWorkingHourss.csv", index=False)
