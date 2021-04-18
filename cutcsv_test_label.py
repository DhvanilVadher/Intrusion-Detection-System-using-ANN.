
import pandas as pd
df = pd.read_csv('./attack2.csv')
df_reorder = df[['pkt_size_avg','active_mean','active_min','flow_iat_mean','flow_duration','fwd_pkt_len_mean','totlen_fwd_pkts','subflow_fwd_byts','bwd_iat_mean','bwd_pkt_len_std','bwd_pkt_len_min','label']] # rearrange column here
df_reorder.to_csv('attack3.csv', index=False)