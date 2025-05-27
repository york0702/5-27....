# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:56:31 2025

@author: user
"""

import pandas as pd
myfile = 'kbars_MXF202412_2023-12-21-2024-04-11'                  
'''
'kbars_2330_2022-01-01-2024-04-09', 'kbars_1522_2020-01-01-2024-04-12', 'kbars_2356_2020-01-01-2024-04-12'
'kbars_TXF202412_2023-12-21-2024-04-11', 'kbars_MXF202412_2023-12-21-2024-04-11'

'''
df = pd.read_pickle(myfile+'.pkl')
df.to_excel(myfile+'.xlsx', index=False)