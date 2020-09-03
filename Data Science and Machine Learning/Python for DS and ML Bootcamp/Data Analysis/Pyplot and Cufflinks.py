import pandas as pd
import numpy as np

#%pip install plotly 
import plotly 
from plotly import __version__

#%pip install cufflinks 
import cufflinks as cf 

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot 
init_notebook_mode(connected=True)
cf.go_offline()

df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})

df.plot()
df.iplot(kind='surface')