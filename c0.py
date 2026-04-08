import subprocess, sys
for pkg in ['statsmodels','plotly','ipywidgets','kaleido','scikit-learn','yfinance']:
    subprocess.check_call([sys.executable,'-m','pip','install',pkg,'-q'])
import warnings; warnings.filterwarnings('ignore')
import sqlite3, numpy as np, pandas as pd
import logging
from itertools import combinations
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, HTML
pd.set_option('display.float_format','{:.4f}'.format)
import yfinance as yf
import time
import os
# print('套件導入完成。')