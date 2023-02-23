import matplotlib as mpl
import os
import pandas as pd
import numpy as np
from numpy import hstack
#from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io
import numpy as np
from numpy import genfromtxt
import itertools
np.random.seed(0)
#import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
#from json.decoder import JSONDecodeError
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.metrics import r2_score
import json
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
from sklearn.model_selection import KFold
from sklearn import tree
#import pydotplus
from IPython.display import Image
from scipy.stats import pearsonr
from sklearn.utils import shuffle
import pickle
from sklearn.metrics import precision_recall_fscore_support
from pandas.io.json import json_normalize
###########################################################
import pymysql
import pymssql
import yaml
import pandas as pd
import traceback

def get_creds(db):
    creds = dict()
    with open('/home/shared/utils/creds.yaml') as file:
        creds = yaml.load(file)[db]
    return creds



def bankapp(query):
    # creds = dict()
    # with open('/home/shared/utils/creds.yaml') as file:
    #     creds = yaml.load(file,Loader=yaml.FullLoader)['bankapp']
    creds = get_creds('bankapp')
    try:
        conn = pymysql.connect(host=creds['host'],
                                port=creds['port'],
                                db=creds['database'],
                                user=creds['username'],
                                password=creds['password'])
        df_queried = pd.read_sql_query(query,con=conn)
    finally:
        conn.close()
    return df_queried


def iloans(query):
    # creds = dict()
    # with open('/home/shared/utils/creds.yaml') as file:
    #     creds = yaml.load(file,Loader=yaml.FullLoader)['iloans']
    creds = get_creds('iloans')
    try:
        conn = pymssql.connect(server=creds['server'],
                                port=creds['port'],
                                database=creds['database'],
                                user=creds['username'],
                                password=creds['password'])
        df_queried = pd.read_sql_query(query,con=conn)
    finally:
        conn.close()
    return df_queried






#############FlattenJsonFUnc
def flatten_json(y):
	out = {}
	def flatten(x, name=''):
		if type(x) is dict:
			for a in x:
				flatten(x[a], name + a + '_')
		elif type(x) is list:
			i = 0
			for a in x:
				flatten(a, name + str(i) + '_')
				i += 1
		else:
			out[name[:-1]] = x
	flatten(y)
	return out



def flatten_json_get_loc(y, attr_name):
    out = {}
    attr_addr = ''
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
            if (name[:-1].endswith('_name') and x == attr_name):
                out['attr_addr'] = name[:-5]+str('value')

    flatten(y)
    return out['attr_addr']


def flatten_json_get_attr(y, attr_loc):
    out = {}
    attr_addr = ''
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
            if (name[:-1] == attr_loc):
                out['attr'] = x
    flatten(y)
    return out['attr']


def flatten_json_fill_attr(FJ2, attribute):
	try:
		FJ2[attribute] =  flatten_json_get_attr(Loaded,flatten_json_get_loc(Loaded, attribute))
	except:
		FJ2[attribute] = "NA"
	return FJ2


#Month_num = 1
#Month_name = 'Jan'

pd.set_option('display.max_rows', None)



dataRibbit = iloans('''select LE.leadid, LE.timeadded, r.rawresponse, 
r.actioncode, r.NumericScore as score, rr.rawresponse as BLPresponse, 
LA.loanid, LA.TotalPrincipal, LA.OriginalPrincipal, LA.PaidPrincipal, 
LA.PaidFinanceFee, LA.PaidFeeCharges, 
(case when LA.IsFirstDefault='True' or (LA.IsFirstDefault is null and LS.isbad=1) then 1 else 0 end) as IsFirstDefault,
(Case when ISNULL(LA.LoanAge,0) > 0 then 1 when (LA.LoanStatusID) = 16 then 1 end) as Mature,
LA.loanstatus, LA.LoanStatusid, LS.IsOriginated, LS.IsGood, LS.IsBad 
from dbo.view_FCL_LeadAccepted LE 
left join [FreedomCashLenders].[dbo].[view_FCL_RibbitBVPlusReportData] r on LE.leadid = r.leadid 
left join [FreedomCashLenders].[dbo].[view_FCL_RibbitBLPlusReportData] rr on LE.leadid = rr.leadid 
left  join view_fcl_loan LA on LE.loanid = LA.loanid 
left join view_fcl_loanstatus LS on LA.LoanStatusid = LS.loanStatusId 
where LE.timeadded >= '2022-01-12' and  LE.timeadded < '2022-01-13' and rr.rawresponse is not null
and LE.Subscription not like '%Reloan%' 
and LE.subscription not like '%Admin%' 
and LE.subscription not like '%VIP-Holiday%'
 and LE.subscription not like '%Month Installment%'
and LE.subscription not like '%ReUp%';''')


dataRibbit2 = dataRibbit[dataRibbit['rawresponse'].notnull()]

dataRibbit2 = dataRibbit2[dataRibbit2['BLPresponse'].notnull()]


dataRibbit2 = dataRibbit2.fillna(0)


dataRibbit2['netpaid'] = dataRibbit2['PaidPrincipal'] + dataRibbit2['PaidFinanceFee']+dataRibbit2['PaidFeeCharges']

dataRibbit2['PerPaid'] = dataRibbit2['netpaid']/dataRibbit2['OriginalPrincipal']

dataRibbit2['TenPaid'] = np.where(dataRibbit2['PerPaid']>0.1, 1, 0)

dataRibbit2['TwentyPaid'] = np.where(dataRibbit2['PerPaid']>0.2, 1, 0)

dataRibbit2['ThirtyPaid'] = np.where(dataRibbit2['PerPaid']>0.3, 1, 0)


dataRibbit4 = dataRibbit2

dataRibbit4.to_csv('/home/manikanta/blp_data_jan_2022_01_09_2022.csv',sep='|')
#####################################################################################################################

JSON = dataRibbit4['BLPresponse'][dataRibbit4.index[0]]

Loaded = json.loads(JSON)
dictJson = flatten_json(Loaded)
FlatJSON = json_normalize(dictJson)




### is used for specified columns only
#####loading static cols pickle file
# file = open("BVP_column_dict.pkl",'rb')
# cols = pickle.load(file)
# file.close()

# try:
# 	BVPattrs = FlatJSON[cols]
# except Exception as e:
# 	FlatJSON=pd.concat([FlatJSON,pd.DataFrame(columns=cols)])
# 	BVPattrs = FlatJSON[cols]

#BVPattrs = FlatJSON

cols=['leadid','timeadded','bvactioncode','bvscore','loanid']
BVPattrs =pd.DataFrame(columns=cols,index=[0])
# BVPattrs['leadid']=np.NaN
# BVPattrs['timeadded']=np.NaN
# BVPattrs['bvactioncode']=np.NaN
# BVPattrs['bvscore']=np.NAN
# BVPattrs['loanid']=np.NAN





#dataRibbit4 = dataRibbit4.reset_index()
#BVPattrs['leadid'] = dataRibbit4['leadid'][1]
#BVPattrs['timeadded'] = dataRibbit4['timeadded'][1]
#BVPattrs['bvactioncode'] = dataRibbit4['actioncode'][1]
#BVPattrs['bvscore'] = dataRibbit4['score'][1]
#BVPattrs['loanid'] = dataRibbit4['loanid'][1]



df1=pd.DataFrame()

for i in range(0, dataRibbit4.shape[0]):
	JSON = dataRibbit4['BLPresponse'][dataRibbit4.index[i]]
	try:
		Loaded = json.loads(JSON)
		dictJson = flatten_json(Loaded)
		#print(i)
		print(str(i)+'/'+str(dataRibbit4.shape[0]))
		FlatJSON = json_normalize(dictJson)
		#FlatJSON2 = FlatJSON[cols]
		FlatJSON2 = FlatJSON
		FlatJSON2['leadid'] = dataRibbit4['leadid'][i]
		FlatJSON2['timeadded'] = dataRibbit4['timeadded'][i]
		FlatJSON2['bvactioncode'] = dataRibbit4['actioncode'][i]
		FlatJSON2['bvscore'] = dataRibbit4['score'][i]
		FlatJSON2['loanid'] = dataRibbit4['loanid'][i]
		#FlatJSON2['BLPresponse'] = dataRibbit4['BLPresponse'][i]
		BVPattrs = pd.concat((BVPattrs, FlatJSON2), axis=0)
	except Exception as e:
		print(e)
		pass
		





#BVPattrs["IsFirstDefault"] = BVPattrs["IsFirstDefault"].astype(int)


BVPattrs.to_csv('/home/manikanta/blp_attrs_12th_jan_2022_02_22_2023.csv',sep='|')


