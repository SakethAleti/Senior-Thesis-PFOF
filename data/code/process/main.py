import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import os
from datetime import datetime




#### Functions

def import_dict(dict_loc):
    """ Loads a dictionary from a csv """
    df = pd.read_csv(dict_loc)
    df.index = df.iloc[:,0]
    df = df.drop(df.columns[0], axis = 1)
    return df.to_dict()

def convertDateToQuarter(date):
    quarter = (date.month-1)//3 + 1
    return (str(date.year) + 'Q' + str(quarter))




#### Data Import

# 605 and 606 csv Directory
dir_605 = '../../605/'
dir_606 = '../../606/'



### Raw Data


## Import 605 data
# Find market center csvs
marketcenter_csv_list = [x for x in os.listdir(dir_605) if x[-4:] == '.csv']
# Get .csv directories
marketcenter_csv_dirs = [dir_605 + x for x in marketcenter_csv_list]
# Read .csv files
csv_df_list_605 = [pd.read_csv(file, sep = ',') for file in marketcenter_csv_dirs]
# Merge each marketcenter's data
rawdata_605 = pd.concat(csv_df_list_605)
# Clean up
del(csv_df_list_605)


## Import 606 Data
csv_df_list_606 = []
# Find broker folders
broker_folders = [x for x in os.listdir(dir_606) if '.' not in x]
# Merge .csv's for each broker
for broker in broker_folders:
    # Get file locations of csv's for each broker
    directory = dir_606 + broker
    broker_csv_list  = [x for x in os.listdir(directory)]
    broker_csv_dirs  = [dir_606 + broker + '/' + csv for csv in broker_csv_list]
    # Read csv's as dataframes
    csv_df_list_606_broker = [pd.read_csv(file) for file in broker_csv_dirs]
    csv_df_list_606.append(pd.concat(csv_df_list_606_broker))
    
# Merge each broker's data
rawdata_606 = pd.concat(csv_df_list_606)
# Clean up
del(csv_df_list_606)



### Dictionaries
symbol_dict = import_dict('../../keys/symbols.csv')['Exchange']
mktctr_mpid_dict = import_dict('../../keys/mpids.csv')['MPID']
ordertype_dict = {11: 'Market', 12: 'Limit'}
broker_vol_dict = import_dict('../../keys/broker_volumes.csv')




#### Data Prep



### Broker Data


## Prepare Raw Data
# Import Data
data_606 = rawdata_606.copy()

# Fix Routing Venue labels
data_606['RoutingVenue'] = data_606['RoutingVenue'].apply(
    lambda x: mktctr_mpid_dict.get(x.strip(), "(Unknown) " + str(x.strip())))

# Drop unknown routing venues
data_606 = data_606[data_606['RoutingVenue'].apply(lambda x: not x.startswith('(Unk'))]

# Convert date to quarter
data_606['Quarter'] = data_606['Date'].apply(
    lambda x: convertDateToQuarter(datetime.strptime(str(x), '%Y%m')))
data_606['Quarter'] = pd.PeriodIndex(data_606['Quarter'], freq='Q').values
data_606 = data_606.drop('Date', axis=1)

# Change column names
data_606 = data_606.rename(
    columns={'RoutingVenue': 'MarketCenter', 'Pct': 'MktShare'})

# Merge known marketcenters of same firm
data_606 = data_606.groupby(['Broker', 'Exchange', 'OrderType', 'Quarter', 'Rebate', 'MarketCenter']).sum().reset_index()

# Add binary var for presence of rebates
data_606['Rebate_Dummy'] = (data_606['Rebate'].apply(lambda x: (x > 0))
                            | data_606['Broker'].apply(lambda x: x == 'TD Ameritrade')).apply(lambda x: int(x))

# Filter 606 data to market centers with data available
mktctrs_available = rawdata_605['MarketCenter'].unique()
data_606 = data_606[data_606['MarketCenter'].apply(lambda x: x in mktctrs_available)]


## Fill in Missing 0's
data_606['Obs_id'] = data_606['Broker'] + '-' + data_606['MarketCenter'] + '-' + data_606['Exchange'] + '-' + data_606['OrderType']
data_606_new = data_606.copy()

dates_set = pd.Series(list(data_606['Quarter'].unique())).sort_values()

rebate_dummy_dict = {broker: data_606.query('Broker == "' + broker + '"').iloc[0]['Rebate_Dummy'] for broker in data_606['Broker'].unique()}

# from second element onwards
for quarter in dates_set.iloc[0:]: 
    
    print(quarter, end = ' ')
    mask_1 = (data_606['Quarter'] <  quarter) & (data_606['Quarter'] >= (quarter - 2)) # within last given period
    mask_2 = (data_606['Quarter'] == quarter)
            
    set_1 = set(data_606.loc[mask_1]['Obs_id'].unique())
    set_2 = set(data_606.loc[mask_2]['Obs_id'].unique())
    
    # missing id's (last period obs that are not in this period)
    id_list = [list(x.split('-')) for x in (set_1 - set_2)]    
    
    # add missing id's
    print('(%d)' % len(id_list), end = ', ')
    for missing_id in id_list:
        
        data_606_new = data_606_new.append({'Broker': missing_id[0], 'MarketCenter': missing_id[1], 'Exchange': missing_id[2], 
                            'OrderType': missing_id[3], 'Quarter': quarter, 'Obs_id': '-'.join(missing_id),
                            'Rebate_Dummy': rebate_dummy_dict.get(missing_id[0], np.nan), 'MktShare': 0}, 
                           ignore_index = True)

data_606 = data_606_new.copy()
data_606.head()



### Market Center Data

## Prepare Raw Data
# Import data
data_605 = rawdata_605.copy()

# Quarter column
data_605['Quarter'] = data_605['idate'].apply(lambda x: convertDateToQuarter(datetime.strptime(str(int(x)), '%Y%m')))
data_605['Quarter'] = pd.PeriodIndex(data_605['Quarter'], freq='Q').values
data_605 = data_605.drop('idate', axis = 1)

# Temporary Variables for Aggregation
data_605['PrImp_TotalT']     = data_605['PrImpShares']    * data_605['PrImp_AvgT']
data_605['PrImp_TotalAmt']   = data_605['PrImpShares']    * data_605['PrImp_AvgAmt']
data_605['ATQ_TotalT']       = data_605['ATQShares']      * data_605['ATQ_AvgT']
data_605['OTQ_TotalT']       = data_605['OTQShares']      * data_605['OTQ_AvgT']
data_605['AvgRealSpread_T']  = data_605['AvgRealSpread']  * data_605['ExecShares']
data_605['AvgEffecSpread_T'] = data_605['AvgEffecSpread'] * data_605['ExecShares']

data_605 = data_605.groupby(['MarketCenter', 'Quarter', 'Exchange', 'OrderCode']) \
        .sum().reset_index()

# Reconstruct original variables
data_605['PrImp_AvgT']     = data_605['PrImp_TotalT']     / data_605['PrImpShares']
data_605['PrImp_AvgAmt']   = data_605['PrImp_TotalAmt']   / data_605['PrImpShares'] 
data_605['ATQ_AvgT']       = data_605['ATQ_TotalT']       / data_605['ATQShares']
data_605['OTQ_AvgT']       = data_605['OTQ_TotalT']       / data_605['OTQShares']
data_605['AvgRealSpread']  = data_605['AvgRealSpread_T']  / data_605['ExecShares']
data_605['AvgEffecSpread'] = data_605['AvgEffecSpread_T'] / data_605['ExecShares'] 
data_605['PrImp_Pct']      = data_605['PrImpShares']      / data_605['ExecShares']
data_605['ATQ_Pct']        = data_605['ATQShares']        / data_605['ExecShares']
data_605['OTQ_Pct']        = data_605['OTQShares']        / data_605['ExecShares']


## New Vars
# Absolute
data_605['OrderType']    = data_605['OrderCode'].apply(lambda x: ordertype_dict.get(x, 'Other'))
data_605['PrImp_ExpAmt'] = data_605['PrImp_AvgAmt'] * data_605['PrImp_Pct']
data_605['All_AvgT']     = (data_605['PrImp_TotalT'] + data_605['ATQ_TotalT'] + data_605['OTQ_TotalT']) \
                            / data_605['ExecShares']
data_605 = data_605.rename(columns = {'idate': 'Date'})

# Relative values
data_605_grouped = data_605.groupby(['Exchange', 'OrderType', 'Quarter'])

data_605['MktCtrAvg_PrImp_Pct']  = data_605_grouped['PrImp_Pct'].transform("mean")
data_605['Rel_PrImp_Pct']        = data_605['PrImp_Pct'] - data_605['MktCtrAvg_PrImp_Pct']
data_605['MktCtrAvg_PrImp_AvgT'] = data_605_grouped['PrImp_AvgT'].transform("mean")
data_605['Rel_PrImp_AvgT']       = data_605['PrImp_AvgT'] - data_605['MktCtrAvg_PrImp_AvgT']    
data_605['MktCtrAvg_PrImp_ExpAmt'] = data_605_grouped['PrImp_ExpAmt'].transform("mean")
data_605['Rel_PrImp_ExpAmt']       = data_605['PrImp_ExpAmt'] - data_605['MktCtrAvg_PrImp_ExpAmt']   
data_605['MktCtrAvg_All_AvgT'] = data_605_grouped['All_AvgT'].transform("mean")
data_605['Rel_All_AvgT']       = data_605['All_AvgT'] - data_605['MktCtrAvg_All_AvgT']   

data_605.head()




#### Merge Datasets
data_merged = data_605.merge(data_606)

data_merged = data_merged.query('OrderCode < 13')
data_merged = data_merged.drop('Obs_id', axis = 1)
data_merged.set_index(['Quarter'])
data_merged['Broker_Size'] = data_merged['Broker'].apply(lambda x: broker_vol_dict['Size'].get(x))


print('Total Observations: ' + str(len(data_merged)))
print('Brokers: ' + str(len(set(list(data_merged['Broker'])))))
print('Market Centers: ' + str(len(set(list(data_merged['MarketCenter'])))))

data_merged.head()



### Data Export

## Construct Data
# Panel
data_merged[data_merged['Quarter'] == pd.Period('2017Q1')].query('Broker == "Deutsche" & OrderType == "Market" & Exchange == "NASDAQ"')

# Fixed Effects
data_merged_demeaned = data_merged - data_merged.groupby(
    ['Broker', 'MarketCenter', 'Exchange', 'OrderType']).transform("mean")

data_merged_demeaned[['Broker', 'Exchange', 'MarketCenter', 'OrderType', 'Quarter']] = data_merged[['Broker', 'Exchange', 'MarketCenter', 'OrderType', 'Quarter']]

data_merged_demeaned['Rebate_Dummy'] = data_merged['Rebate_Dummy']


## To CSV
# Panel
data_merged['Quarter'] = data_merged['Quarter'].apply(lambda x: str(x)[:4] + ' ' + str(x)[4:])
data_merged.to_csv('../../processed/regression_data_levels.csv', index=False)

# Demeaned
data_merged_demeaned.to_csv(
    '../../processed/regression_data_levels_demeaned.csv', index=False)

# 605 and 606
data_605.to_csv('../../processed/605_processed.csv', index = False)
data_606.to_csv('../../processed/606_processed.csv', index = False)

# raw
rawdata_605.to_csv('../../rawdata_605.csv', index = False)
rawdata_606.to_csv('../../rawdata_606.csv', index = False)