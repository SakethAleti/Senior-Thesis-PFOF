import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import os


### Functions

def load_symbol_dict(symbol_dict_loc):
    """ Loads a dictionary to reduce operations to find the exchange of a 
    symbol to O(1) instead of O(n). The csv is contained in 
    ../ExchangeData/symbols_df
    """
    symbol_df = pd.read_csv(symbol_dict_loc)
    symbol_df.index = symbol_df['Symbol']
    symbol_df = symbol_df.drop('Symbol', axis = 1)
    return symbol_df.to_dict()['Exchange']

def aggregate_605MktCtrMonth_data(data, symbol_dict):
    """ Aggregates 605 data by exchange
    """

    ordertype_dict = {11: 'Market', 12: 'Limit', 13: 'Other', 
        14: 'Other', 15: 'Other'}

    # New Variables
    data['Exchange']   = data['Symbol'].apply(lambda x: 
        symbol_dict.get(x, "Other"))
    data['ExecShares'] = data['MktCtrExecShares'] + data['AwayExecShares']

    # Temporary Variables for Aggregation
    data['PrImp_TotalT']     = data['PrImpShares']    * data['PrImp_AvgT']
    data['PrImp_TotalAmt']   = data['PrImpShares']    * data['PrImp_AvgAmt']
    data['ATQ_TotalT']       = data['ATQShares']      * data['ATQ_AvgT']
    data['OTQ_TotalT']       = data['OTQShares']      * data['OTQ_AvgT']
    data['AvgRealSpread_T']  = data['AvgRealSpread']  * data['ExecShares']
    data['AvgEffecSpread_T'] = data['AvgEffecSpread'] * data['ExecShares']

    # Aggregate data by exchange
    data = data.query('SizeCode == 21') \
        .groupby(['MarketCenter', 'idate', 'Exchange', 'OrderCode']) \
        .sum().reset_index()

    # Reconstruct original variables
    data['PrImp_AvgT']     = data['PrImp_TotalT']     / data['PrImpShares']
    data['PrImp_AvgAmt']   = data['PrImp_TotalAmt']   / data['PrImpShares'] 
    data['ATQ_AvgT']       = data['ATQ_TotalT']       / data['ATQShares']
    data['OTQ_AvgT']       = data['OTQ_TotalT']       / data['OTQShares']
    data['AvgRealSpread']  = data['AvgRealSpread_T']  / data['ExecShares']
    data['AvgEffecSpread'] = data['AvgEffecSpread_T'] / data['ExecShares']

    # New varaibles 
    data['PrImp_Pct'] = data['PrImpShares'] / data['ExecShares']
    data['ATQ_Pct']   = data['ATQShares']   / data['ExecShares']
    data['OTQ_Pct']   = data['OTQShares']   / data['ExecShares']
    data['OrderType'] = data['OrderCode'].apply(lambda x: ordertype_dict[x])

    # Clean up non-sensical vars
    data = data.drop(['SizeCode'], axis = 1)

    return data


### Parameters

# names of columns of 605 data (https://www.sec.gov/interps/legal/slbim12b.htm)
column_names_605 = {1: 'MarketCenter', 2: 'idate', 3: 'Symbol', 
    4: 'OrderCode', 5: 'SizeCode', 6: 'CoveredOrders', 
    7: 'CoveredShares', 8: 'CancelledShares', 9: 'MktCtrExecShares', 
    10: 'AwayExecShares', 11: 'ExecShares_0_9', 
    12: 'ExecShares_10_29', 13: 'ExecShares_30_59', 
    14: 'ExecShares_60_299', 15: 'ExecShares_5_30', 
    16: 'AvgRealSpread', 17: 'AvgEffecSpread', 18: 'PrImpShares',
    19: 'PrImp_AvgAmt', 20: 'PrImp_AvgT', 21: 'ATQShares', 
    22: 'ATQ_AvgT', 23: 'OTQShares', 24: 'OTQ_AvgAmt', 
    25: 'OTQ_AvgT'}

# 605 csv directory
dir_605 = '../../605/'


### Main

# Dictionary of symbols to find their exchanges
symbol_dict = load_symbol_dict('../../keys/symbols.csv')


# IO
input_command = input('Enter "all" or a list of market centers to update: ')

if input_command == "all":

    # Returns folders in the 605 directory
    dir_605_folders = [x for x in os.walk(dir_605)][0][1]

else:

    dir_605_folders = input_command.replace(' ', '').split(',')

print([x for x in os.walk(dir_605)][0][1])
print(dir_605_folders)

# For each market center (folder)
for folder in dir_605_folders:
    
    print('\nReading data for ' + folder + '...')

    # Get .csv file names
    file_names = [x for x in os.walk(dir_605 + folder)][0][2]
    file_locations = [dir_605 + folder + '/' + x for x in file_names]

    data_605_mktctr_list = []

    # Progress bar
    pbar = tqdm(total = len(file_names))

    # for each month
    for file in file_locations:

        # read csv
        data_605_mktctr_month = pd.read_csv(file, 
            sep = '|', header = None)

        # update column names
        data_605_mktctr_month = data_605_mktctr_month.rename(
            columns = column_names_605).drop(0, axis = 1)

        # append aggregated data
        data_605_mktctr_list.append(
            aggregate_605MktCtrMonth_data(data_605_mktctr_month, symbol_dict))

        # update progress bar
        pbar.update(1)

    # close progress bar
    pbar.close()

    # merge aggregated data into one data frame
    data_605_mktctr = pd.concat(data_605_mktctr_list)

    # update market center label
    data_605_mktctr['MarketCenter'] = folder

    # save data to csv
    data_605_mktctr.to_csv(dir_605 + folder + '.csv', index = False)

print("\nComplete")