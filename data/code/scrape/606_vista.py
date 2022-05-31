import datetime
from dateutil.rrule import rrule, MONTHLY
import zipfile
import urllib.request
import os
import glob
import pandas as pd
from lxml import etree
from tqdm import tqdm


### Functions

def convert_quarter_to_month(quarter):
    """ Converts a quarter (int) to month in MM (str)
    """

    if int(quarter)*3 != 12:
        month = '0'  + str(int(quarter)*3)
    else:
        month = '12'
    return month

def read_HTML_table(s, exchange):
    """ Reads an HTML table as a list of lists and appends a string (exchange)
    to the end of each list of lists
    """

    data = []
    
    # Parse HTML table into a list of lists
    table = etree.HTML(s).find("body/table")
    rows = iter(table)
    headers = [col.text for col in next(rows)]
    
    for row in rows:
        values = [col.text for col in row]
        values.append(exchange)
        # Ignore rows starting with total
        if 'Total' not in values[0]:
            data.append(values)

    return data

def download_vista606(broker, broker_id, year, quarter, rebate):
    """ Downloads 606 data for a given broker and date and formats it into
    a pandas dataframe
    """

    # Download html
    url = 'https://vrs.vista-one-solutions.com/data/' + broker_id + '/' \
        + broker_id + '_' + year + '_' + quarter + '.html'
    response = urllib.request.urlopen(url)

    # Convert to string
    raw = response.read()
    rawstr = raw.decode('utf-8')

    ## Parse data

    exchanges = {1: 'NYSE', 2: 'NASDAQ', 3: 'Other'}
    
    if len(rawstr) > 1500:

        # Break html file into peices for each table
        tables = rawstr.split('<table BORDER=2 BORDERCOLOR="black">')

        data = []

        # Read the 3 tables
        for i in range(1,4):

            data.append(read_HTML_table('<table BORDER=2 BORDERCOLOR="black">' 
                + tables[i], exchanges[i]))

        # flatten list of lists
        data = sum(data, [])

        # Convert list data into a dataframe
        data_df = pd.DataFrame(data).rename(
            columns = {0: 'RoutingVenue', 1: 'Total', 2: 'Market', 3: 'Limit', 
            4: 'Other', 5: 'Exchange'})

    else:
        
        return pd.DataFrame()
    
    ## Clean up dataframe

    data_df = data_df.melt(id_vars = ['RoutingVenue', 'Exchange'], 
        value_name = 'Pct', var_name = 'OrderType')
    data_df['Broker'] = broker

    month = convert_quarter_to_month(quarter)
    data_df['Date'] = year + month

    data_df['Pct'] = data_df['Pct'].apply(lambda x: str(round(float(x)/100, 4)))

    data_df['Rebate'] = rebate

    return data_df


### Params

dir_606 = '../../606/'

## IO
broker = input('Enter broker name: ')

broker_id = input('Enter broker ID from \n'
    + 'view-source:https://vrs.vista-one-solutions.com/public1_6.aspx\n')
year = int(input('Enter starting year: '))

rebate = int(input('Does the firm accept (0 or 1) rebates? '))

### Main

print('\nDownloading 606 Data...')

# Missing data list
missing_list = []

# Progress bar
pbar = tqdm(total = ((2018-year)*(4)))

# Download, format, and save 606 data
for year in range(year,2018):

    for quarter in range(1,5):
        
        save_loc = (dir_606 + broker + '/' + str(year) 
                    + convert_quarter_to_month(str(quarter)) + '.csv')
        
        # Get 606 dataframe for quarter
        data_606_temp = download_vista606(broker, broker_id, 
            str(year), str(quarter), rebate)
        
        # Save as .csv
        if len(data_606_temp) > 0: 
            
            data_606_temp.to_csv(save_loc, index = False)
        
        else:

            missing_list.append(str(year) + 'Q' + str(quarter))

        # Update progress bar
        pbar.update(1)

pbar.close()

## Print missing data
if missing_list:
    print('\nNo data for ' + broker + ' in ', end = '')
    print(','.join(missing_list))    
     
print('\nComplete\n')