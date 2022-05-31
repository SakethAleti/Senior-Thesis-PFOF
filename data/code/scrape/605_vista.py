import datetime
from dateutil.rrule import rrule, MONTHLY
import zipfile
import urllib.request
import os
import glob
from tqdm import tqdm


### Functions

def get_year_month(date):
    """ Converts a datetime object to YYYYMM 
    """
    year = str(date.year)
    month_num = date.month
    if month_num < 10:
        month = str(0) + str(month_num)
    else:
        month = str(month_num)
    return year + month 

def get_vista605_url(firm, mpid, yearmonth):
    """ Gets url for 605 data hosted by Vista 
    """
    return 'https://vrs.vista-one-solutions.com/DataFiles/' \
        + firm + '/' + mpid + yearmonth + '.zip'


### Params

# 605 csv directory
dir_605 = '../../605/'

# temporary directory to hold .zip files
dir_temp = '../../temp/'

## IO

# Label used by Vista 605 to organize data
firm = input('Enter firm label found at ' 
    + 'view-source:https://vrs.vista-one-solutions.com/sec605rule.aspx\n')

mpid = input('Enter firm MPID: ')

firm_folder = input('Enter firm folder name: ')

strt_yr = input('Enter starting year: ')
strt_mn = input('Enter starting month: ')
strt_dt = datetime.date(int(strt_yr), int(strt_mn), 1)

# takes data until most recent date
end_dt  = datetime.date(2018, 1, 1)


### Main

dates = [dt for dt in rrule(MONTHLY, dtstart=strt_dt, until=end_dt)][::-1]
download_urls = [get_vista605_url(firm, mpid, get_year_month(date)) 
    for date in dates]

## Download all files to temp directory and extract to proper directory

extract_directory = dir_605 + firm_folder + '/'
url_errors = []

print('\nExtracting... ')

# Progress bar
pbar = tqdm(total = len(download_urls))
      
for url in download_urls:

    save_file = dir_temp + url[-10:]

    try:

        # download data
        urllib.request.urlretrieve(url, save_file)
        
        # unzip file
        zip_ref = zipfile.ZipFile(save_file, 'r')
        zip_ref.extractall(extract_directory)
        zip_ref.close()
    
    except:
        
        url_errors.append(url)

    # update progress bar
    pbar.update(1)
        
pbar.close()

## Manage download errors

error_dates = []

if url_errors: 

    # Delete those files
    error_filelocations = [(dir_temp + x[-10:]) for x in url_errors]
    [os.remove(x) for x in error_filelocations];
    
    # Print files which threw errors when downloading
    print('\nError getting data for ', end = '') 
    error_dates = [x[-10:-4] for x in error_filelocations]
    print(', '.join(error_dates))

## Rename files as csv's

current_filenames = [x[(-10-len(mpid)):-4] + '.dat' for x in download_urls]
current_filelocations = [extract_directory + x for x in current_filenames]

for file in current_filelocations:
    
    try: 

        filename_noext = os.path.splitext(file)[0]
        file_newname   = filename_noext[-6:] + '.csv'
        os.rename(file, dir_605 + '/' + firm_folder + '/' + file_newname)
    
    except:

        # for files that were properly downloaded but not renamed
        if file[-10:-4] not in error_dates: print('Error renaming ' + file)
        continue

## Remove temporary files

print('Removing temporary files...')

files = glob.glob(dir_temp + '*')
for f in files:
    os.remove(f)



print('Complete')