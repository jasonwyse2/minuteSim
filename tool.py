import pickle as pk
import os
from datetime import datetime,timedelta
import time
import pandas as pd
from dateutil.relativedelta import relativedelta
def dumppkl(data, path):
    datafile = open(path, 'w')
    pk.dump(data, datafile)
    datafile.close()
    return


def getpkl(path):
    datafile = open(path, 'r')
    ret = pk.load(datafile)
    datafile.close()
    return ret


def write_df_to_file(df, dest_dir, filename):

    fileName = ''.join([filename, '.csv'])
    full_fileName = os.path.join(dest_dir, fileName)
    if not os.path.exists(full_fileName):
        df.to_csv(full_fileName, index=False)

def append_df_to_file(df,dest_dir,filename):
    fileName_tmp = ''.join([filename+'tmp', '.csv'])
    full_fileName_tmp = os.path.join(dest_dir, fileName_tmp)
    df.to_csv(full_fileName_tmp, index=False, header = False)
    fileName = ''.join([filename, '.csv'])
    full_fileName = os.path.join(dest_dir, fileName)
    with open(full_fileName, 'ab') as f:
        f.write(open(full_fileName_tmp, 'rb').read())

    if os.path.exists(full_fileName_tmp):
        os.remove(full_fileName_tmp)

def mkdir(path):
    path = path.strip()
    # path = path.rstrip("\\")
    isExists = os.path.exists(path)
    flag = 1
    if not isExists:
        os.makedirs(path)
        flag = 0
    return flag

def currentTime_forward_delta(currentTime, min_deltaTime):
    time_format = '%Y%m%d%H%M'
    curr = datetime.strptime(currentTime, time_format)
    forward = (curr + relativedelta(minutes=+min_deltaTime))
    currTime = forward.strftime(time_format)
    return currTime


def from_timestamp13_to_localtime(timestamp):
    local_str_time = datetime.fromtimestamp(timestamp / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f')
    return str(local_str_time)[:-3]
def from_timestamp10_to_localtime(timestamp,format_str = '%Y-%m-%d %H:%M:%S'):
    local_str_time = datetime.fromtimestamp(timestamp).strftime(format_str)
    return str(local_str_time)
def get_timestamp10_minutes_ago(num):
    days_ago = (datetime.now() - timedelta(minutes=num))
    timeStamp = int(time.mktime(days_ago.timetuple()))
    return timeStamp
def get_timestamp13_from_time_str(time_str, format ='%Y-%m-%d %H:%M:%S'):
    st = time.strptime(time_str, format)
    timestamp = int(time.mktime(st))*1000
    return timestamp
def get_timestamp10_from_time_str(time_str, format ='%Y-%m-%d %H:%M:%S'):
    st = time.strptime(time_str, format)
    timestamp = int(time.mktime(st))
    return timestamp
def get_local_datetime(format_str='%Y-%m-%d %H:%M:%S'):
    timestamp10 = time.time()
    tl = time.localtime(timestamp10)
    format_time = time.strftime(format_str, tl)
    return format_time

if __name__=="__main__":

    series = pd.Series([1,2,3,4])
    se1 = series.copy()
    se1.values[1] = 10
    print(se1)