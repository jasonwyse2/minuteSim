# encoding:utf-8
from datetime import datetime
import time
import pandas as pd
import numpy as np
from tool import *
from indicator import Indicator
import tool
from const import quantity_volume
from os import walk
import re
import copy

class MinuteSim():
    startDate='20180103'
    endDate = '20180907'
    basicDatetime_date = '20180103'
    tradeDate_baseDir = '/data2/ctasim/data2'
    saveData_dir = './data'
    fields = ['open', 'high', 'low', 'close', 'volume', 'twap', 'oi',
              'increment', 'amount', 'volume', 'openposition',
              'closeposition', 'directionBuy', 'directionSell','vwap']

    def __init__(self):
        self.universeInstruments_array = self.__get_universe_instruments()
        self.allTradeDate_array = self.__get_allTradeDate()
        self.basic_timestamp_offset_index = self.__get_basic_timestamp_offset_index()

    def __get_allTradeDate(self):
        for (dirpath, dirnames, filenames) in walk(self.tradeDate_baseDir):
            dirnames = np.sort(dirnames)
            self.allTradeDate_array = np.array(dirnames)
            break
        return self.allTradeDate_array

    def __get_universe_instruments(self, file_path='c1_for_zl.csv'):
        df = pd.read_csv(file_path)
        columns = df.columns.values[1:] # the first field is 'date', remove it away
        cols = np.sort(list(set(columns)))
        return np.array(cols)

    def __get_basic_timestamp_offset_index(self):
        date = self.basicDatetime_date
        def get_fileName_by_dateSymbol(date, symbol='au'):
            date_dir = os.path.join(self.tradeDate_baseDir, date)
            for (dirpath, dirnames, filenames) in walk(date_dir):
                fileName_list = filenames
                break
            for i in range(len(fileName_list)):
                if symbol in fileName_list[i]:
                    fileName = fileName_list[i]
                    break
            return fileName

        fileName = get_fileName_by_dateSymbol(date,symbol='au')
        full_fileName = os.path.join(self.tradeDate_baseDir,date,fileName)
        df = pd.read_csv(full_fileName)
        datetime_list = list(df['time'])
        date_ts10 = get_timestamp10_from_time_str(date, format='%Y%m%d')
        date_str = from_timestamp10_to_localtime(date_ts10, format_str='%Y-%m-%d')
        datetime_auction = date_str+' 08:59:00'# 'au'的数据中缺08:59:00,该时间是集合竞价成交
        if datetime_auction not in datetime_list:
            datetime_list.append(datetime_auction)
        datetime_max = date_str+' 15:00:00' # 文件包含15:17:00 的数据，没有意义,将分钟时刻最大交易时间设置为15:00:00
        datetime_array = np.array(datetime_list)
        datetime_array1 = np.sort(datetime_array[datetime_array<=datetime_max])
        # 将获取的标准一天所包含的分钟，转化为相对的时间戳
        ts_series = pd.Series(datetime_array1).apply(lambda x:get_timestamp10_from_time_str(x))
        basic_timestamp_offset_index = ts_series.values-date_ts10
        return basic_timestamp_offset_index

    def __get_datetimeIndex_by_timestampOffset(self,date):
        date_ts10 = get_timestamp10_from_time_str(date, format='%Y%m%d')
        ts_index = date_ts10+self.basic_timestamp_offset_index
        datetime_index = pd.Series(ts_index).apply(from_timestamp10_to_localtime)
        return datetime_index.values
    def load_data(self,startDate='20180103',endDate='20180228'):
        selectedDate_array = self.allTradeDate_array[(self.allTradeDate_array>=startDate)&(self.allTradeDate_array<=endDate)]
        return selectedDate_array

    def get_mainContractCodeArray_by_date(self, date, file_path ='c1_for_zl.csv'):
        df = pd.read_csv(file_path)
        date_series = df['date'].astype(int)
        mainContract_array = df[date_series==int(date)].dropna(axis=1).values
        mainContract_array = mainContract_array[:,1:].flatten() # remove 'date' colume
        return mainContract_array

    def get_datetime_index(self,date):
        def get_fileName_by_dateSymbol(date, symbol='au'):
            date_dir = os.path.join(self.tradeDate_baseDir, date)
            for (dirpath, dirnames, filenames) in walk(date_dir):
                fileName_list = filenames
                break
            for i in range(len(fileName_list)):
                if symbol in fileName_list[i]:
                    fileName = fileName_list[i]
                    break
            return fileName

        fileName = get_fileName_by_dateSymbol(date,symbol='au')
        full_fileName = os.path.join(self.tradeDate_baseDir,date,fileName)
        df = pd.read_csv(full_fileName)
        datetime_list = list(df['time'])
        date_ts10 = get_timestamp10_from_time_str(date, format='%Y%m%d')
        date_str = from_timestamp10_to_localtime(date_ts10, format_str='%Y-%m-%d')
        datetime_auction = date_str+' 08:59:00'# 'au'的数据中缺08:59:00,该时间是集合竞价成交
        if datetime_auction not in datetime_list:
            datetime_list.append(datetime_auction)
        datetime_max = date_str+' 15:00:00' # 文件包含15:17:00 的数据，没有意义,将分钟时刻最大交易时间设置为15:00:00
        datetime_array = np.array(datetime_list)
        datetime_array1 = np.sort(datetime_array[datetime_array<=datetime_max])
        # 如果某一天黄金(au)的交易分钟数不是标准分钟长度(一般出现在放长假后的第一天),则人为生成当天标准交易分钟长度
        if len(datetime_array1)<len(self.basic_timestamp_offset_index):
            datetime_array2 = self.__get_datetimeIndex_by_timestampOffset(date)
            return datetime_array2
        else:
            return datetime_array1

    def get_sectionalData_dict(self,date):
        sectionalData_dict = {}
        fields = self.fields
        for i in range(len(fields)):
            field = fields[i]
            sectionalData_dict[field] = self.get_sectionalData_by_field(date, field)
        return sectionalData_dict

    def get_allDate_sectionalData_dict(self):
        sectionalData_allDate_dict = {}
        for i in range(len(self.allTradeDate_array)):
            date = self.allTradeDate_array[i]
            start = time.clock()
            sectionalData_dict = self.get_sectionalData_dict(date)
            end = time.clock()
            print(date,'time elapsed:',end-start)
            sectionalData_allDate_dict[date] = sectionalData_dict
        return sectionalData_allDate_dict

    def get_sectionalData_date_dict(self, startDate, endDate):
        self.startDate,self.endDate = startDate, endDate
        sectionalData_date_dict={}
        intervalDate_array = self.allTradeDate_array[(self.allTradeDate_array>=startDate)&(self.allTradeDate_array<=endDate)]
        for i in range(len(intervalDate_array)):
            date = intervalDate_array[i]
            start = time.clock()
            sectionalData_dict = self.get_sectionalData_dict(date)
            end = time.clock()
            print(date,'time elapsed:',end-start)
            sectionalData_date_dict[date] = sectionalData_dict
        return sectionalData_date_dict

    def get_sectionalData_by_field(self, date, field):
        datetime_index_array = self.get_datetime_index(date)
        shape = (len(datetime_index_array), len(self.universeInstruments_array))
        sectional_array = np.full(shape, np.nan)
        sectional_df = pd.DataFrame(sectional_array, index=datetime_index_array, columns=self.universeInstruments_array)
        mainContractFilename_array = self.get_mainContractFilename_array(date)
        ts10 = get_timestamp10_from_time_str(date,format='%Y%m%d')
        date_str = from_timestamp10_to_localtime(ts10,format_str='%Y-%m-%d')
        datetime_max = date_str + ' 15:00:00'

        for i in range(len(mainContractFilename_array)):
            fileName = mainContractFilename_array[i]
            symbol = self.get_instrument_symbol(fileName)
            quantity = quantity_volume[symbol]
            full_fileName = os.path.join(self.tradeDate_baseDir,date,fileName)
            # df_inst 某一个品种的tick数据，已经被处理成分钟数据，但不同品种的交易时间不同，需要做时间对齐
            df_inst = pd.read_csv(full_fileName,index_col=0)
            df_inst['vwap'] = pd.Series(df_inst['amount'].values / (df_inst['volume'].values*quantity)) # 生成vwap
            tmp = df_inst[['twap','vwap']] #用来调试 观察变量
            df_inst.index = df_inst['time']

            df_inst1 = df_inst[df_inst['time']<=datetime_max]

            # 将某一品种的实际交易分钟与该交易日的标准交易分钟做一个交集并且排序
            datetime_array = np.sort(list(set(df_inst1['time']).intersection(set(datetime_index_array))))
            # 将交集后的分钟数据赋值到标准大小的dataframe中，确保每一个交易日都是等长的
            sectional_df.loc[datetime_array,symbol] = df_inst1.loc[datetime_array,field]

        return sectional_df

    def get_instrument_symbol(self,fileName):
        pattern_str = '^\d+\_([a-zA-Z]+)(\d+)'
        tmp1 = re.match(pattern_str, fileName).groups()
        symbol = tmp1[0]
        return symbol

    def get_adjustCoefficient_df(self,file_path ='c1_for_zl.csv'):
        def get_full_fileName(date, contract):
            fileName = date + '_' + contract + '.csv'
            full_fileName = os.path.join(self.tradeDate_baseDir, date, fileName)
            return full_fileName

        shape = (len(self.allTradeDate_array), len(self.universeInstruments_array))
        sectional_array = np.full(shape, np.nan)
        sectional_df = pd.DataFrame(sectional_array, index=self.allTradeDate_array, columns=self.universeInstruments_array)

        df = pd.read_csv(file_path)
        df['date']=df['date'].astype(int).astype(str)
        df1 = df[list(self.universeInstruments_array)]
        columns = df1.columns

        df_shape = df1.shape
        for i in range(1,df_shape[0]): # rows
            for j in range(0,df_shape[1]): # columns
                if isinstance(df1.iloc[i-1,j], str) and isinstance(df1.iloc[i,j], str) and df1.iloc[i,j]!=df1.iloc[i-1,j]:
                    date = df['date'].loc[i]
                    mainContract_full_fileName = get_full_fileName(date, df1.iloc[i, j])
                    submainContract_full_fileName = get_full_fileName(date, df1.iloc[i - 1, j])
                    main_df = pd.read_csv(mainContract_full_fileName)
                    submain_df = pd.read_csv(submainContract_full_fileName)
                    submain_close = submain_df['close'].values[-1]
                    main_close = main_df['close'].values[-1]
                    sectional_df.loc[date,columns[j]] = float(submain_close/main_close)

        return sectional_df

    def get_mainContractFilename_array(self, date):
        date = str(date)
        mainContract_array = self.get_mainContractCodeArray_by_date(date) # 通过读取c1_for_zl.csv文件 获取主力合约
        mainContractFilename_array = np.copy(mainContract_array)
        for i in range(len(mainContract_array)):
            mainContractFilename_array[i]=date+'_'+mainContract_array[i]+'.csv'
        return mainContractFilename_array

    def get_continueSectionalData_field_dict(self, sectionalData_date_dict):
        date_array = np.sort(sectionalData_date_dict.keys())
        fields = self.fields
        sectional_df_dict = {}
        # 新建用于存放横截面数据的各个dataframe
        for field in fields:
            sectional_df_dict[field] = pd.DataFrame(columns=self.universeInstruments_array)
        # 遍历每个日期，每个日期里包含['close','open',...,]等横截面数据
        for i in range(len(date_array)):
            sectionalData_dict = sectionalData_date_dict[date_array[i]]
            # 将每个field里的横截面数据进行汇总，拼接成连续数据
            for j in range(len(fields)):
                sectional_df_dict[fields[j]] = pd.concat([sectional_df_dict[fields[j]], sectionalData_dict[fields[j]]])
        return sectional_df_dict

    def save_continueSectionalData_field_dict(self, sectionalData_field_dict):
        saveData_dir = self.saveData_dir
        hdfFile = self.startDate+'_'+self.endDate+'.h5'
        full_hdfFile = os.path.join(saveData_dir, hdfFile)

        fields = np.sort(sectionalData_field_dict.keys())
        for field in fields:
            sectional_df = sectionalData_field_dict[field]
            sectional_df.to_hdf(full_hdfFile,key=field,mode='a')

    def get_sectionalDataDict_from_hdf(self, startDate, endDate,key=''):
        saveData_dir = self.saveData_dir
        hdfFile = startDate + '_' + endDate + '.h5'
        full_hdfFile = os.path.join(saveData_dir, hdfFile)
        fields = self.fields
        sectionalData_dict = {}
        if key=='':
            for field in fields:
                sectionalData_dict[field] = pd.read_hdf(full_hdfFile, field)
            return sectionalData_dict
        else:
            sectionalData = pd.read_hdf(full_hdfFile, key)
            return sectionalData

    def save_adjustCoefficient_df(self,key,adjustCoefficient_df):
        saveData_dir = self.saveData_dir
        hdfFile = self.startDate + '_' + self.endDate + '.h5'
        full_hdfFile = os.path.join(saveData_dir, hdfFile)
        adjustCoefficient_df.to_hdf(full_hdfFile, key=key, mode='a')

    def get_continueAdjustSectionalData_field_dict1(self,sectionalData_dict,adjustCoefficient_minute_array):
        fields = ['open', 'high', 'low', 'close', 'twap', 'vwap']
        sectionalData_dict_copy = copy.deepcopy(sectionalData_dict)
        for field in fields:
            tmp = sectionalData_dict_copy[field].values
            sectionalData_dict_copy[field].loc[:,:] = (sectionalData_dict_copy[field].values)*adjustCoefficient_minute_array
        return sectionalData_dict_copy

    # def get_continueAdjustSectionalData_field_dict(self, sectionalData_dict, adjustCoefficient_df):
    #     instrument_array = self.universeInstruments_array
    #     sectionalData_dict_copy = copy.deepcopy(sectionalData_dict)
    #     # sectionalData_dict_copy = (sectionalData_dict)
    #     # date_array = self.allTradeDate_array
    #     fields = ['open','high','low','close','twap','vwap']
    #     datetime_index = self.get_datetime_index(date=self.basicDatetime_date)
    #     minuteBar_len = len(datetime_index)
    #     for i in range(len(instrument_array)): #遍历所有的品种
    #         instrument = instrument_array[i]
    #         for field in fields:
    #             sectional_df = sectionalData_dict_copy[field]
    #             coeff_array = adjustCoefficient_df[instrument].values
    #             coeff_index_array = np.where(~ np.isnan(coeff_array))[0]
    #             for j in range(len(coeff_index_array)):
    #                 coeff_index = coeff_index_array[j]
    #                 tmp1 = coeff_array[coeff_index]
    #                 coeff_offset = coeff_index*minuteBar_len
    #                 tmp2 = sectional_df[instrument].values[coeff_offset:]
    #                 sectional_df.loc[coeff_offset:,instrument] = tmp2*tmp1
    #                 pass
    #
    #     return sectionalData_dict_copy
    def rescale_coefficient_array(self,adjustCoefficient_df):
        basicMinutes_num = len(self.basic_timestamp_offset_index)
        total_minute_num = len(adjustCoefficient_df.index)*basicMinutes_num
        shape = (total_minute_num,len(adjustCoefficient_df.columns))
        coefficient_array = np.ones(shape)
        # coefficient_df = pd.DataFrame(coefficient_array,columns=)
        for i in range(len(adjustCoefficient_df.columns)):
            instrument = adjustCoefficient_df.columns[i]
            coeff_array = adjustCoefficient_df[instrument].values
            coeff_index_array = np.where(~ np.isnan(coeff_array))[0]
            for j in range(len(coeff_index_array)):
                coeff_index = coeff_index_array[j]
                coeff = coeff_array[coeff_index]
                coefficient_array[basicMinutes_num*coeff_index:,i]=coefficient_array[basicMinutes_num*coeff_index:,i]*coeff

        return coefficient_array



if __name__ == '__main__':
    sim = MinuteSim()

    start = time.clock()
    # 20180103 20180222
    startDate = '20180103'#根据 startDate和 endDate指定的区间范围来确定所要读取的文件夹
    endDate = '20180907' # 20180907, 20180104

    # 生成连续横截面数据，并保存到 hdf文件中，
    # startDate,endDate 用来确定交易日范围，这两个日期可以不是交易日；这两个日期还会用来确定hdf文件的文件名
    # sectionalData_date_dict = sim.get_sectionalData_date_dict(startDate, endDate)
    # continueSectionalData_field_dict = sim.get_continueSectionalData_field_dict(sectionalData_date_dict)
    # sim.save_continueSectionalData_field_dict(continueSectionalData_field_dict)

    # 生成除权系数并保存,只需要生成保存一次即可.该dataframe的index是date，columns是instruments
    # adjustCoefficient_date_df = sim.get_adjustCoefficient_df()
    # sim.save_adjustCoefficient_df('adjustCoeff',adjustCoefficient_date_df)

    # startData, endDate 用来生成 hdf文件名。 key为空的话，默认返回所有的横截面数据；如果key不为空的话，返回指定数据
    continue_sectionalData_dict = sim.get_sectionalDataDict_from_hdf(startDate, endDate)
    adjustCoefficient_date_df1 = sim.get_sectionalDataDict_from_hdf(startDate, endDate, key='adjustCoeff')
    # 生成与横截面数据大小完全相同的复权系数矩阵
    adjustCoefficient_minute_array = sim.rescale_coefficient_array(adjustCoefficient_date_df1)
    adjustCoefficient_minute_df = pd.DataFrame(adjustCoefficient_minute_array,
                                               index=continue_sectionalData_dict['close'].index
                                               , columns=continue_sectionalData_dict['close'].columns)

    # 根据横截面数据以及复权系数矩阵，对横截面数据进行复权
    # continueAdjust_sectionalData_dict = sim.get_continueAdjustSectionalData_field_dict(continue_sectionalData_dict,
    #                                                                                    adjustCoefficient_date_df1)
    continueAdjust_sectionalData_dict1 = sim.get_continueAdjustSectionalData_field_dict1(continue_sectionalData_dict,
                                                                                         adjustCoefficient_minute_array)
    # df_adjust = continueAdjust_sectionalData_dict['close']

    end = time.clock()
    elapsed = (end - start)
    print('get adjust coefficient df total elapsed time:', elapsed)
    rows=[1052,15425,52052,68452]
    cols =[25,4,11,30]
    for i in rows:
        for j in cols:
            tmp1 = continueAdjust_sectionalData_dict['close'].iloc[i, j]
            tmp2 = continueAdjust_sectionalData_dict1['close'].iloc[i, j]
            tmp3 = continue_sectionalData_dict['close'].iloc[i, j]
            pass

    pass




