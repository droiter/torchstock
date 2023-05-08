import akshare as ak
import pandas as pd
import time
import json
import os
import shutil
import datetime
import calendar
import lstm_stock_log as log

#merge the new data to existed data in file.
def merge(stock_new_df,data_file) :
    existed_data_df=pd.read_csv(data_file,encoding='UTF-8',index_col=0)
    existed_date_list=existed_data_df["日期"].tolist()
    new_date_list=stock_new_df["日期"].tolist()
    new_df_columns_name_list=stock_new_df.columns.tolist()
    existed_df_columns_name_list=existed_data_df.columns.tolist()
    if new_df_columns_name_list != existed_df_columns_name_list :
        #data file format changed ,we can not merge it.
        return False
    to_be_deleted=[]
    for date in new_date_list:
        if date in existed_date_list:
            to_be_deleted.append(new_date_list.index(date))
    stock_new_df=stock_new_df.drop(to_be_deleted,axis=0)
    if len(stock_new_df)>0 :
        to_be_merged_df=[]
        to_be_merged_df.append(existed_data_df)
        to_be_merged_df.append(stock_new_df)
        latest_data_df=pd.concat(to_be_merged_df)
        latest_data_df.to_csv(data_file,encoding='UTF-8')
    else:
        print("data file:%s is not changed since no new data" %data_file)    
        log.output("data file:%s is not changed since no new data" %data_file) 
    return True       

#get history data for one stock and save to data file from 20000101 to current.
def get_stock_hist_data_all(stock_code,data_path):  
    current=  time.strftime("%Y%m%d", time.localtime())
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20000101", end_date=current, adjust="")
    data_file="stock-hist"+stock_code+".csv"
    #before write new data file, we should backup it.
    if(True==os.path.exists(data_path+data_file)):
        #move the current file to backup directory.
        if(True==os.path.exists('./data/backup/'+ data_file)): 
            os.remove('./data/backup/'+ data_file)
        shutil.move(data_path+data_file,'./data/backup/')
        log.output("data file:%s will be moved to backup directory" %data_file)
    stock_zh_a_hist_df.to_csv(data_path+data_file,encoding='UTF-8')
    print("stock %s downloaded in full mode." %stock_code)
    log.output("stock %s downloaded in full mode." %stock_code)
    
#get the last week data.    
def  get_stock_hist_data_latest(stock_code):
    data_file=data_path+"stock-hist"+stock_code+".csv"
    if(True==os.path.exists(data_file)):
        start_sec=time.time()-7*24*60*60 #one week for each download data oper.
        start=  time.strftime("%Y%m%d", time.localtime(start_sec))
        end=time.strftime("%Y%m%d", time.localtime())
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start, end_date=end, adjust="")
        result=merge(stock_zh_a_hist_df,data_file) #merge the new data to existed data in file.
        if result :
            print("stock %s partial downloaded and merged to existed data." %stock_code)
            log.output("Stock%s's spot data downloaded in partial mode and merged to existed data!" %stock_code)  
            return True  
        else:    
            print("data formart change as merge new data , change download mode as full mode.")
            log.output("data formart change as merge new data , change download mode as ALL.")
            os.remove(data_file)
            current=  time.strftime("%Y%m%d", time.localtime())
            stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20000101", end_date=current, adjust="")
            stock_zh_a_hist_df.to_csv(data_file,encoding='UTF-8')
            print("stock %s's data downloaded all since format changed." %stock_code)
            log.output("Stock%s's spot data downloaded in full mode since format changed!" %stock_code)    
            return False
    else:
        current=  time.strftime("%Y%m%d", time.localtime())
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20000101", end_date=current, adjust="")
        stock_zh_a_hist_df.to_csv(data_file,encoding='UTF-8')
        print("stock %s downloaded all in first time." %stock_code)
        log.output("Stock%s's spot data downloaded in full mode since maybe it is new one!" %stock_code)    
        return True
        
#scan stock list to download stock data.
def  download_data(list_file,data_path):
    stock_list_file=open(list_file,mode="r",encoding='UTF-8',errors = 'ignore') 
    json_str=stock_list_file.read()
    stock_list_file.close()
    stock_list=json.loads(json_str)
    download_all=download_all_or_not()
    for stock in stock_list:
        if(download_all):           
            get_stock_hist_data_all(stock['stockid'],data_path)
        else:
            if False==get_stock_hist_data_latest(stock['stockid']):
                download_all=True    #data format changed, we download all.
    print("%d stocks downloaded." %len(stock_list))
    log.output("%d stock data downloaded!" %len(stock_list))    
            
#generally we download data for latest week, but after some days we will download all. just now it is one week.            
def download_all_or_not():
    currentdate = datetime.date.today()
    currentday =calendar.weekday(currentdate.year,currentdate.month,currentdate.day)
    if(5==currentday):#saturday. we want download data all in saturday and calculate models in sunday.
        return True
    else:
        return True    
                    
if __name__ == '__main__' :
    data_path="./data/" #directory for downloaded data to be stored. ./bak :backup directory.
    list_file="./data/list/stock_list.json"
    download_data(list_file,data_path)