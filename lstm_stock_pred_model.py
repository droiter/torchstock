import sys
import os
import time
import lstm_stock_log as log

def get_stockid(data_file_name):
    stockid=data_file_name[-10:-4] 
    return stockid
#walk ./data directory to find stock list whose model to be calculated.
def get_stockid_list():
    stocklist=[]
    layer=0
    for root,dir,files in os.walk("./data"):    
        if layer>0: break    
        for file_name in files: 
            stockid=get_stockid(file_name) 
            stocklist.append(stockid) 
        layer=layer+1    
    return stocklist

def calc_model(pred_type,nextday):
    stocklist=get_stockid_list()
    if len(stocklist)>0:
        for stockid in stocklist:
            cmd='python'+' ' + 'lstm_stock_pred_multi_ms.py' +' '+'-sid' + ' ' + stockid +' '+'-predtype'+' '+ pred_type +' '+'-nextday' + ' ' + str(nextday) +' '+'-plot' + ' ' + 'False'
            print(cmd)
            os.system(cmd)
            log.output("Stock%s's lstm model calculated in nextday:%d by issue cmd:%s" %(stockid,nextday,cmd))    
    else:
        print("can not find any data file in directory: ./data")
        log.output("can not find any data file in directory: ./data!", level=1)
    log.output("%d stock's lstm model calculated!" %len(stocklist))  
    return len(stocklist)  
        
#the main procedure.   
if __name__ == '__main__' :
    run_time=time.time() 
    num_model=0;
    #for pred_type in ["open","close","high","low"]:
    for pred_type in ["open","close","all"]:
        for nextday in range(1,4): #defaultly nextday1,2,3 will be predicted, so three model for nextday1,2,3 will be calcaulated.
            num_model+=calc_model(pred_type,nextday)
    elapse_time=time.time()-run_time;
    print("%d models calculated and %0.2f seconds used." %(num_model,elapse_time))                            
    log.output("%d models calculated and %0.2f seconds used." %(num_model,elapse_time))                            
           