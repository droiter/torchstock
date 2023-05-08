import numpy as np 
import torch
from torch import device, nn
import pandas as pd 
from torch.utils.data import DataLoader
import torch.utils.data as Data
from tqdm import tqdm, trange
from torch.optim import lr_scheduler
from itertools import chain
from scipy.interpolate import make_interp_spline
import json
import os
import argparse
import time
import sys
import lstm_stock_log as log
   
#cmd line parmeters.
def cmd_line():
    parser = argparse.ArgumentParser(description='Predict stock price according to stockid')
    parser.add_argument('-sid', '--stockid', type=str, help='Stock id')
    cmd_args = parser.parse_args() 
    stock_id=cmd_args.stockid
    return stock_id
    
#get args.
def get_args():
    return args;

def get_data_maxmin(index):
    if index > len(data_mm):
        return data_mm[:args["output_size"]]
    else:    
        mm=data_mm[index-data_col_bypass] #bypass tble head  in dataframe.
        return mm

#load data from csv file and clean it.
def load_data(file_name):
    missing_values = ["n/a", "na", "--","None"]
    df=pd.read_csv(file_name,encoding='UTF-8',na_values = missing_values,index_col=0)
    columns=df.columns
    columns_want=['日期', '开盘', '收盘', '最高','最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    columns_to_be_deleted=[]
    for column in columns  :
        if( column not in columns_want):
            columns_to_be_deleted.append(column)
    df=df.drop(columns_to_be_deleted,axis=1) #data format is changed in sometime we will remove some no useful columns.
    print("dataFrame:",df.shape)
    df.dropna(axis='index', how='any',inplace=True)
    #reverse
    #df=df.reindex(index=df.index[::-1]) 
    #print(df)   
    return df 

#create our dataset.
class MyDataset(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
    
# Create dataset.   
def process(data, batch_size, shuffle,data_index):
    args=get_args()
    data1=data.iloc[:,data_col_bypass:] #remove header from original dataframe.
    load = data1.to_numpy() 
    #normalize
    for i in range(load.shape[1]):
        max,min=np.max(load[:,i]),np.min(load[:,i])
        load[:,i] = (load[:,i]-min) / (max-min)
    
    seq = []
    seq_len=get_args()["seq_len"]
    i =(len(load) - seq_len)
    train_seq = []
    for j in range(i, i + seq_len):
        x=[]
        for k in data_index:
            x.append(load[j][k])
        train_seq.append(x)
    train_seq = torch.FloatTensor(train_seq)    
    seq.append((train_seq)) 
    
    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return seq

# split date and create datasets for train /validate and test.   
def nn_data_seq(batch_size):
    print('data processing...')
    #test the data file exit or not.
    if(False==os.path.exists(data_file_name)): 
        print("can not find data file:%s" %data_file_name)
        log.output("can not find data file:%s" %data_file_name,level=1)
        sys.exit(0)      
        
    dataset = load_data(data_file_name)
    for i in range(data_col_bypass,dataset.shape[1]):
        m, n = np.max(dataset[dataset.columns[i]]), np.min(dataset[dataset.columns[i]])
        mm={}
        mm['max']=m
        mm['min']=n
        data_mm.append(mm)    
    #dataset.
    Dte = process(dataset,  batch_size, False,data_index_set)
    return Dte

#nn module.
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0)) # output(5, 30, 64)
        pred = self.linear(output)  # (5, 24, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred
    
#Bidirection nn module.
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.num_directions * self.hidden_size, self.output_size)
    	
    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)  # pred()
        pred = pred[:, -1, :]        
        return pred	
    

#predict sotck price
def test(args, Dte):
    args=get_args()
    input_size, hidden_size, num_layers = args['input_size'], args['hidden_size'], args['num_layers']        
    log.output("Stock%s is queried!" %stock_id)    
    info_out=[]    
    #prepare the file list for module files.
    layer=0
    for root,dir,files in os.walk("./model"): 
        if(layer>0): break       
        for file_name in files:   
            if (file_name.find('.pkl')>0 and file_name.find(stock_id)>0) :
                #get model info from model json file.
                json_file_name=file_name.replace('.pkl','.json')
                mape=0
                nextday=0
                pred_type="close"
                if json_file_name in files:
                    with open('./model/'+json_file_name,'r') as json_file:
                        json_str=json_file.read()
                        dict_str=json.loads(json_str)
                        mape=dict_str['mape']
                        nextday=dict_str['multi_steps']
                        pred_type=dict_str["pred_type"]
                #set data_pred_index according to pred_type
                if pred_type in ["open","close","high","low","all"]:
                    args['pred_type']=pred_type
                    pred_type_option={"open":1,"close":2,"high":3,"low":4,"all":4000} #all: single model for all predtype: open/close/high/low.
                    data_pred_index=pred_type_option[pred_type]
                if pred_type =="all" :
                    args["output_size"]=4
                else:
                    args["output_size"]=1
                output_size = args['output_size']
                #define model. since for each model, the output_size maybe difference, so we put it here.
                if args['bidirectional']:
                    model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args['batch_size']).to(device)
                else:
                    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args['batch_size']).to(device)
                        
                info_item={}
                if args['pred_type']=="all":
                    pred=np.empty(shape=(0,4))
                else:
                    pred=np.empty(shape=(0,1))
                path='./model/'+file_name
                print('loading models...')
                model.load_state_dict(torch.load(path)['models'])    
                model.eval()
                print('predicting for %s' %file_name)
                for (seq) in tqdm(Dte):
                    seq = seq.to(device)
                    with torch.no_grad():
                        y_pred = model(seq)
                        pred=np.append(pred,y_pred.cpu().numpy(),axis=0)
                m=[];n=[]        
                mn=get_data_maxmin(data_pred_index)
                if(type(mn).__name__=="list"):
                    for i in range(len(mn)):
                        m.append(mn[i]["max"]);n.append(mn[i]["min"])
                else:
                    m.append(mn["max"]);n.append(mn["min"]) 
                      
                m=np.array(m);n=np.array(n)                         
                pred = (m - n) * pred + n
                pred = list(chain.from_iterable(pred.data.tolist()))
                #for all mode there are four pred values: open/close/high/low.
                pred_values={}
                if pred_type == "all":
                    pred_values["open"] ="%0.2f"%pred[0]
                    pred_values["close"]="%0.2f"%pred[1]
                    pred_values["high"] ="%0.2f"%pred[2]
                    pred_values["low"]  ="%0.2f"%pred[3]
                else:
                    pred_values[pred_type] ="%0.2f"%pred[0]
                    
                print("pred result : %s mape: %s  nextday: %d" %(str(pred_values),mape,nextday)) 
                info_item['price']=pred_values
                info_item['nextday']=nextday
                info_item['mape']=mape 
                info_item['time']=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                info_out.append(info_item)
        layer=layer+1
        
    log.output("Query for stock%s result is %s !" %(stock_id,info_out))    
    
    if(len(info_out)>0):
        json_str=json.dumps(info_out)
        with open("./query/stock-"+str(stock_id)+'-'+time.strftime("%Y%m%d%H", time.localtime())+".json",'w+') as stockfile : 
            stockfile.write(json_str) 
    else:
        print("can not find model for stock: %s !" %stock_id) 
        log.output("can not find model for stock: %s !" %stock_id,level=1)

#the main procedure.    
if __name__ == '__main__' :
    data_index_set=[0,1,2,3,4,5,6,7,8,9] #data columns can be selected in dataframe one or more columns can be selected.input_size parameter in algs also be modified.
    data_pred_index=2 #close prise" columns index ( from "0" begin) in data csv file to be predicted and is our trarget.
    data_col_bypass=1 #header columns is not useful,such as index or date, should be pypassed.
    #model parameters some parameters maybe changed per your condition:
    #input_size should be modified by number input data columns to be used.
    #batch_size should be 5-8 for sinble parameters pridiction and 100 for multi-parameter prediction.
    #xxx_begin and xxx_end for data split, can be modified per yourslef.
    args={
          "input_size":len(data_index_set), #number of input parameters used in predition. you can modify it in data index list.
          "hidden_size":64,#number of cells in one hidden layer.
          "num_layers":1,  #number of hidden layers in predition module.
          "output_size":1, #number of parameter will be predicted.
          "lr":0.005,
          "weight_decay":0.0001,
          "bidirectional":False,
          "optimizer":"adam",
          "step_size":5,
          "gamma":0.5,
          "epochs":30,
          "batch_size":1,#batch of data will be push in model. large batch in multi parameter prediction will be better.
          "seq_len":24, #one contionus series input data will be used to predict the selected parameter.
          "multi_steps":1,#next x days's stock price can be predictions. maybe 1,2,3....
          "train_end":0.9,
          "val_begin":0.8,
          "val_end":0.9,
          "test_begin":0.9
        }
    #Data max min.
    data_mm=[]
    #module description in json
    result={}
    #stock data file in csv
    data_file_name="" 
    #save module data to file in the path.
    run_time=time.time()
    stock_id=cmd_line()
    data_file_name="./data/"+"stock-hist"+stock_id+".csv"
    print("stockid:%s to be predicted..." %stock_id)
    #test cuda
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA or CPU:", device)
    #load data to a dataFrame.
    Dte = nn_data_seq(args['batch_size'])
    #test for each model one by one.
    test(args, Dte)  
    elapse_time=time.time()-run_time
    print("query for stock %s and %0.2f seconds used."%(stock_id,elapse_time))
    log.output("query for stock %s and %0.2f seconds used."%(stock_id,elapse_time))