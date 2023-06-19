import numpy as np 
import torch
from torch import device, nn
import pandas as pd 
from torch.utils.data import DataLoader
import torch.utils.data as Data
from tqdm import tqdm, trange
from torch.optim import lr_scheduler
import copy
from itertools import chain
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline   
import json
import time
import os
import argparse
import ast
import sys
import lstm_stock_log as log
import math
from sklearn.metrics import precision_score, recall_score, f1_score
from torchsampler import ImbalancedDatasetSampler
import pickle
# from pytorchtools import EarlyStopping

BKS = ['000001', '880301', '880305', '880310', '880318', '880324', '880330', '880335', '880344', '880350', '880351', '880355', '880360', '880367', '880372', '880380', '880387', '880390', '880398', '880399', '880400', '880406', '880414', '880418', '880421', '880422', '880423', '880424', '880430', '880431', '880432', '880437', '880440', '880446', '880447', '880448', '880452', '880453', '880454', '880455', '880456', '880459', '880464', '880465', '880471', '880472', '880473', '880474', '880476', '880482', '880489', '880490', '880491', '880492', '880493', '880494', '880497', '399001']

#BKS = ["000001", "880367", "399001"]
BK_SIZE = 1 #len(BKS)
BK_TOPN = 10
COLS = ["open", "close", "high", "low", "vol"]
TCH_EARLYSTOP_PATIENCE = 200
ONLY_PREDICT = True
CLOSE_LABEL_THRESHOLD = 0.94
NP_TOPN = 20

#cmd line parmeters.
def cmd_line():
    parser = argparse.ArgumentParser(description='Calculate model according to stockid')
    parser.add_argument('-sid', '--stockid', type=str, help='Stock id')
    parser.add_argument('-predtype', '--predtype', type=str, help='open/close/high/low price can be predicted.')
    parser.add_argument('-nextday', '--nextday', type=int, help='which following day price to be predicted')
    parser.add_argument('-plot', '--plot', type=ast.literal_eval, help='plot or not for test result')    
    cmd_args = parser.parse_args() 
    stock_id=cmd_args.stockid
    pred_type=cmd_args.predtype;
    nextday=cmd_args.nextday
    plot=cmd_args.plot    
    return stock_id,pred_type,nextday,plot
#walk data directory to find data file for the given stock id.
def find_data_file(stock_id):
    for root,dir,files in os.walk("./data"):        
        for file_name in files:   
            if (file_name.find('.csv')>0 and file_name.find(stock_id)>0) :
                return file_name
    return None        

#get args.
def get_args():
    return args;
def get_data_maxmin(index):
    if index > len(data_mm):
        return data_mm[:args["output_size"]]
    else:    
        mm=data_mm[index-data_col_bypass] #bypass tble head  in dataframe.
        return mm
    
#calculate the mape. the input parameters is numpy.
def get_mape(actual,pred):
        temp=np.mean(np.abs((actual - pred) / actual)) * 100 
        return temp       

#load data from csv file and clean it.
def load_bkdata(file_name):
    if os.path.exists("bkcalc.hdf"):
        logdf = pd.read_hdf("bkcalc.hdf")
    else:
        # df=pd.read_csv('./data/'+file_name,encoding='UTF-8',na_values = missing_values,index_col=0)
        alldf = pd.read_hdf(file_name, "index")
        alldf = alldf.loc[alldf["date"]>"2008-06-26"]
        alldf = alldf.set_index(["date", "exchange", "code"]).sort_index()
        # df=df.drop(["up_count", "down_count", "update_time"],axis=1) #data format is changed in sometime we will remove some no useful columns.
        # df.dropna(axis='index', how='any',inplace=True)
        ratiodfList = []
        logdfList = []
        for code in alldf.index.get_level_values("code").unique():
            df = alldf.loc[ (alldf.index.get_level_values("code")==code) ]
            print(code, min(df.index.get_level_values(0)))
            # rdf = df.loc[ :, ["open", "close", "high", "low", "vol", "amount"]].rolling(2, min_periods=2).apply(lambda x: x.iloc[1]/x.iloc[0])
            # rdf = rdf.dropna()
            # ratiodfList += [ rdf ]
            ldf = df.loc[:, ["open", "close", "high", "low"]].rolling(2, min_periods=2).apply(lambda x: math.log(x.iloc[1]/x.iloc[0], 1.3)) #, "vol", "amount"
            ldf.loc[ :, ["vol"]] = df.loc[:, ["vol"]].rolling(2, min_periods=2).apply(lambda x: math.log(x.iloc[1]/x.iloc[0], 120)) #, "vol", "amount"
            ldf.loc[ :, ["amount"]] = df.loc[:, ["amount"]].rolling(2, min_periods=2).apply(lambda x: math.log(x.iloc[1]/x.iloc[0], 45)) #, "vol", "amount"
            ldf = ldf.dropna()
            logdfList += [ ldf ]

        # ratiodf = pd.concat(ratiodfList).sort_index()
        logdf = pd.concat(logdfList).sort_index()
        logdf.to_hdf("bkcalc.hdf", "idxcalc", complevel=1, complib="blosc:snappy", format="table")

    logdf = logdf.drop(columns=["amount"])
    print("bk size", len(logdf.loc[logdf.index.get_level_values("date")[0]].index), "col size", len(logdf.iloc[0]))
    return logdf

#load data from csv file and clean it.
def load_data(file_name):
    missing_values = ["n/a", "na", "--","None"]
    df=pd.read_csv('./data/'+file_name,encoding='UTF-8',na_values = missing_values,index_col=0)
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
    return df 

def load_stocks_data(file_name):
    df = pd.read_hdf("rlcalc.hdf", "rlcalc")
    df = df.reset_index().set_index(["exchange", "code", "date"]).sort_index()
    df = df.loc[df.index.get_level_values("date") > "2017-1-1"]
    # print(df.loc[(df>1.0).any(axis=1)].index.get_level_values("code"))
    # print(df.loc[(df<-1.0).any(axis=1)].index.get_level_values("code"))
    for code in df.loc[(df>1.0).any(axis=1)].index.get_level_values("code"):
        print(code)
        df = df.loc[df.index.get_level_values("code")!=code]
    for code in df.loc[(df<-1.0).any(axis=1)].index.get_level_values("code"):
        print(code)
        df = df.loc[df.index.get_level_values("code")!=code]
    return df

#create our dataset.
class MyDataset(Data.Dataset):
    def __init__(self, data):
        self.data = data
        self.labels = [ x[1].item() for x in data ]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels
    
# Create dataset.   
def process(data, batch_size, shuffle,data_index_set):
    args=get_args()
    data1=data.iloc[:,data_col_bypass:] #remove header from original dataframe.
    #print(data1)
    load = data1.to_numpy() 
    #normalize
    for i in range(load.shape[1]):
        max,min=np.max(load[:,i]),np.min(load[:,i])
        load[:,i] = (load[:,i]-min) / (max-min)
    
    seq = []
    seq_len=get_args()["seq_len"]
    steps=args["multi_steps"]
    offset=seq_len+steps-1;#keep the last seq data has it's label.
    for i in range(len(load) - offset):
        train_seq = []
        train_label = []
        for j in range(i, i + seq_len):
            x=[]
            for k in data_index_set:
                x.append(load[j][k])
            train_seq.append(x)
        if(data_pred_index>load.shape[1]):
            for m in range(args["output_size"]): #pred mode: all, open/close/high/low items in df load will be used as label.
                train_label.append(load[i + offset, m])
        else:        
            train_label.append(load[i + offset, data_pred_index-data_col_bypass]) #get the corresponding label by the given offset.
        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label)) 
    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=True)
    return seq

# Create dataset.
def process_bkMultiLabel(data, batch_size, shuffle,data_index_set):
    args=get_args()
    seq_len=get_args()["seq_len"]
    steps=args["multi_steps"]

    seq = []
    train_seqs = []
    train_labels = []
    predstep = steps + seq_len
    dataLen = 0
    for date in data.index.get_level_values("date").unique():
        train_seq = data.loc[date].to_numpy().flatten().tolist()
        close_topn = data.loc[date, "close"].sort_values(ascending=False).iloc[BK_TOPN]
        train_label = (data.loc[date, "close"]>=close_topn).to_numpy().flatten().tolist()
        train_seqs += [train_seq]
        train_labels += [train_label]

        dataLen += 1
        if dataLen >= predstep:
            train_seq_ts = torch.FloatTensor(train_seqs[dataLen - predstep:dataLen - predstep+seq_len])
            train_label_ts = torch.FloatTensor(train_labels[-1]).view(-1)
            seq.append((train_seq_ts, train_label_ts))
    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return seq

# Create dataset.
def process_stocks(dataAll, batch_size, shuffle,data_index_set, test_pred=False):
    args=get_args()

    seq_pkl_name = "torch_stock_seq_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}' + '.pkl'
    last_pkl_name = "torch_stock_last_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}' + '.pkl'


    if os.path.exists(seq_pkl_name):
        seq_pkl_file = open(seq_pkl_name, "rb")
        last_pkl_file = open(last_pkl_name, "rb")
        seq_pkl = pickle.load(seq_pkl_file)
        last_pkl = pickle.load(last_pkl_file)
        seq_pkl_file.close()
        last_pkl_file.close()
        return seq_pkl, last_pkl

    seq_len=get_args()["seq_len"]
    steps=args["multi_steps"]
    dataAll = dataAll.rename(columns={"hfq_open": "open", "hfq_high": "high", "hfq_low": "low", "hfq_close": "close"})

    dataAll = dataAll.loc[:, COLS]
    dataAll = dataAll.loc[dataAll.index.get_level_values("code") < "300000"] #"300000"

    predstep = steps + seq_len
    seq = []
    last_seq_ts = []

    for code in dataAll.index.get_level_values("code").unique():
        data = dataAll.loc[("szse", code)].sort_index()
        print("code", code)
        train_seqs = []
        train_labels = []
        dataLen = 0

        for date in data.index.get_level_values("date").unique().sort_values():
            train_seq = data.loc[date].to_numpy().flatten().tolist()
            # close_topn = data.loc[date, "close"].sort_values(ascending=False).iloc[BK_TOPN]
            # train_label = (data.loc[date, "close"]>=close_topn).to_numpy().flatten().tolist()
            # if args["type"] != "MultiLabelLSTM":
            #     train_label = data.loc[date, "close"].to_numpy().flatten().tolist()
            # else:
            #     close_topn = data.loc[date, "close"].sort_values(ascending=False).iloc[BK_TOPN]
            #     train_label = (data.loc[date, "close"]>=close_topn).to_numpy().flatten().tolist()
            train_seqs += [train_seq]
            train_labels += [[data.loc[date, "close"]]]

            dataLen += 1
            if dataLen >= predstep:
                train_seq_ts = torch.FloatTensor(train_seqs[dataLen - predstep:dataLen - predstep+seq_len])
                # train_label_ts = torch.FloatTensor(train_labels[-1]).view(-1)
                train_label_ts = torch.FloatTensor(np.array(train_labels)[-steps:].sum(axis=0)).view(-1)
                seq.append((train_seq_ts, train_label_ts))
        last_seq_ts += [(torch.FloatTensor(train_seqs[-seq_len:]), train_label_ts, code)] #todo train_label_ts is not the true label of future seq, but doesn't matter
    if test_pred == True:
        last_seq_ts = MyDataset(last_seq_ts)
        #if not test_pred else 1, drop_last=(not test_pred)
        if args["type"] == "MultiLabelLSTM":
            last_seq_ts = DataLoader(dataset=last_seq_ts, batch_size= 1, shuffle=False, num_workers=0, drop_last=False) #shuffle=shuffle,
        else:
            last_seq_ts = DataLoader(dataset=last_seq_ts, batch_size= 1, shuffle=False, num_workers=0, drop_last=False) #shuffle=shuffle,
    else:
        last_seq_ts = None
    seq = MyDataset(seq)
    #if not test_pred else 1, drop_last=(not test_pred)
    if args["type"] == "MultiLabelLSTM":
        seq = DataLoader(dataset=seq, batch_size=batch_size if not test_pred else 1, sampler=ImbalancedDatasetSampler(seq), num_workers=0, drop_last=(not test_pred)) #shuffle=shuffle,
    else:
        seq = DataLoader(dataset=seq, batch_size=batch_size if not test_pred else 1, shuffle=shuffle, num_workers=0, drop_last=(not test_pred)) #shuffle=shuffle,

    pkl_file = open(seq_pkl_name, "wb")
    pickle.dump(seq, pkl_file)
    pkl_file.close()
    pkl_file = open(last_pkl_name, "wb")
    pickle.dump(last_seq_ts, pkl_file)
    pkl_file.close()
    return seq, last_seq_ts

# Create dataset.
def process_bk(data, batch_size, shuffle,data_index_set, test_pred=False):
    args=get_args()
    seq_len=get_args()["seq_len"]
    steps=args["multi_steps"]

    #data = data.loc[:, COLS]

    seq = []
    train_seqs = []
    train_labels = []
    predstep = steps + seq_len
    dataLen = 0
    for date in data.index.get_level_values("date").unique():
        train_seq = data.loc[date].to_numpy().flatten().tolist()
        # close_topn = data.loc[date, "close"].sort_values(ascending=False).iloc[BK_TOPN]
        # train_label = (data.loc[date, "close"]>=close_topn).to_numpy().flatten().tolist()
        if args["type"] != "MultiLabelLSTM":
            train_label = data.loc[date, "close"].to_numpy().flatten().tolist()
        else:
            close_topn = data.loc[date, "close"].sort_values(ascending=False).iloc[BK_TOPN]
            train_label = (data.loc[date, "close"]>=close_topn).to_numpy().flatten().tolist()
        train_seqs += [train_seq]
        train_labels += [train_label]

        dataLen += 1
        if dataLen >= predstep:
            train_seq_ts = torch.FloatTensor(train_seqs[dataLen - predstep:dataLen - predstep+seq_len])
            # train_label_ts = torch.FloatTensor(train_labels[-1]).view(-1)
            train_label_ts = torch.FloatTensor(np.array(train_labels)[-steps:].sum(axis=0)).view(-1)
            seq.append((train_seq_ts, train_label_ts))
    if test_pred == True:
        last_seq_ts = torch.FloatTensor(train_seqs[-seq_len:])
        seq.append((last_seq_ts, train_label_ts))
        print("fixme if step more than 1")
    else:
        last_seq_ts = None
    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size if not test_pred else 1, shuffle=shuffle, num_workers=2, drop_last=(not test_pred))
    return seq, last_seq_ts

# split date and create datasets for train /validate and test.
def nn_stocksdata_seq(batch_size, lstmtype):
    print('data processing...')
    data_file_name = "rlcalc.hdf"
    dataset = load_stocks_data(data_file_name)
    # dataset = dataset.loc[(slice(None), slice(None), BKS), COLS].sort_index()

    algs=get_args()
    #check number of data items in df, if it is too less , we can not train it.
    algs["train_end"]=0.7
    algs["val_begin"]=0.7
    algs["val_end"]=0.9
    algs["test_begin"]=0.9
    print("fixme algs")

    dates = dataset.index.get_level_values("date").unique().sort_values()

    all_code_len = len(dates)
    train_date_end = dates[int(all_code_len*algs["train_end"])]
    train = dataset.loc[dataset.index.get_level_values("date")<=train_date_end]
    val_date_end = dates[int(all_code_len*algs["val_end"])]
    val = dataset.loc[(dataset.index.get_level_values("date")>train_date_end) & (dataset.index.get_level_values("date")<=val_date_end)]
    test = dataset.loc[dataset.index.get_level_values("date")>val_date_end]

    # # split
    # # train = dataset.iloc[:int(len(dataset.index)/BK_SIZE * algs["train_end"])*BK_SIZE]
    # # val  = dataset.iloc[int(len(dataset.index)/BK_SIZE * algs["val_begin"])*BK_SIZE:int(len(dataset.index)/BK_SIZE * algs["val_end"])*BK_SIZE]
    # # test = dataset.iloc[int(len(dataset.index)/BK_SIZE * algs["test_begin"])*BK_SIZE:len(dataset.index)]
    # for i in range(data_col_bypass,dataset.shape[1]):
    #     m, n = np.max(dataset[dataset.columns[i]]), np.min(dataset[dataset.columns[i]])
    #     mm={}
    #     mm['max']=m
    #     mm['min']=n
    #     data_mm.append(mm)

    #dataset.
    if ONLY_PREDICT == False:
        Dtr, _ = process_stocks(train, batch_size, True,data_index_set)
        Val, _ = process_stocks(val,   batch_size, True,data_index_set)
    else:
        Dtr = None
        Val = None
    Dte, last_seq_ts = process_stocks(test,  batch_size, False,data_index_set, test_pred=True)

    return Dtr, Val, Dte, last_seq_ts, test

# split date and create datasets for train /validate and test.
def nn_bkdata_seq(batch_size, lstmtype):
    print('data processing...')
    data_file_name = "bkidx.hdf"
    dataset = load_bkdata(data_file_name)
    dataset = dataset.loc[(slice(None), slice(None), BKS), COLS].sort_index()

    algs=get_args()
    #check number of data items in df, if it is too less , we can not train it.
    if len(dataset) < 300 :
        print("number data in %s is too less, can not train model." %data_file_name)
        log.output("number data in %s is too less, can not train model." %data_file_name,level=1)
        sys.exit(0) #can not find the data file.
    else:
        algs["train_end"]=0.8
        algs["val_begin"]=0.8
        algs["val_end"]=0.9
        algs["test_begin"]=0.9
    # split
    train = dataset.iloc[:int(len(dataset.index)/BK_SIZE * algs["train_end"])*BK_SIZE]
    val  = dataset.iloc[int(len(dataset.index)/BK_SIZE * algs["val_begin"])*BK_SIZE:int(len(dataset.index)/BK_SIZE * algs["val_end"])*BK_SIZE]
    test = dataset.iloc[int(len(dataset.index)/BK_SIZE * algs["test_begin"])*BK_SIZE:len(dataset.index)]
    for i in range(data_col_bypass,dataset.shape[1]):
        m, n = np.max(dataset[dataset.columns[i]]), np.min(dataset[dataset.columns[i]])
        mm={}
        mm['max']=m
        mm['min']=n
        data_mm.append(mm)

    #dataset.
    if lstmtype == "BiLSTM" or lstmtype == "LSTM" or True:
        Dtr, _ = process_bk(train, batch_size, True,data_index_set)
        Val, _ = process_bk(val,   batch_size, True,data_index_set)
        Dte, last_seq_ts = process_bk(test,  batch_size, False,data_index_set, test_pred=True)
    else:
        Dtr = process_bkMultiLabel(train, batch_size, True,data_index_set)
        Val = process_bkMultiLabel(val,   batch_size, True,data_index_set)
        Dte = process_bkMultiLabel(test,  batch_size, False,data_index_set, test_pred=True)
        last_seq_ts = None
        print("fixme last_seq_ts")

    return Dtr, Val, Dte, last_seq_ts, test

# split date and create datasets for train /validate and test.
def nn_data_seq(batch_size):
    print('data processing...')
    dataset = load_data(data_file_name+".CSV")
    algs=get_args()
    #check number of data items in df, if it is too less , we can not train it.
    if len(dataset) < 300 : 
        print("number data in %s is too less, can not train model." %data_file_name)
        log.output("number data in %s is too less, can not train model." %data_file_name,level=1)
        sys.exit(0) #can not find the data file.  
    if len(dataset)<1000:
        algs["train_end"]=0.95
        algs["val_begin"]=0.1 
        algs["val_end"]=0.95
        algs["test_begin"]=0.1         
    # split
    train = dataset[:int(len(dataset) * algs["train_end"])]
    val  = dataset[int(len(dataset) * algs["val_begin"]):int(len(dataset) * algs["val_end"])]
    test = dataset[int(len(dataset) * algs["test_begin"]):len(dataset)]
    for i in range(data_col_bypass,dataset.shape[1]):
        m, n = np.max(dataset[dataset.columns[i]]), np.min(dataset[dataset.columns[i]])
        mm={}
        mm['max']=m
        mm['min']=n
        data_mm.append(mm)
    
    #dataset.
    Dtr = process(train, batch_size, True,data_index_set)
    Val = process(val,   batch_size, True,data_index_set)
    Dte = process(test,  batch_size, False,data_index_set)

    return Dtr, Val, Dte

#nn module.
class MultiLabelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0)) # output(5, 30, 64)
        output = self.classifier(output)
        pred = self.sigmoid(output)  # (5, 24, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred

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
    
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision':      precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall':         recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1':             f1_score(y_true=target, y_pred=pred, average='micro'),
            # 'samples/precision':    precision_score(y_true=target, y_pred=pred, average='samples'),
            # 'samples/recall':       recall_score(y_true=target, y_pred=pred, average='samples'),
            # 'samples/f1':           f1_score(y_true=target, y_pred=pred, average='samples'),
            'macro/precision':      precision_score(y_true=target, y_pred=pred, average=None),
            'macro/recall':         recall_score(y_true=target, y_pred=pred, average=None),
            'macro/f1':             f1_score(y_true=target, y_pred=pred, average=None),
            }

def topn_rank(model_result, targets, topn=1):
    algs=get_args()
    num = 0
    rankAva = 0
    for idx in range(len(model_result)):
        # maxIdx = model_result[idx].argmax()
        topnidx = model_result[idx].argsort()[-topn]
        ranks = targets[idx].argsort().argsort()
        rank = np.take(ranks, topnidx).mean()
        # rank = ranks.mean()
        rankAva += rank
        num += 1
    print(f"\nTop {topn} Average Rank is:", (rankAva/num), flush=True)

#train it.
def train(args, Dtr, Val, path):
    args=get_args()
    input_size, hidden_size, num_layers = args['input_size'], args['hidden_size'], args['num_layers']
    output_size = args['output_size']
    
    if args["type"] == 'BiLSTM':
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)
    elif args["type"] == 'LSTM':
        model = LSTM(  input_size, hidden_size, num_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)
    else:
        model = MultiLabelLSTM(  input_size, hidden_size, num_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.BCELoss().to(device)

    if args['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'],momentum=0.9, weight_decay=args['weight_decay'])
        
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    if os.path.exists(path):
        print('loading models...')
        model.load_state_dict(torch.load(path)['models'])

    # training
    best_model = None
    train_loss_all=[]
    val_loss_all=[]
    best_loss=None
    # initialize the early_stopping object
    # early_stopping = EarlyStopping(patience=TCH_EARLYSTOP_PATIENCE, verbose=True)
    patience = 0

    print('training...')
    for epoch in tqdm(range(args['epochs'])):
        # validation
        val_loss=0
        num_item=0
        targets = []
        model_result = []
        model.eval()
        for (seq, label) in Val:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            model_result.extend( y_pred.detach().cpu().numpy() )
            targets.extend( label.detach().cpu().numpy() )
            loss = loss_function(y_pred, label)
            val_loss+=loss.item()*len(y_pred)
            num_item+=len(y_pred)
        if num_item>0:
            val_loss_all.append(val_loss/num_item)
        else:
            val_loss_all.append(val_loss)

        if best_loss is None:
            best_loss = val_loss_all[-1]

        #get the best model.
        if(val_loss_all[-1]<best_loss):
            best_loss=val_loss_all[-1]
            best_model=copy.deepcopy(model)
            state = {'models': best_model.state_dict()}
            print('Saving models...')
            torch.save(state, path)
            patience = 0
        elif val_loss_all[-1]>best_loss:
            patience += 1
            if patience > TCH_EARLYSTOP_PATIENCE:
                break

        if args["type"] == "MultiLabelLSTM":
            result = calculate_metrics(np.array(model_result), np.array(targets))
            for key in result:
                print(key)
                print(result[key])
        else:
            topnidx = np.array(model_result).argsort(axis=0)[-NP_TOPN:, :]
            topnclose = np.take(targets, topnidx)
            print("\nAverage close is:", topnclose.mean())

        train_loss = 0
        num_item=0
        model.train()
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss+=loss.item()*len(y_pred)
            num_item+=len(y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if num_item>0:
            train_loss_all.append(train_loss/num_item)
        else:
            train_loss_all.append(train_loss)

        print('\nepoch {:03d} train_loss {:.8f} val_loss {:.8f} best_loss {:.8f} patience {:04d}'.format(epoch, train_loss_all[-1], val_loss_all[-1], best_loss, patience), flush=True)

    #save the best model.
    state = {'models': best_model.state_dict()}
    torch.save(state, path)
    
    # plot for loss.
    if plot :
        x = [i for i in range(1, len(train_loss_all)+1)]
        x_smooth = np.linspace(np.min(x), np.max(x), 1000)
        y_model=make_interp_spline(x, train_loss_all)
        y_smooth = y_model(x_smooth)
        plt.title('loss')
        plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='train_loss')

        pred_model=make_interp_spline(x, val_loss_all)
        y_smooth = pred_model(x_smooth)
        plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='val_loss')
        plt.grid(axis='y')
        plt.legend()
        plt.show()

    
#validate
def test(args, Dte, path,data_pred_index, last_seq_ts, testdf):
    
    # m=[];n=[]
    # mn=get_data_maxmin(data_pred_index)
    # if(type(mn).__name__=="list"):
    #     for i in range(len(mn)):
    #         m.append(mn[i]["max"]);n.append(mn[i]["min"])
    # else:
    #     m.append(mn["max"]);n.append(mn["min"])
    # m=np.array(m);n=np.array(n)
        
    args=get_args()
    input_size, hidden_size, num_layers = args['input_size'], args['hidden_size'], args['num_layers']
    output_size = args['output_size']
    
    if args['type'] == "BiLSTM": #lstm, bidirection-lstm, multilabel-lstm
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=1).to(device)
    elif args['type'] == "LSTM":
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=1).to(device)
    else:
        model = MultiLabelLSTM(input_size, hidden_size, num_layers, output_size, batch_size=1).to(device)

    print('loading models...')
    model.load_state_dict(torch.load(path)['models'])
    
    model.eval()
    print('predicting...')
    if args['pred_type']=="all" and False:
        pred=np.empty(shape=(0,4))
        y=np.empty(shape=(0,4))
    else:
        pred=np.empty(shape=(0,args["output_size"]))
        y=np.empty(shape=(0,args["output_size"]))
    targets = []
    model_result = []
    #'''
    for (seq, target) in tqdm(Dte):
        y=np.append(y,target.numpy(),axis=0)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            # pred=np.append(pred,y_pred.cpu().numpy(),axis=0)
            model_result.extend( y_pred.detach().cpu().numpy() )
            targets.extend( target.detach().cpu().numpy() )

    if args["type"] == "MultiLabelLSTM":
        result = calculate_metrics(np.array(model_result)[:-args["multi_steps"]], np.array(targets)[:-args["multi_steps"]])
        for key in result:
            print(key)
            print(result[key])
    else:
        # topn_rank(y[:-args["multi_steps"]], pred[:-args["multi_steps"]], 1)
        # topn_rank(y[:-args["multi_steps"]], pred[:-args["multi_steps"]], 2)
        # topn_rank(y[:-args["multi_steps"]], pred[:-args["multi_steps"]], 3)
        #
        # last_pred = y_pred.cpu().numpy()[0]
        # top3idx = last_pred.argsort()[-3]
        # top3mask = last_pred>=last_pred[top3idx]
        # print("predict top3 at", testdf.index[-1][0], testdf.loc[testdf.index[-1][0]].index.get_level_values(1)[top3mask])
        topnidx = np.array(model_result).argsort(axis=0)[-NP_TOPN:, :]
        topnclose = np.take(targets, topnidx)
        print("\nAverage close is:", topnclose.mean())
    #'''
    codes = []
    targets = []
    model_result = []
    for (seq, target, code) in last_seq_ts:
        # y=np.append(y,target.numpy(),axis=0)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            # pred=np.append(pred,y_pred.cpu().numpy(),axis=0)
            model_result.extend( y_pred.detach().cpu().numpy() )
            targets.extend( target.detach().cpu().numpy() )
            codes.extend(code)

    topnidx = np.array(model_result).argsort(axis=0)[-NP_TOPN:, :]
    topnclose = np.take(codes, topnidx)
    print("topn codes")
    print(topnclose)

    # y = (m - n) * y + n
    # pred = (m - n) * pred + n
    # mape=get_mape(y, pred)
    # print("mape:%.2f%% with pred_type: %s price multi_steps: %d" %(mape,args['pred_type'],args['multi_steps']))
    
    #add module info to result dict.
    # result['filename']=path
    # result['datetime']=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # result['pred_type']=args['pred_type']
    # result['multi_steps']=args['multi_steps']
    # result['input_para_num']=str(data_index_set)
    # result['mape']="%0.2f%%"%mape
    # #write module info to json file
    # path_tmp=path.replace('.pkl','.json')
    # json_file=open(path_tmp, mode='w+')
    # json_str=json.dumps(result);
    # json_file.write(json_str)
    # json_file.close()
    
    # plot for pred.
    if plot and False:
        x = [i for i in range(1, len(pred)+1)]
        x_smooth = np.linspace(np.min(x), np.max(x), 1000)
        y_model=make_interp_spline(x, y)
        y_smooth = y_model(x_smooth)
        plt.title('stock pred:'+"bk")
        plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

        pred_model=make_interp_spline(x, pred)
        y_smooth = pred_model(x_smooth)
        plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
        plt.grid(axis='y')
        plt.legend()
        plt.show()

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
          "input_size":BK_SIZE*len(COLS), #number of input parameters used in predition. you can modify it in data index list.
          "hidden_size":int(BK_SIZE*len(COLS)*3),#number of cells in one hidden layer.
          "num_layers":4,  #number of hidden layers in predition module.
          "output_size":BK_SIZE, #number of parameter will be predicted.
          "lr":1e-4,
          "weight_decay":0.0, #0001,
          "bidirectional":False,
          "type": "LSTM", # BiLSTM, LSTM, MultiLabelLSTM
          "optimizer":"adam",
          "step_size":500000000000,
          "gamma":0.5,
          "epochs":50000,
          "batch_size":64,#batch of data will be push in model. large batch in multi parameter prediction will be better.
          "seq_len":80, #one contionus series input data will be used to predict the selected parameter.
          "multi_steps":2,#next x days's stock price can be predictions. maybe 1,2,3....
          "pred_type":"close",#open price / close price / high price / low price.
          "train_end":1.0,
          "val_begin":0.6,
          "val_end":0.8,
          "test_begin":0.8
        }
    stock_id,pred_type,nextday,plot=cmd_line()
    if nextday in [1,2,3]:
        args['multi_steps']=nextday    
    if pred_type in ["open","close","high","low","all"]:
        args['pred_type']=pred_type
        pred_type_option={"open":1,"close":2,"high":3,"low":4,"all":4000} #all: single model for all predtype: open/close/high/low.
        data_pred_index=pred_type_option[pred_type]
    if pred_type =="all" :
        args["output_size"]=4   #for open / close /high/low.

    torch.set_num_threads(os.cpu_count()-1)

    #Data max min.
    data_mm=[]
    #module description in json
    result={}
    #stock data file in csv
    #data_file_name="guizhoumaotai600519" 
    # data_file_name=""
    # temp_name=find_data_file(stock_id)
    # if temp_name != None :
    #     data_file_name=temp_name[:-4] #remove the .csv in filename.
    #     print("%s 's model will be calcuated." %data_file_name)
    # else:
    #     print("can not find data file for stock: %s !" %stock_id)
    #     log.output("can not find data file for stock: %s !" %stock_id, level=1)
    #     sys.exit(0) #can not find the data file.
    #save module data to file in the path.
    # path_file='./model/'+'module'+ '-'+ data_file_name +'-'+pred_type+'-0'+ str(args['multi_steps']) +'.pkl' #Module for the next "x" day's stock price prediction.
    #test cuda
    path_file='./model/'+'module-' + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}' + '.pkl' #Module for the next "x" day's stock price prediction.
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA or CPU:", device)
    #load data to a dataFrame.
    Dtr, Val, Dte, last_seq_ts, testdf = nn_stocksdata_seq(args['batch_size'], args['type'])
    #train it.
    if ONLY_PREDICT == False:
        train(args, Dtr, Val, path_file)
    #teest it.
    test(args, Dte, path_file, data_pred_index, last_seq_ts, testdf)
    