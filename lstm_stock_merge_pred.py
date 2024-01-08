import numpy as np
import pandas
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
# import pickle
import _pickle as pickle
from exp_tools import dfCfg, dfCfgNorm, denorm_fn
import datetime
from torch.utils.data import random_split
# from pytorchtools import EarlyStopping
import signal
import bz2
from mmdataset import MMDataset, PklDataset
import random
from itertools import chain
from tsai.models.RNNAttention import LSTMAttention
# import tsai.models.RNNAttention
from torchmetrics.regression import R2Score
from modelmerge import LSTMs
from nn_macro import *

BKS = ['000001', '880301', '880305', '880310', '880318', '880324', '880330', '880335', '880344', '880350', '880351', '880355', '880360', '880367', '880372', '880380', '880387', '880390', '880398', '880399', '880400', '880406', '880414', '880418', '880421', '880422', '880423', '880424', '880430', '880431', '880432', '880437', '880440', '880446', '880447', '880448', '880452', '880453', '880454', '880455', '880456', '880459', '880464', '880465', '880471', '880472', '880473', '880474', '880476', '880482', '880489', '880490', '880491', '880492', '880493', '880494', '880497', '399001']

#BKS = ["000001", "880367", "399001"]
BK_SIZE = 1 #len(BKS)
BK_TOPN = 10
# COLS = ["open", "close", "high", "low", "vol"] #, 'buy_sm_vol', 'sell_sm_vol',  'buy_md_vol', 'sell_md_vol', 'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol'] #
# COLS = ["open", "close", "high", "low", "vol", "idxopen", "idxhigh", "idxlow", "idxclose", "idxvol"] #, 'buy_sm_vol', 'sell_sm_vol',  'buy_md_vol', 'sell_md_vol', 'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol'] #
TCH_EARLYSTOP_PATIENCE = 1 #20
# ONLY_PREDICT = True
NO_TRAIN = False
NO_TEST = False
DUMP_DATASET_TRAIN_VAL_PKL = True
RANGE_NORM = False   #True pred bad

TORCH_REPRODUCIBILITY = False
reproduct_args = {}

MAX_SEQ_LEN = 128
LABEL_INFO_LEN = 3 #label, code, date, info

CLOSE_LABEL_THRESHOLD = 0.94
NP_TOPN = 10
BIG_RISE = -math.inf #0.6
RISE_WIN = 5
LSTM_ADJUST_START = "O"  #"O" "N"
LSTM_ADJUST_END = "N"  #"O" "N"

MODULE_LOAD_BEST = NO_TRAIN
MODULE_CURRENT_SAVED = None
MODULE_SAVED_PING = 0
MODULE_SAVED_PONG = 1

DATA_SETS = ["zxbintra", "zxbzzintra"]
# DATA_FN_KEY = "zxbintra"
# DATA_FN_KEY = "zxbintra_zxbzzintra"
DATA_FN_KEY = "_".join(DATA_SETS)

COLS = ["open", "close", "high", "low", "vol"]
# COLS += [ stockType+c for stockType in DATA_SETS[1:] for c in COLS]
# COLS += [ stockType + ohlv[len("hfq_"):] if ohlv.startswith("hfq_") else stockType + ohlv  for ohlv in dfCfgNorm[stockType]["OHLCV"].keys() for stockType in DATA_SETS[1:] ]
for stockType in DATA_SETS[1:]:
    for ohlv in dfCfgNorm[stockType]["OHLCV"].keys():
        COLS += [stockType + ohlv[len("hfq_"):] if ohlv.startswith("hfq_") else stockType + ohlv]

DATA_ALL_FN = f"rlcalc_{DATA_FN_KEY}_all.hdf"
DATA_TRAIN_FN = f"rlcalc_{DATA_FN_KEY}_train.hdf"
DATA_VAL_FN = f"rlcalc_{DATA_FN_KEY}_val.hdf"
DATA_TEST_FN = f"rlcalc_{DATA_FN_KEY}_test.hdf"
DATA_PRED_FN = f"rlcalc_{DATA_FN_KEY}_pred.hdf"

print(DATA_ALL_FN, DATA_TRAIN_FN, DATA_VAL_FN, DATA_TEST_FN, DATA_PRED_FN)
TEST_FLAG = False

FS_COLS = ["038开发支出", "009研发费用", "036五、现金及现金等价物净增加额", "001一、营业总收入", "010经营活动产生的现金流量净额", "001一、营业总收入",
           "003二、营业总成本", "020流动资产合计", "015三、营业利润", "087所有者权益(或股东权益)合计", "039商誉", "050预收款项", "030固定资产及清理(合计)",
           "027在建工程(合计)", "044资产总计", "008预付款项", "041递延所得税资产", "019四、利润总额", "扣除非经常性损益后的净利润", "063流动负债合计",
           "022归属于母公司所有者的净利润", "053财务费用", "010资产减值损失", "042资产减值准备", "045长期待摊费用摊销", "044无形资产摊销",
           "043固定资产折旧、油气资产折耗、生产性物资折旧"]
DATA_FS_FN = "exp_fs_yearly.hdf"
EXP_FS_YEARS = 3

EXP_DK1D_SEQ_LEN = 20
EXP_FS1Y_SEQ_LEN = 3

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
    global COLS
    df = pd.read_hdf(file_name, "rlcalc")
    COLS = list(df.columns)
    print(COLS)
    df = df.reset_index().set_index(["exchange", "code", "date"]).sort_index()
    # df = df.rename(columns={"hfq_open": "open", "hfq_high": "high", "hfq_low": "low", "hfq_close": "close"})
    df = df.loc[:, COLS]
    df = df.loc[df.index.get_level_values("date") > "2012-5-1", COLS]
    # print("fixme debug")
    # df = df.loc[(df.index.get_level_values("date") > "2012-5-1")&(df.index.get_level_values("code")>"002900"), COLS]
    # print(df.loc[(df>1.0).any(axis=1)].index.get_level_values("code"))
    # print(df.loc[(df<-1.0).any(axis=1)].index.get_level_values("code"))
    for code in df.loc[(df>1.0).any(axis=1)].index.get_level_values("code").unique():
        print("drop", code)
        df = df.loc[df.index.get_level_values("code")!=code]
    for code in df.loc[(df<-1.0).any(axis=1)].index.get_level_values("code").unique():
        print("drop", code)
        df = df.loc[df.index.get_level_values("code")!=code]
    return df

def load_stock_fs_date(fn):
    df = pd.read_hdf(fn, "fs")
    df = df.loc[:, FS_COLS]
    return df

#create our dataset.
class MyDataset(Data.Dataset):
    def __init__(self, data):
        self.data = data
        # self.labels = [ x[1].item() for x in data ]

    def __getitem__(self, item):
        #print(len(self.data), item, "\r\n", self.data[item][0].numpy()[::34], self.data[item][1].numpy())
        return self.data[item]

    def __len__(self):
        return len(self.data)

    # def get_labels(self):
    #     return self.labels
    
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
#todo: for O1/O0
def process_stocks_O2toO1(dataAll, batch_size, shuffle,data_index_set, test_pred=False):
    args=get_args()

    seq_pkl_name = "torch_stock_seq_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}-{args["multi_steps"]}' + '.pkl'
    last_pkl_name = "torch_stock_last_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}-{args["multi_steps"]}' + '.pkl'


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
        print("proc code", code)
        train_seqs = []
        train_labels = []
        train_adjs = []
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
            train_labels += [[data.loc[date, "close"]+data.loc[date, "open"]/np.log(dfCfg[DATA_FN_KEY]["hfq_close"])*np.log(dfCfg[DATA_FN_KEY]["hfq_open"][0])]]
            train_adjs += [[-data.loc[date, "open"]/np.log(dfCfg[DATA_FN_KEY]["hfq_close"])*np.log(dfCfg[DATA_FN_KEY]["hfq_open"][0])]]

            dataLen += 1
            if dataLen >= predstep:
                train_seq_ts = torch.FloatTensor(train_seqs[dataLen - predstep:dataLen - predstep+seq_len])
                # train_label_ts = torch.FloatTensor(train_labels[-1]).view(-1)
                if steps == 1:
                    train_label_ts = torch.FloatTensor(train_adjs[-steps]).view(-1)
                else:
                    train_label_ts = torch.FloatTensor(train_adjs[-steps] + np.array(train_labels)[-steps+1:].sum(axis=0)).view(-1)
                seq.append((train_seq_ts, train_label_ts, code, date.value))
        last_seq_ts += [(torch.FloatTensor(train_seqs[-seq_len:]), train_label_ts, code, (date + datetime.timedelta(days=steps)).value)] #todo train_label_ts is not the true label of future seq, but doesn't matter
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

def process_stocks_norm_c2c1(dataAll, batch_size, shuffle,data_index_set, test_pred=False):
    args=get_args()

    seq_pkl_name = "torch_stock_seq_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}-{args["multi_steps"]}' + '.pkl'
    last_pkl_name = "torch_stock_last_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}-{args["multi_steps"]}' + '.pkl'


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
        print("proc code", code)
        train_seqs = []
        train_labels = []
        train_adjs = []
        dataLen = 0
        missing_seq = []
        bigrise_seq = []

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
            train_labels += [[denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["hfq_close"])]]
            train_adjs += [[1.0]]

            if data.loc[date].sum(skipna=False) != data.loc[date].sum(skipna=False):
                missing_seq += [True]
            else:
                missing_seq += [False]

            if data.loc[date, "close"] > BIG_RISE:
                bigrise_seq += [True]
            else:
                bigrise_seq += [False]

            dataLen += 1
            if dataLen >= predstep:
                train_seq_ts = torch.FloatTensor(train_seqs[dataLen - predstep:dataLen - predstep+seq_len])
                # train_label_ts = torch.FloatTensor(train_labels[-1]).view(-1)
                if steps == 1:
                    train_label_ts = torch.FloatTensor(train_adjs[-steps]).view(-1)
                else:
                    train_label_ts = torch.FloatTensor(train_adjs[-steps] * np.array(train_labels)[-steps+1:].prod(axis=0)/train_labels[-steps]).view(-1)
                if any(missing_seq[dataLen - predstep:dataLen - predstep+seq_len]) ==False and missing_seq[-steps]==False and any(missing_seq[-steps+1:])==False \
                        and any(bigrise_seq[-RISE_WIN-steps:-steps]):
                    seq.append((train_seq_ts, train_label_ts, code, date.value))
                else:
                    pass
                    #print("missing", sum(missing_seq[dataLen - predstep:dataLen - predstep+seq_len]))
        if any(missing_seq[-seq_len:]) == False and any(bigrise_seq[-RISE_WIN:]):
            last_seq_ts += [(torch.FloatTensor(train_seqs[-seq_len:]), train_label_ts, code, (date + datetime.timedelta(days=steps)).value)] #todo train_label_ts is not the true label of future seq, but doesn't matter
        else:
            pass
            #print("missing")
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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def reproducibility_init():
    global reproduct_args
    if TORCH_REPRODUCIBILITY == True:
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        g = torch.Generator()
        g.manual_seed(0)
        reproduct_args = {
            "worker_init_fn": seed_worker,
            "generator": g
        }

def process_stocks_norm(dataAll, batch_size, shuffle,data_index_set, test_pred=False, stage="train_test_", dump_file = True):
    global reproduct_args
    args=get_args()

    seq_pkl_name = "torch_stock_" + stage + "seq_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["output_size"]}-{args["multi_steps"]}' + '.pkl'
    last_pkl_name = "torch_stock_" + stage + "last_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["output_size"]}-{args["multi_steps"]}' + '.pkl'


    if os.path.exists(seq_pkl_name):
        # seq_pkl_file = open(seq_pkl_name, "rb")
        # last_pkl_file = open(last_pkl_name, "rb")
        with bz2.BZ2File(seq_pkl_name, "rb") as seq_pkl_file:
            seq_pkl = pickle.load(seq_pkl_file)
        with bz2.BZ2File(last_pkl_name, "rb") as last_pkl_file:
            last_pkl = pickle.load(last_pkl_file)
        # seq_pkl_file.close()
        # last_pkl_file.close()
        return seq_pkl, last_pkl

    seq_len=get_args()["seq_len"]
    steps=args["multi_steps"]
    dataAll = dataAll.rename(columns={"hfq_open": "open", "hfq_high": "high", "hfq_low": "low", "hfq_close": "close"})

    dataAll = dataAll.loc[:, COLS]
    dataAll = dataAll.loc[dataAll.index.get_level_values("code") < "300000"] #"300000"
    dataAll = dataAll.sort_index()

    predstep = steps + seq_len
    seq = []
    last_seq_ts = []

    for code in dataAll.index.get_level_values("code").unique():
        data = dataAll.loc[("szse", code)].sort_index()
        # print("proc code", code)
        train_seqs = []
        train_labels = []
        train_adjs = []
        train_adjs_end = []
        dataLen = 0
        missing_seq = []
        bigrise_seq = []
        infos = []

        # idxvol_idx = list(data.columns).index("idxvol")
        # vod_idx    = list(data.columns).index("vol")
        range_norm_list = []
        for i in range(len(data.columns)):
            if "vol" in data.columns[i]:
                range_norm_list += [i]
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
            train_labels += [[denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_close"]["range"])]]

            if LSTM_ADJUST_START == "O":
                train_adjs += [[1.0/(denorm_fn(data.loc[date, "open"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_open"]["range"])*denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_close"]["range"]))]]
            elif LSTM_ADJUST_START == "L":
                train_adjs += [[1.0/(denorm_fn(data.loc[date, "low"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_low"]["range"])*denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_close"]["range"]))]]
            else:
                train_adjs += [[1.0]]

            if LSTM_ADJUST_END == "O":
                train_adjs_end += [[denorm_fn(data.loc[date, "open"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_open"]["range"])]]
            elif LSTM_ADJUST_END == "L":
                train_adjs_end += [[denorm_fn(data.loc[date, "low"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_low"]["range"])]]
            else:
                train_adjs_end += [[1.0]]

            # infos += [[denorm_fn(data.loc[date, "low"], dfCfgNorm[DATA_SETS[0]]["hfq_low"]) <= (1.0/denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["hfq_close"]))*0.99]]
            infos += [[True]]

            # train_adjs += [[denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["hfq_close"])/denorm_fn(data.loc[date, "open"], dfCfgNorm[DATA_SETS[0]]["hfq_open"])]]

            if data.loc[date].sum(skipna=False) != data.loc[date].sum(skipna=False):
                missing_seq += [True]
            else:
                missing_seq += [False]

            if data.loc[date, "close"] > BIG_RISE:
                bigrise_seq += [True]
            else:
                bigrise_seq += [False]

            dataLen += 1
            if dataLen >= predstep:
                npa = np.array(train_seqs[dataLen - predstep:dataLen - predstep+seq_len])
                if RANGE_NORM == True:
                    for cidx in range_norm_list:
                        npa[:, cidx] = (2*npa[:, cidx] - np.max(npa[:, cidx]) - np.min(npa[:, cidx]))/(np.max(npa[:, cidx]) - np.min(npa[:, cidx]))
                # train_seq_ts = torch.FloatTensor(train_seqs[dataLen - predstep:dataLen - predstep+seq_len])
                train_seq_ts = torch.FloatTensor(npa)
                # train_label_ts = torch.FloatTensor(train_labels[-1]).view(-1)
                if False: #steps == 1:
                    train_label_ts = torch.FloatTensor(train_adjs[-steps]).view(-1)
                else:
                    train_label_ts = torch.FloatTensor(train_adjs[-steps] * np.array(train_labels)[-steps:].prod(axis=0)*train_adjs_end[-1]).view(-1)
                if any(missing_seq[dataLen - predstep:dataLen - predstep+seq_len]) ==False and missing_seq[-steps]==False and any(missing_seq[-steps+1:])==False \
                        and any(bigrise_seq[-RISE_WIN-steps:-steps]):
                    seq.append((train_seq_ts, train_label_ts, code, date.value, infos[-steps]))
                else:
                    pass
                    #print("missing", sum(missing_seq[dataLen - predstep:dataLen - predstep+seq_len]))
        if any(missing_seq[-seq_len:]) == False and any(bigrise_seq[-RISE_WIN:]):
            last_seq_ts += [(torch.FloatTensor(train_seqs[-seq_len:]), train_label_ts, code, (date + datetime.timedelta(days=steps)).value)] #todo train_label_ts is not the true label of future seq, but doesn't matter
        else:
            pass
            #print("missing")
    if test_pred == True:
        last_seq_ts = MyDataset(last_seq_ts)
        #if not test_pred else 1, drop_last=(not test_pred)
        if args["type"] == "MultiLabelLSTM":
            last_seq_ts = DataLoader(dataset=last_seq_ts, batch_size= 1, shuffle=False, num_workers=0, **reproduct_args, drop_last=False) #shuffle=shuffle,
        else:
            last_seq_ts = DataLoader(dataset=last_seq_ts, batch_size= 1, shuffle=False, num_workers=0, **reproduct_args, drop_last=False) #shuffle=shuffle,
    else:
        last_seq_ts = None

    print(f"---->{stage} size: {len(seq)}", list(dataAll.index.get_level_values("date").unique().sort_values())[0], list(dataAll.index.get_level_values("date").unique().sort_values())[-1])
    seq = MyDataset(seq)
    #if not test_pred else 1, drop_last=(not test_pred)
    if args["type"] == "MultiLabelLSTM":
        seq = DataLoader(dataset=seq, batch_size=batch_size if not test_pred else 1, sampler=ImbalancedDatasetSampler(seq), num_workers=0, **reproduct_args,drop_last=(not test_pred)) #shuffle=shuffle,
    else:
        seq = DataLoader(dataset=seq, batch_size=batch_size if not test_pred else 1, shuffle=shuffle, num_workers=0, **reproduct_args,drop_last=(not test_pred)) #shuffle=shuffle,

    if dump_file == True:
        # pkl_file = open(seq_pkl_name, "wb")
        with bz2.BZ2File(seq_pkl_name, "wb") as pkl_file:
            pickle.dump(seq, pkl_file)
        # pkl_file.close()
        # pkl_file = open(last_pkl_name, "wb")
        with bz2.BZ2File(last_pkl_name, "wb") as pkl_file:
            pickle.dump(last_seq_ts, pkl_file)
        # pkl_file.close()
    return seq, last_seq_ts

def process_stocks_norm_mmdataset(dataAll, batch_size, shuffle,data_index_set, test_pred=False, stage="train_test_", dump_file = True):
    global reproduct_args
    args=get_args()

    seq_pkl_name = "torch_stock_" + stage + "seq_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["output_size"]}-{args["multi_steps"]}' #+ '.pkl'
    last_pkl_name = "torch_stock_" + stage + "last_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["output_size"]}-{args["multi_steps"]}' + '.pkl'

    if os.path.exists(f"{seq_pkl_name}_mmcfg.pkl"):
        mmdataset = MMDataset(seq_name=seq_pkl_name)
        if args["type"] == "MultiLabelLSTM":
            seq_pkl = DataLoader(dataset=mmdataset, batch_size=batch_size if not test_pred else 1, sampler=ImbalancedDatasetSampler(mmdataset), num_workers=0, drop_last=(not test_pred)) #shuffle=shuffle,
        else:
            seq_pkl = DataLoader(dataset=mmdataset, batch_size=batch_size if not test_pred else 1, shuffle=shuffle, num_workers=0, drop_last=(not test_pred)) #shuffle=shuffle,
        with bz2.BZ2File(last_pkl_name, "rb") as last_pkl_file:
            last_pkl = pickle.load(last_pkl_file)
        return seq_pkl, last_pkl

    seq_len=get_args()["seq_len"]
    steps=args["multi_steps"]
    dataAll = dataAll.rename(columns={"hfq_open": "open", "hfq_high": "high", "hfq_low": "low", "hfq_close": "close"})

    dataAll = dataAll.loc[:, COLS]
    dataAll = dataAll.loc[dataAll.index.get_level_values("code") < "300000"] #"300000"
    dataAll = dataAll.sort_index()

    predstep = steps + seq_len
    seq = []
    last_seq_ts = []

    mmdataset = MMDataset(seq_name=seq_pkl_name, size=len(dataAll.index), input_shape=(seq_len, len(dataAll.columns)), label_shape=(args["output_size"], ), info_shape=(LABEL_INFO_LEN, ))
    mmdataset_idx = 0

    for code in dataAll.index.get_level_values("code").unique():
        data = dataAll.loc[("szse", code)].sort_index()
        # print("proc code", code)
        train_seqs = []
        train_labels = []
        train_adjs = []
        train_adjs_end = []
        dataLen = 0
        missing_seq = []
        bigrise_seq = []
        infos = []

        # idxvol_idx = list(data.columns).index("idxvol")
        # vod_idx    = list(data.columns).index("vol")
        range_norm_list = []
        for i in range(len(data.columns)):
            if "vol" in data.columns[i]:
                range_norm_list += [i]
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
            train_labels += [[denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_close"]["range"])]]

            if LSTM_ADJUST_START == "O":
                train_adjs += [[1.0/(denorm_fn(data.loc[date, "open"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_open"]["range"])*denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_close"]["range"]))]]
            elif LSTM_ADJUST_START == "L":
                train_adjs += [[1.0/(denorm_fn(data.loc[date, "low"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_low"]["range"])*denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_close"]["range"]))]]
            else:
                train_adjs += [[1.0]]

            if LSTM_ADJUST_END == "O":
                train_adjs_end += [[denorm_fn(data.loc[date, "open"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_open"]["range"])]]
            elif LSTM_ADJUST_END == "L":
                train_adjs_end += [[denorm_fn(data.loc[date, "low"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_low"]["range"])]]
            else:
                train_adjs_end += [[1.0]]

            # infos += [[denorm_fn(data.loc[date, "low"], dfCfgNorm[DATA_SETS[0]]["hfq_low"]) <= (1.0/denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["hfq_close"]))*0.99]]
            infos += [[True]]

            # train_adjs += [[denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["hfq_close"])/denorm_fn(data.loc[date, "open"], dfCfgNorm[DATA_SETS[0]]["hfq_open"])]]

            if data.loc[date].sum(skipna=False) != data.loc[date].sum(skipna=False):  #fixme performance
                missing_seq += [True]
            else:
                missing_seq += [False]

            if data.loc[date, "close"] > BIG_RISE:
                bigrise_seq += [True]
            else:
                bigrise_seq += [False]

            dataLen += 1
            if dataLen >= predstep:
                npa = np.array(train_seqs[dataLen - predstep:dataLen - predstep+seq_len])
                if RANGE_NORM == True:
                    for cidx in range_norm_list:
                        npa[:, cidx] = (2*npa[:, cidx] - np.max(npa[:, cidx]) - np.min(npa[:, cidx]))/(np.max(npa[:, cidx]) - np.min(npa[:, cidx]))
                #train_seq_ts = torch.FloatTensor(npa)
                #if False: #steps == 1:
                #    train_label_ts = torch.FloatTensor(train_adjs[-steps]).view(-1)
                #else:
                #    train_label_ts = torch.FloatTensor(train_adjs[-steps] * np.array(train_labels)[-steps:].prod(axis=0)*train_adjs_end[-1]).view(-1)

                if any(missing_seq[dataLen - predstep:dataLen - predstep+seq_len]) ==False and missing_seq[-steps]==False and any(missing_seq[-steps+1:])==False \
                        and any(bigrise_seq[-RISE_WIN-steps:-steps]):
                    # seq.append((train_seq_ts, train_label_ts, code, date.value, infos[-steps]))
                    mmdataset.map(mmdataset_idx, npa, train_adjs[-steps] * np.array(train_labels)[-steps:].prod(axis=0)*train_adjs_end[-1],
                                                       np.array([float(code), float(date.value), float(infos[-steps][0])]))
                    mmdataset_idx += 1
                else:
                    pass
                    #print("missing", sum(missing_seq[dataLen - predstep:dataLen - predstep+seq_len]))
        if any(missing_seq[-seq_len:]) == False and any(bigrise_seq[-RISE_WIN:]):
            train_label_ts = torch.FloatTensor(train_adjs[-steps] * np.array(train_labels)[-steps:].prod(axis=0)*train_adjs_end[-1]).view(-1)
            last_seq_ts += [(torch.FloatTensor(train_seqs[-seq_len:]), train_label_ts, code, (date + datetime.timedelta(days=steps)).value)] #todo train_label_ts is not the true label of future seq, but doesn't matter
        else:
            pass
            #print("missing")
    mmdataset.save()
    if test_pred == True:
        last_seq_ts = MyDataset(last_seq_ts)
        #if not test_pred else 1, drop_last=(not test_pred)
        if args["type"] == "MultiLabelLSTM":
            last_seq_ts = DataLoader(dataset=last_seq_ts, batch_size= 1, shuffle=False, num_workers=0, **reproduct_args, drop_last=False) #shuffle=shuffle,
        else:
            last_seq_ts = DataLoader(dataset=last_seq_ts, batch_size= 1, shuffle=False, num_workers=0, **reproduct_args, drop_last=False) #shuffle=shuffle,
    else:
        last_seq_ts = None

    print(f"---->{stage} size: {len(mmdataset)}", list(dataAll.index.get_level_values("date").unique().sort_values())[0], list(dataAll.index.get_level_values("date").unique().sort_values())[-1])
    # seq = MyDataset(seq)
    #if not test_pred else 1, drop_last=(not test_pred)
    if args["type"] == "MultiLabelLSTM":
        seq = DataLoader(dataset=mmdataset, batch_size=batch_size if not test_pred else 1, sampler=ImbalancedDatasetSampler(mmdataset), num_workers=0, **reproduct_args, drop_last=(not test_pred)) #shuffle=shuffle,
    else:
        seq = DataLoader(dataset=mmdataset, batch_size=batch_size if not test_pred else 1, shuffle=shuffle, num_workers=0, **reproduct_args,drop_last=(not test_pred)) #shuffle=shuffle,

    # if dump_file == True:
    #     with bz2.BZ2File(seq_pkl_name, "wb") as pkl_file:
    #         pickle.dump(seq, pkl_file)
    with bz2.BZ2File(last_pkl_name, "wb") as pkl_file:
        pickle.dump(last_seq_ts, pkl_file)
    return seq, last_seq_ts

def process_stocks_norm_dataset(dataAll, fsdata, mlpdataset, batch_size, shuffle,data_index_set, test_pred=False, stage="train_test_", dataset_type="PklDataset", dump_file = True):
    global reproduct_args
    args=get_args()

    seq_pkl_name = "torch_stock_" + stage + dataset_type + "_seq_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["output_size"]}-{args["multi_steps"]}' #+ '.pkl'
    last_pkl_name = "torch_stock_" + stage + dataset_type + "_last_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["output_size"]}-{args["multi_steps"]}' + '.pkl'

    if os.path.exists(f"{seq_pkl_name}_mmcfg.pkl"):
        mmdataset = globals()[dataset_type](seq_name=seq_pkl_name)
        if args["type"] == "MultiLabelLSTM":
            seq_pkl = DataLoader(dataset=mmdataset, batch_size=batch_size if not test_pred else 1, sampler=ImbalancedDatasetSampler(mmdataset), num_workers=0, drop_last=(not test_pred)) #shuffle=shuffle,
        else:
            seq_pkl = DataLoader(dataset=mmdataset, batch_size=batch_size if not test_pred else 1, shuffle=shuffle, num_workers=0, drop_last=(not test_pred)) #shuffle=shuffle,
        with bz2.BZ2File(last_pkl_name, "rb") as last_pkl_file:
            last_pkl = pickle.load(last_pkl_file)
        return seq_pkl, last_pkl

    seq_len=get_args()["seq_len"][EXP_LSTM_SEQ_IDX_DK1D]
    steps=args["multi_steps"]
    dataAll = dataAll.rename(columns={"hfq_open": "open", "hfq_high": "high", "hfq_low": "low", "hfq_close": "close"})

    dataAll = dataAll.loc[:, COLS]
    dataAll = dataAll.loc[dataAll.index.get_level_values("code") < "300000"] #"300000"
    dataAll = dataAll.sort_index()

    predstep = steps + seq_len
    seq = []
    last_seq_ts = []

    mmdataset = globals()[dataset_type](seq_name=seq_pkl_name, size=len(dataAll.index),
                                        input_shape=[[(seq_len, len(dataAll.columns)), (EXP_FS_YEARS, len(FS_COLS))], args["input_size"][EXP_MODS_MLP_IDX]],
                                        label_shape=(args["output_size"], ), info_shape=(LABEL_INFO_LEN, ))
    mmdataset_idx = 0

    fscode = fsdata.index.get_level_values("code").unique()

    for code in dataAll.index.get_level_values("code").unique():
        data = dataAll.loc[("szse", code)].sort_index()
        if code not in fscode:
            continue #fixme bank type stock not exist in fsdata
        fsdf = fsdata.loc[("szse", code)].sort_index()
        mlpdf = mlpdataset.loc[("szse", code)] #.sort_index()
        # print("proc code", code)
        train_seqs = []
        train_labels = []
        train_adjs = []
        train_adjs_end = []
        dataLen = 0
        missing_seq = []
        bigrise_seq = []
        infos = []
        last_input_seq = None
        last_label = None
        last_date = None

        # idxvol_idx = list(data.columns).index("idxvol")
        # vod_idx    = list(data.columns).index("vol")
        range_norm_list = []
        for i in range(len(data.columns)):
            if "vol" in data.columns[i]:
                range_norm_list += [i]
        for date in data.index.get_level_values("date").unique().sort_values():
            year_begin = pandas.Timestamp(year=date.year-get_args()["seq_len"][EXP_LSTM_SEQ_IDX_FS1Y], month=date.month,
                                          day=date.day if date.month!=2 else min(date.day, 28))
            fsdfseq = fsdf.loc[(fsdf.index.get_level_values("date")<=date)&(fsdf.index.get_level_values("date")>year_begin)]
            if len(fsdfseq.index) < EXP_FS_YEARS:
                continue
            train_seq = data.loc[date].to_numpy().flatten().tolist()
            # close_topn = data.loc[date, "close"].sort_values(ascending=False).iloc[BK_TOPN]
            # train_label = (data.loc[date, "close"]>=close_topn).to_numpy().flatten().tolist()
            # if args["type"] != "MultiLabelLSTM":
            #     train_label = data.loc[date, "close"].to_numpy().flatten().tolist()
            # else:
            #     close_topn = data.loc[date, "close"].sort_values(ascending=False).iloc[BK_TOPN]
            #     train_label = (data.loc[date, "close"]>=close_topn).to_numpy().flatten().tolist()
            train_seqs += [train_seq]
            train_labels += [[denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_close"]["range"])]]

            if LSTM_ADJUST_START == "O":
                train_adjs += [[1.0/(denorm_fn(data.loc[date, "open"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_open"]["range"])*denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_close"]["range"]))]]
            elif LSTM_ADJUST_START == "L":
                train_adjs += [[1.0/(denorm_fn(data.loc[date, "low"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_low"]["range"])*denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_close"]["range"]))]]
            else:
                train_adjs += [[1.0]]

            if LSTM_ADJUST_END == "O":
                train_adjs_end += [[denorm_fn(data.loc[date, "open"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_open"]["range"])]]
            elif LSTM_ADJUST_END == "L":
                train_adjs_end += [[denorm_fn(data.loc[date, "low"], dfCfgNorm[DATA_SETS[0]]["OHLCV"]["hfq_low"]["range"])]]
            else:
                train_adjs_end += [[1.0]]

            # infos += [[denorm_fn(data.loc[date, "low"], dfCfgNorm[DATA_SETS[0]]["hfq_low"]) <= (1.0/denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["hfq_close"]))*0.99]]
            infos += [[True]]

            # train_adjs += [[denorm_fn(data.loc[date, "close"], dfCfgNorm[DATA_SETS[0]]["hfq_close"])/denorm_fn(data.loc[date, "open"], dfCfgNorm[DATA_SETS[0]]["hfq_open"])]]

            if data.loc[date].sum(skipna=False) != data.loc[date].sum(skipna=False):  #fixme performance
                missing_seq += [True]
            else:
                missing_seq += [False]

            if data.loc[date, "close"] > BIG_RISE:
                bigrise_seq += [True]
            else:
                bigrise_seq += [False]

            dataLen += 1
            if dataLen >= predstep:
                npa = np.array(train_seqs[dataLen - predstep:dataLen - predstep+seq_len])
                if RANGE_NORM == True:
                    for cidx in range_norm_list:
                        npa[:, cidx] = (2*npa[:, cidx] - np.max(npa[:, cidx]) - np.min(npa[:, cidx]))/(np.max(npa[:, cidx]) - np.min(npa[:, cidx]))
                #train_seq_ts = torch.FloatTensor(npa)
                #if False: #steps == 1:
                #    train_label_ts = torch.FloatTensor(train_adjs[-steps]).view(-1)
                #else:
                #    train_label_ts = torch.FloatTensor(train_adjs[-steps] * np.array(train_labels)[-steps:].prod(axis=0)*train_adjs_end[-1]).view(-1)

                if any(missing_seq[dataLen - predstep:dataLen - predstep+seq_len]) ==False and missing_seq[-steps]==False and any(missing_seq[-steps+1:])==False \
                        and any(bigrise_seq[-RISE_WIN-steps:-steps]):
                    # seq.append((train_seq_ts, train_label_ts, code, date.value, infos[-steps]))
                    last_input_seq = [[npa, fsdfseq.values], mlpdf.loc[date].values]
                    last_label = train_adjs[-steps] * np.array(train_labels)[-steps:].prod(axis=0)*train_adjs_end[-1]
                    last_date = date
                    mmdataset.map(mmdataset_idx, last_input_seq, last_label,
                                                       np.array([float(code), float(last_date.value), float(infos[-steps][0])]))
                    mmdataset_idx += 1
                else:
                    pass
                    #print("missing", sum(missing_seq[dataLen - predstep:dataLen - predstep+seq_len]))
        if last_input_seq is not None and any(bigrise_seq[-RISE_WIN:]):
            # train_label_ts = torch.FloatTensor(train_adjs[-steps] * np.array(train_labels)[-steps:].prod(axis=0)*train_adjs_end[-1]).view(-1)
            # np.empty(0) if no mlp input
            last_seq_ts += [(tuple(last_input_seq[EXP_MODS_LSTM_IDX]), mlpdf.loc[date].values, last_label, code, last_date.value)] #todo train_label_ts is not the true label of future seq, but doesn't matter
        else:
            pass
            #print("missing")
    mmdataset.save()
    if test_pred == True:
        last_seq_ts = MyDataset(last_seq_ts)
        #if not test_pred else 1, drop_last=(not test_pred)
        if args["type"] == "MultiLabelLSTM":
            last_seq_ts = DataLoader(dataset=last_seq_ts, batch_size= 1, shuffle=False, num_workers=0, **reproduct_args, drop_last=False) #shuffle=shuffle,
        else:
            last_seq_ts = DataLoader(dataset=last_seq_ts, batch_size= 1, shuffle=False, num_workers=0, **reproduct_args, drop_last=False) #shuffle=shuffle,
    else:
        last_seq_ts = None

    print(f"---->{stage} size: {len(mmdataset)}", list(dataAll.index.get_level_values("date").unique().sort_values())[0], list(dataAll.index.get_level_values("date").unique().sort_values())[-1])
    # seq = MyDataset(seq)
    #if not test_pred else 1, drop_last=(not test_pred)
    if args["type"] == "MultiLabelLSTM":
        seq = DataLoader(dataset=mmdataset, batch_size=batch_size if not test_pred else 1, sampler=ImbalancedDatasetSampler(mmdataset), num_workers=0, **reproduct_args, drop_last=(not test_pred)) #shuffle=shuffle,
    else:
        seq = DataLoader(dataset=mmdataset, batch_size=batch_size if not test_pred else 1, shuffle=shuffle, num_workers=0, **reproduct_args,drop_last=(not test_pred)) #shuffle=shuffle,

    # if dump_file == True:
    #     with bz2.BZ2File(seq_pkl_name, "wb") as pkl_file:
    #         pickle.dump(seq, pkl_file)
    with bz2.BZ2File(last_pkl_name, "wb") as pkl_file:
        pickle.dump(last_seq_ts, pkl_file)
    return seq, last_seq_ts

#todo: for C1/O0
def process_stocks(dataAll, batch_size, shuffle,data_index_set, test_pred=False):
    args=get_args()

    seq_pkl_name = "torch_stock_seq_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}-{args["multi_steps"]}' + '.pkl'
    last_pkl_name = "torch_stock_last_" + str(len(dataAll)) + f"_{dataAll.index[-1][1]}_{dataAll.index[-1][2].date()}_" + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}-{args["multi_steps"]}' + '.pkl'


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
        print("proc code", code)
        train_seqs = []
        train_labels = []
        train_adjs = []
        dataLen = 0
        missing_seq = []

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
            train_adjs += [[-data.loc[date, "open"]/np.log(dfCfg[DATA_FN_KEY]["hfq_close"])*np.log(dfCfg[DATA_FN_KEY]["hfq_open"][0])]]

            if data.loc[date]["open"] == 0.0 and data.loc[date]["close"]==0.0 and data.loc[date]["high"]==-1.0 and data.loc[date]["low"]==1.0 \
                and data.loc[date]["vol"]==-1.0:
                missing_seq += [True]
            else:
                missing_seq += [False]

            dataLen += 1
            if dataLen >= predstep:
                train_seq_ts = torch.FloatTensor(train_seqs[dataLen - predstep:dataLen - predstep+seq_len])
                # train_label_ts = torch.FloatTensor(train_labels[-1]).view(-1)
                if steps == 1:
                    train_label_ts = torch.FloatTensor(train_adjs[-steps]).view(-1)
                else:
                    train_label_ts = torch.FloatTensor(train_adjs[-steps] + np.array(train_labels)[-steps+1:].sum(axis=0)).view(-1)
                if any(missing_seq[dataLen - predstep:dataLen - predstep+seq_len]) == False and missing_seq[-steps]==False and any(missing_seq[-steps+1:])==False:
                    seq.append((train_seq_ts, train_label_ts, code, date.value))
                else:
                    print("missing", sum(missing_seq[dataLen - predstep:dataLen - predstep+seq_len]))
        if any(missing_seq[-seq_len:]) == False:
            last_seq_ts += [(torch.FloatTensor(train_seqs[-seq_len:]), train_label_ts, code, (date + datetime.timedelta(days=steps)).value)] #todo train_label_ts is not the true label of future seq, but doesn't matter
        else:
            print("missing")
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
def nn_stocksdata_seq_split_by_code_n_date(batch_size, lstmtype):
    print('data processing...')

    if NO_TRAIN == False or NO_TEST == False:
        data_file_name = DATA_ALL_FN
        dataset = load_stocks_data(data_file_name)
        dateFirstIndex = dataset.reset_index().set_index(["date", "exchange", "code"]).sort_index().index
        # dataset = dataset.loc[(slice(None), slice(None), BKS), COLS].sort_index()

        algs=get_args()
        #check number of data items in df, if it is too less , we can not train it.
        algs["train_end"]=0.7
        algs["val_begin"]=0.7
        algs["val_end"]=0.8
        algs["test_begin"]=0.8
        print("fixme algs")

        all_code_len = len(dateFirstIndex)
        train_date_end = dateFirstIndex[int(all_code_len*algs["train_end"])][0]
        val_date_end = dateFirstIndex[int(all_code_len*algs["val_end"])][0]
        print("train_end", train_date_end, "val_end", val_date_end)

    if NO_TRAIN == False:
        if os.path.exists(DATA_TRAIN_FN) and False:
            train = load_stocks_data(DATA_TRAIN_FN)
        else:
            print("missing", DATA_TRAIN_FN)
            # train_val = dataset.loc[dataset.index.get_level_values("date")<=train_date_end]
            train = dataset.loc[(dataset.index.get_level_values(1).astype(int)%7>0) & (dataset.index.get_level_values("date")<=train_date_end)]
            val0 = dataset.loc[(dataset.index.get_level_values(1).astype(int)%7==0) & (dataset.index.get_level_values("date")<=train_date_end)]
        # if os.path.exists(DATA_VAL_FN) and False:
        #     val = load_stocks_data(DATA_VAL_FN)
        # else:
        #     print("missing", DATA_VAL_FN)
        #     val = dataset.loc[(dataset.index.get_level_values("date")>train_date_end) & (dataset.index.get_level_values("date")<=val_date_end)]
        val1 = dataset.loc[(dataset.index.get_level_values("date")>train_date_end) & (dataset.index.get_level_values("date")<=val_date_end)]
        val = pandas.concat([val0, val1])

    if NO_TEST == False:
        if os.path.exists(DATA_TEST_FN) and False:
            test = load_stocks_data(DATA_TEST_FN)
        else:
            print("missing", DATA_TEST_FN)
            test = dataset.loc[dataset.index.get_level_values("date")>val_date_end]

    if os.path.exists(DATA_PRED_FN):
        pred = load_stocks_data(DATA_PRED_FN)
    else:
        pred = None

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

    #dataset. process_stocks_O2toO1 process_stocks_norm_c2c1
    if NO_TRAIN == False:
        Dtr, _ = process_stocks_norm(train, batch_size, True,data_index_set, stage="train_")
        Val, _ = process_stocks_norm(val,   batch_size, True,data_index_set, stage="val_")
        # Dtrval, _ = process_stocks_norm(train_val, batch_size, True,data_index_set)
        # generator = torch.Generator().manual_seed(42)
        # Dtr, Val =  random_split(Dtrval, [0.75, 0.25], generator)
        # Dtr = Dtr.dataset
        # Val = Val.dataset
        # Dtr, _ = process_stocks_norm(train, batch_size, True,data_index_set)
        # Val, _ = process_stocks_norm(val,   batch_size, True,data_index_set)
        # Dtr, _ = process_stocks_O2toO1(train, batch_size, True,data_index_set)
        # Val, _ = process_stocks_O2toO1(val,   batch_size, True,data_index_set)
    else:
        Dtr = None
        Val = None

    if NO_TEST == False:
        # Dte, _ = process_stocks_norm_c2c1(test,  1, False,data_index_set)
        # _, last_seq_ts = process_stocks_norm_c2c1(pred,  batch_size, False,data_index_set, test_pred=True)
        Dte, _ = process_stocks_norm(test,  1, False,data_index_set, stage="test_")
        _, last_seq_ts = process_stocks_norm(pred,  batch_size, False,data_index_set, test_pred=True, stage="pred_")
        # Dte, _ = process_stocks_O2toO1(test,  1, False,data_index_set)
        # _, last_seq_ts = process_stocks_O2toO1(pred,  batch_size, False,data_index_set, test_pred=True)
    else:
        # Dte = None
        # Dte, last_seq_ts = process_stocks_norm_c2c1(pred,  batch_size, False,data_index_set, test_pred=True)
        Dte, last_seq_ts = process_stocks_norm(pred,  batch_size, False,data_index_set, test_pred=True, stage="pred_")
        # Dte, last_seq_ts = process_stocks_O2toO1(pred,  batch_size, False,data_index_set, test_pred=True)

    return Dtr, Val, Dte, last_seq_ts, pred

# split date and create datasets for train /validate and test.
def nn_stocksdata_seq_split_by_code(batch_size, lstmtype):
    print('data processing...')

    if NO_TRAIN == False or NO_TEST == False:
        data_file_name = DATA_ALL_FN
        dataset = load_stocks_data(data_file_name)
        dateFirstIndex = dataset.reset_index().set_index(["date", "exchange", "code"]).sort_index().index
        # dataset = dataset.loc[(slice(None), slice(None), BKS), COLS].sort_index()

        algs=get_args()
        #check number of data items in df, if it is too less , we can not train it.
        algs["train_end"]=0.7
        algs["val_begin"]=0.6
        algs["val_end"]=0.8
        algs["test_begin"]=0.8
        print("fixme algs")

        all_code_len = len(dateFirstIndex)
        train_date_end = dateFirstIndex[int(all_code_len*algs["train_end"])][0]
        val_date_end = dateFirstIndex[int(all_code_len*algs["val_end"])][0]
        print("train_end", train_date_end, "val_end", val_date_end)

    if NO_TRAIN == False:
        if os.path.exists(DATA_TRAIN_FN) and False:
            train = load_stocks_data(DATA_TRAIN_FN)
        else:
            print("missing", DATA_TRAIN_FN)
            train_val = dataset.loc[dataset.index.get_level_values("date")<=val_date_end]
        # if os.path.exists(DATA_VAL_FN) and False:
        #     val = load_stocks_data(DATA_VAL_FN)
        # else:
        #     print("missing", DATA_VAL_FN)
        #     val = dataset.loc[(dataset.index.get_level_values("date")>train_date_end) & (dataset.index.get_level_values("date")<=val_date_end)]

    if NO_TEST == False:
        if os.path.exists(DATA_TEST_FN) and False:
            test = load_stocks_data(DATA_TEST_FN)
        else:
            print("missing", DATA_TEST_FN)
            test = dataset.loc[dataset.index.get_level_values("date")>val_date_end]

    if os.path.exists(DATA_PRED_FN):
        pred = load_stocks_data(DATA_PRED_FN)
    else:
        pred = None

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

    #dataset. process_stocks_O2toO1 process_stocks_norm_c2c1
    if NO_TRAIN == False:
        Dtr, _ = process_stocks_norm(train_val.loc[train_val.index.get_level_values(1).astype(int)%4>0], batch_size, True,data_index_set, stage="train_")
        Val, _ = process_stocks_norm(train_val.loc[train_val.index.get_level_values(1).astype(int)%4==0],   batch_size, True,data_index_set, stage="val_")
        # Dtrval, _ = process_stocks_norm(train_val, batch_size, True,data_index_set)
        # generator = torch.Generator().manual_seed(42)
        # Dtr, Val =  random_split(Dtrval, [0.75, 0.25], generator)
        # Dtr = Dtr.dataset
        # Val = Val.dataset
        # Dtr, _ = process_stocks_norm(train, batch_size, True,data_index_set)
        # Val, _ = process_stocks_norm(val,   batch_size, True,data_index_set)
        # Dtr, _ = process_stocks_O2toO1(train, batch_size, True,data_index_set)
        # Val, _ = process_stocks_O2toO1(val,   batch_size, True,data_index_set)
    else:
        Dtr = None
        Val = None

    if NO_TEST == False:
        # Dte, _ = process_stocks_norm_c2c1(test,  1, False,data_index_set)
        # _, last_seq_ts = process_stocks_norm_c2c1(pred,  batch_size, False,data_index_set, test_pred=True)
        Dte, _ = process_stocks_norm(test,  1, False,data_index_set, stage="test_")
        _, last_seq_ts = process_stocks_norm(pred,  batch_size, False,data_index_set, test_pred=True, stage="pred_")
        # Dte, _ = process_stocks_O2toO1(test,  1, False,data_index_set)
        # _, last_seq_ts = process_stocks_O2toO1(pred,  batch_size, False,data_index_set, test_pred=True)
    else:
        # Dte = None
        # Dte, last_seq_ts = process_stocks_norm_c2c1(pred,  batch_size, False,data_index_set, test_pred=True)
        Dte, last_seq_ts = process_stocks_norm(pred,  batch_size, False,data_index_set, test_pred=True, stage="pred_")
        # Dte, last_seq_ts = process_stocks_O2toO1(pred,  batch_size, False,data_index_set, test_pred=True)

    return Dtr, Val, Dte, last_seq_ts, pred

# split date and create datasets for train /validate and test.
def nn_stocksdata_seq_random_split(batch_size, lstmtype):
    print('data processing...')

    if NO_TRAIN == False or NO_TEST == False:
        data_file_name = DATA_ALL_FN
        dataset = load_stocks_data(data_file_name)
        dateFirstIndex = dataset.reset_index().set_index(["date", "exchange", "code"]).sort_index().index
        # dataset = dataset.loc[(slice(None), slice(None), BKS), COLS].sort_index()

        algs=get_args()
        #check number of data items in df, if it is too less , we can not train it.
        algs["train_end"]=0.7
        algs["val_begin"]=0.6
        algs["val_end"]=0.8
        algs["test_begin"]=0.8
        print("fixme algs")

        all_code_len = len(dateFirstIndex)
        train_date_end = dateFirstIndex[int(all_code_len*algs["train_end"])][0]
        val_date_end = dateFirstIndex[int(all_code_len*algs["val_end"])][0]
        print("train_end", train_date_end, "val_end", val_date_end)

    if NO_TRAIN == False:
        if os.path.exists(DATA_TRAIN_FN) and False:
            train = load_stocks_data(DATA_TRAIN_FN)
        else:
            print("missing", DATA_TRAIN_FN)
            train_val = dataset.loc[dataset.index.get_level_values("date")<=val_date_end]
        # if os.path.exists(DATA_VAL_FN) and False:
        #     val = load_stocks_data(DATA_VAL_FN)
        # else:
        #     print("missing", DATA_VAL_FN)
        #     val = dataset.loc[(dataset.index.get_level_values("date")>train_date_end) & (dataset.index.get_level_values("date")<=val_date_end)]

    if NO_TEST == False:
        if os.path.exists(DATA_TEST_FN) and False:
            test = load_stocks_data(DATA_TEST_FN)
        else:
            print("missing", DATA_TEST_FN)
            test = dataset.loc[dataset.index.get_level_values("date")>val_date_end]

    if os.path.exists(DATA_PRED_FN):
        pred = load_stocks_data(DATA_PRED_FN)
    else:
        pred = None

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

    #dataset. process_stocks_O2toO1 process_stocks_norm_c2c1
    if NO_TRAIN == False:
        # Dtr, _ = process_stocks_norm_c2c1(train, batch_size, True,data_index_set)
        # Val, _ = process_stocks_norm_c2c1(val,   batch_size, True,data_index_set)
        Dtrval, _ = process_stocks_norm(train_val, batch_size, True,data_index_set)
        generator = torch.Generator().manual_seed(42)
        Dtr, Val =  random_split(Dtrval, [0.75, 0.25], generator)
        Dtr = Dtr.dataset
        Val = Val.dataset
        # Dtr, _ = process_stocks_norm(train, batch_size, True,data_index_set)
        # Val, _ = process_stocks_norm(val,   batch_size, True,data_index_set)
        # Dtr, _ = process_stocks_O2toO1(train, batch_size, True,data_index_set)
        # Val, _ = process_stocks_O2toO1(val,   batch_size, True,data_index_set)
    else:
        Dtr = None
        Val = None

    if NO_TEST == False:
        # Dte, _ = process_stocks_norm_c2c1(test,  1, False,data_index_set)
        # _, last_seq_ts = process_stocks_norm_c2c1(pred,  batch_size, False,data_index_set, test_pred=True)
        Dte, _ = process_stocks_norm(test,  1, False,data_index_set)
        _, last_seq_ts = process_stocks_norm(pred,  batch_size, False,data_index_set, test_pred=True)
        # Dte, _ = process_stocks_O2toO1(test,  1, False,data_index_set)
        # _, last_seq_ts = process_stocks_O2toO1(pred,  batch_size, False,data_index_set, test_pred=True)
    else:
        # Dte = None
        # Dte, last_seq_ts = process_stocks_norm_c2c1(pred,  batch_size, False,data_index_set, test_pred=True)
        Dte, last_seq_ts = process_stocks_norm(pred,  batch_size, False,data_index_set, test_pred=True)
        # Dte, last_seq_ts = process_stocks_O2toO1(pred,  batch_size, False,data_index_set, test_pred=True)

    return Dtr, Val, Dte, last_seq_ts, pred

# split date and create datasets for train /validate and test.
def nn_stocksdata_seq(batch_size, lstmtype):
    print('data processing...')

    if NO_TRAIN == False or NO_TEST == False:
        data_file_name = DATA_ALL_FN
        dataset = load_stocks_data(data_file_name).astype('float32')
        dateFirstIndex = dataset.reset_index().set_index(["date", "exchange", "code"]).sort_index().index
        # dataset = dataset.loc[(slice(None), slice(None), BKS), COLS].sort_index()

        fsdataset = load_stock_fs_date(DATA_FS_FN)
        mlpdataset = pandas.read_hdf("exp_mv.hdf", "mv")

        algs=get_args()
        #check number of data items in df, if it is too less , we can not train it.
        algs["train_end"]=0.515
        algs["val_begin"]=0.515
        algs["val_end"]=0.725
        algs["test_begin"]=0.725
        print("fixme algs")

        all_code_len = len(dateFirstIndex)
        train_date_end = dateFirstIndex[int(all_code_len*algs["train_end"])][0]
        val_date_end = dateFirstIndex[int(all_code_len*algs["val_end"])][0]
        print("train_end", train_date_end, "val_end", val_date_end)

    if NO_TRAIN == False:
        train = dataset.loc[dataset.index.get_level_values("date")<=train_date_end]
        val = dataset.loc[(dataset.index.get_level_values("date")>train_date_end) & (dataset.index.get_level_values("date")<=val_date_end)]
        # fs_train = fsdataset.loc[fsdataset.index.get_level_values("date")<=train_date_end]
        # fs_val = fsdataset.loc[(fsdataset.index.get_level_values("date")>train_date_end) & (fsdataset.index.get_level_values("date")<=val_date_end)]

    if NO_TEST == False:
        args=get_args()
        test = dataset.loc[dataset.index.get_level_values("date")>val_date_end]
        if MAX_SEQ_LEN < args["seq_len"][EXP_LSTM_SEQ_IDX_DK1D]:
            print("overflow")
            sys.exit()
        test_dates = test.index.get_level_values("date").unique()[MAX_SEQ_LEN-args["seq_len"][EXP_LSTM_SEQ_IDX_DK1D]]
        test = test.loc[test.index.get_level_values("date")>test_dates]

    if os.path.exists(DATA_PRED_FN):
        pred = load_stocks_data(DATA_PRED_FN)
    else:
        pred = None

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

    #dataset. process_stocks_O2toO1 process_stocks_norm_c2c1
    if NO_TRAIN == False:
        # Dtr, _ = process_stocks_norm_c2c1(train, batch_size, True,data_index_set)
        # Val, _ = process_stocks_norm_c2c1(val,   batch_size, True,data_index_set)
        Dtr, _ = process_stocks_norm_dataset(train, fsdataset, mlpdataset, batch_size, True,data_index_set, stage="train_", dataset_type="MMDataset", dump_file=DUMP_DATASET_TRAIN_VAL_PKL)  #process_stocks_norm_mmdataset
        Val, _ = process_stocks_norm_dataset(val,   fsdataset, mlpdataset, batch_size, True,data_index_set, stage="val_", dataset_type="MMDataset", dump_file=DUMP_DATASET_TRAIN_VAL_PKL)
        # Dtr, _ = process_stocks_O2toO1(train, batch_size, True,data_index_set)
        # Val, _ = process_stocks_O2toO1(val,   batch_size, True,data_index_set)
    else:
        Dtr = None
        Val = None

    if NO_TEST == False:
        # Dte, _ = process_stocks_norm_c2c1(test,  1, False,data_index_set)
        # _, last_seq_ts = process_stocks_norm_c2c1(pred,  batch_size, False,data_index_set, test_pred=True)
        Dte, _ = process_stocks_norm_dataset(test,  fsdataset, mlpdataset, 1, False,data_index_set, stage="test_", dataset_type="MMDataset")
        _, last_seq_ts = process_stocks_norm_dataset(pred,  fsdataset, mlpdataset, batch_size, False,data_index_set, test_pred=True, stage="pred_", dataset_type="MMDataset")
        # Dte, _ = process_stocks_O2toO1(test,  1, False,data_index_set)
        # _, last_seq_ts = process_stocks_O2toO1(pred,  batch_size, False,data_index_set, test_pred=True)
    else:
        # Dte = None
        # Dte, last_seq_ts = process_stocks_norm_c2c1(pred,  batch_size, False,data_index_set, test_pred=True)
        Dte, last_seq_ts = process_stocks_norm_dataset(pred,  fsdataset, mlpdataset, batch_size, False,data_index_set, test_pred=True, stage="pred_", dataset_type="MMDataset")
        # Dte, last_seq_ts = process_stocks_O2toO1(pred,  batch_size, False,data_index_set, test_pred=True)

    return Dtr, Val, Dte, last_seq_ts, pred

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

class LossLogMse(nn.Module):
    def __init__(self):
        super(LossLogMse, self).__init__()

    def forward(self, inputs, targets):
        loss = (torch.pow(math.e, inputs) - torch.pow(math.e, targets))**2
        return loss.mean()

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

class MLPX(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            *(chain.from_iterable([[nn.Linear(hidden_size, hidden_size), nn.Dropout(0.5), nn.ReLU()] for i in range(num_layers)])),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, input_seq):
        # input_seq = input_seq.view(input_seq.size(0), -1)
        input_seq = torch.flatten(input_seq, start_dim=1, end_dim=2)
        output = self.mlp(input_seq) # output(5, 30, 64)
        return output

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size

        h0 = torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size).to(device)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)  # Parameter() to update weights
        self.c0 = nn.Parameter(c0, requires_grad=True)

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # h_0 = torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).requires_grad_()
        # c_0 = torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).requires_grad_()
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1))) # output(5, 30, 64)
        pred = self.linear(output)  # (5, 24, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred

#nn module.
class LSTM_noc0h0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        h_0 = torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).requires_grad_()
        c_0 = torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).requires_grad_()
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0.detach(), c_0.detach())) # output(5, 30, 64)
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

def pred_stat(lastbest_pred_target, stage="training"):
    json_cfg = { "threshold": -1 }
    dfList = []
    for s in lastbest_pred_target:
        df = pandas.DataFrame({"pred":s[0], "target": s[1]})
        dfList += [df]
    dfAll = pandas.concat(dfList)
    print(len(dfAll.index))
    dfAll = dfAll.sort_values("pred")
    binsize = 20
    dfPredMin = dfAll.pred.min()
    dfPredMax = dfAll.pred.max()

    stepSize = (dfPredMax - dfPredMin)/binsize
    vStat = { "vMin":[], "vMax": [], "ava": [], "winPrec": [], "perc": [], "ava2e": [], "winPrec2e": [], "perc2e": [], "smpNo": []}

    for i in range(binsize+1):
        vMin = dfPredMin+i*stepSize
        vMax = dfPredMin+i*stepSize+stepSize
        df = dfAll.loc[(dfAll.pred>=vMin)&(dfAll.pred<vMax)]
        df2e = dfAll.loc[(dfAll.pred>=vMin)]
        vStat["vMin"] += [vMin]
        vStat["vMax"] += [vMax]
        vStat["ava"] += [df.target.mean()]
        if len(df.index) > 0:
            vStat["winPrec"] += [sum(df.target>1)/len(df.index)]
        else:
            vStat["winPrec"] += [math.nan]
        vStat["perc"] += [len(df)/len(dfAll)]

        vStat["ava2e"] += [df2e.target.mean()]
        if len(df2e.index) > 0:
            vStat["winPrec2e"] += [sum(df2e.target>1)/len(df2e.index)]
        else:
            vStat["winPrec2e"] += [math.nan]
        vStat["perc2e"] += [len(df2e)/len(dfAll)]

        vStat["smpNo"] += [len(df)]

    dfStat = pandas.DataFrame(vStat)
    print(dfStat)

    opDf = dfStat.loc[(dfStat.index>=(len(dfStat.index)/2))&(dfStat.ava>1.006)&(dfStat.winPrec>0.55)]
    opMin = opDf.vMin.min()
    if os.path.exists("torch_stock.cfg"):
        with open('torch_stock.cfg') as cfg_file:
            json_cfg = json.load(cfg_file)

    if "threshold" in json_cfg and json_cfg["threshold"] > 0:
        opMin = json_cfg["threshold"]
    if opMin != opMin:
        opMin = 1.01
    print(stage, "threshold is", opMin)
    return opMin

#train it.
def train(args, Dtr, Val, paths, Dte, last_seq_ts):
    args=get_args()
    input_size, hidden_size, num_layers = args['input_size'], args['hidden_size'], args['num_layers']
    merged_hidden_size, merged_num_mid_layers = args['merged_hidden_size'], args['merged_num_mid_layers']
    output_size = args['output_size']
    seq_len = args['seq_len']
    
    if args["type"] == 'BiLSTM':
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)
    elif args["type"] == 'LSTM':
        model = LSTM(  input_size, hidden_size, num_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)
        # loss_function = LossLogMse().to(device)
    if args["type"] == 'LSTMs':
        model = LSTMs(  input_size, hidden_size, num_layers, merged_hidden_size, merged_num_mid_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)
    elif args["type"] == 'MLPX':
        model = MLPX(  input_size*seq_len, hidden_size*seq_len, num_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)
        # loss_function = LossLogMse().to(device)
    elif args["type"] == 'RA':
        model = LSTMAttention(input_size, output_size, seq_len, hidden_size, num_layers,  rnn_dropout=0.5, encoder_dropout=0.5, fc_dropout=0.5).to(device)
        loss_function = nn.MSELoss().to(device)
        # loss_function = LossLogMse().to(device)
    else:
        model = MultiLabelLSTM(  input_size, hidden_size, num_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.BCELoss().to(device)

    if args['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'],momentum=0.9, weight_decay=args['weight_decay'])
        
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    path, cur_sav_idx = get_module_saved_path(paths, load_best=MODULE_LOAD_BEST)
    if os.path.exists(path):
        print('loading models...')
        model.load_state_dict(torch.load(path)['models'])

    # best_model=copy.deepcopy(model)
    # state = {'models': best_model.state_dict()}
    # print('Saving models...\r\n', flush=True)
    # cur_sav_idx += 1
    # torch.save(state, paths[cur_sav_idx%(len(paths))])
    # cur_sav_idx += 1
    # torch.save(state, paths[cur_sav_idx%(len(paths))])

    # training
    best_model = None
    train_loss_all=[-1.0]
    val_loss_all=[]
    best_loss=None
    # initialize the early_stopping object
    # early_stopping = EarlyStopping(patience=TCH_EARLYSTOP_PATIENCE, verbose=True)
    patience = 0

    lastbest_pred_target = []
    currbest_pred_target = []
    r2score = R2Score()

    print('training...')
    for epoch in tqdm(range(args['epochs'])):
        # validation
        val_loss=0
        num_item=0
        targets = []
        model_result = []
        model.eval()
        for (seqs, mlp_in, label, _, _, _) in Val:

            # seq = seq.to(device)
            label = label.to(device)
            inputs = [[lstmseq.to(device) for lstmseq in seqs]]
            if input_size[EXP_MODS_MLP_IDX] > 0:
                inputs += [mlp_in]
            y_pred = model(inputs)
            # y_pred = model([[lstmseq.to(device) for lstmseq in seqs[EXP_MODS_LSTM_IDX]], [mlpseq.to(device) for mlpseq in seqs[EXP_MODS_MLP_IDX]]])
            currbest_pred_target += [(y_pred.detach().cpu().numpy().flatten(), label.detach().cpu().numpy().flatten())]
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
            lastbest_pred_target = currbest_pred_target
            currbest_pred_target = []

        if args["type"] == "MultiLabelLSTM":
            result = calculate_metrics(np.array(model_result), np.array(targets))
            for key in result:
                print(key)
                print(result[key])
        else:
            topnidx = np.array(model_result).argsort(axis=0)[-NP_TOPN:, :]
            topnclose = np.take(targets, topnidx)
            print("\nAverage close is:", topnclose.mean(), np.mean(targets))
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f} best_loss {:.8f} R2 {:.4f} patience {:04d}'.format(epoch, train_loss_all[-1], val_loss_all[-1], best_loss, r2score(torch.tensor(np.array(model_result)), torch.tensor(np.array(targets))), patience), flush=True)

        #get the best model.
        if(val_loss_all[-1]<best_loss):
            best_loss=val_loss_all[-1]
            best_model=copy.deepcopy(model)
            state = {'models': best_model.state_dict()}
            cur_sav_idx += 1
            print('Saving models...\r\n', paths[cur_sav_idx%(len(paths))], flush=True)
            torch.save(state, paths[cur_sav_idx%(len(paths))])
            patience = 0
            lastbest_pred_target = currbest_pred_target
            currbest_pred_target = []
        elif val_loss_all[-1]>best_loss:
            patience += 1
            if patience > TCH_EARLYSTOP_PATIENCE:
                break

        global TEST_FLAG
        if TEST_FLAG == True or NO_TRAIN == True:
            buy_threshold = pred_stat(lastbest_pred_target)
            test(None, Dte, path_file, None, last_seq_ts, None, buy_threshold)
            TEST_FLAG = False

            if NO_TRAIN == True:
                return buy_threshold

        train_loss = 0
        num_item=0
        model.train()
        for (seqs, mlp_in, label, _, _, _) in Dtr:
            # seq = seq.to(device)
            label = label.to(device)
            inputs = [[lstmseq.to(device) for lstmseq in seqs]]
            if input_size[EXP_MODS_MLP_IDX] > 0:
                inputs += [mlp_in]
            y_pred = model(inputs)
            # y_pred = model([[lstmseq.to(device) for lstmseq in seqs[EXP_MODS_LSTM_IDX]], [mlpseq.to(device) for mlpseq in seqs[EXP_MODS_MLP_IDX]]])
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

    #save the best model.
    # state = {'models': best_model.state_dict()}
    # torch.save(state, path)
    
    # plot for loss.
    if plot and False:
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

    return pred_stat(lastbest_pred_target)

def update_drawback(high, low, profit, maxdrawback):
    if high < profit:
        high = profit
        low = high
    if low > profit:
        low = profit
        drawback = (high-low)/high
        if drawback > maxdrawback:
            maxdrawback = drawback
    return high, low, maxdrawback

def get_module_saved_path(paths, load_best=True):
    ping_mtime = os.path.getmtime(paths[MODULE_SAVED_PING]) if os.path.exists(paths[MODULE_SAVED_PING]) else 0
    pong_mtime = os.path.getmtime(paths[MODULE_SAVED_PONG]) if os.path.exists(paths[MODULE_SAVED_PONG]) else 0

    MODULE_CURRENT_SAVED = ((ping_mtime > pong_mtime)^load_best)
    path = paths[MODULE_CURRENT_SAVED]
    print("load_best:", load_best, "ping is newest:", (ping_mtime > pong_mtime), "load:", path)
    return path, MODULE_CURRENT_SAVED

#validate
def test(args, Dte, paths,data_pred_index, last_seq_ts, testdf, buy_threshold=math.nan):
    
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
    merged_hidden_size, merged_num_mid_layers = args['merged_hidden_size'], args['merged_num_mid_layers']
    output_size = args['output_size']
    seq_len = args['seq_len']

    if args['type'] == "BiLSTM": #lstm, bidirection-lstm, multilabel-lstm
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=1).to(device)
    elif args['type'] == "LSTM":
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=1).to(device)
    if args["type"] == 'LSTMs':
        model = LSTMs(  input_size, hidden_size, num_layers, merged_hidden_size, merged_num_mid_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)
    elif args["type"] == 'MLPX':
        model = MLPX(  input_size*seq_len, hidden_size*seq_len, num_layers, output_size, batch_size=args['batch_size']).to(device)
    elif args["type"] == 'RA':
        model = LSTMAttention(input_size, output_size, seq_len, hidden_size, num_layers,  rnn_dropout=0.5, encoder_dropout=0.5, fc_dropout=0.5).to(device)
        # loss_function = nn.MSELoss().to(device)
        # loss_function = LossLogMse().to(device)
    else:
        model = MultiLabelLSTM(input_size, hidden_size, num_layers, output_size, batch_size=1).to(device)

    path, _ = get_module_saved_path(paths)

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

    r2score = R2Score()
    if not NO_TEST:
        targets_dates = {}
        model_result_dates = {}
        code_dates = {}
        targets = []
        model_result = []
        infos = {}
        currbest_pred_target = []
        for (seqs, mlp_in, target, code, date, info) in tqdm(Dte):
            date = date.item()
            if date not in targets_dates:
                model_result_dates[date] = []
                targets_dates[date] = []
                code_dates[date] = []
                infos[date] = []
            y=np.append(y,target.numpy(),axis=0)
            # seq = seq.to(device)
            with torch.no_grad():
                inputs = [[lstmseq.to(device) for lstmseq in seqs]]
                if input_size[EXP_MODS_MLP_IDX] > 0:
                    inputs += [mlp_in]
                y_pred = model(inputs)
                # y_pred = model([[lstmseq.to(device) for lstmseq in seqs[EXP_MODS_LSTM_IDX]], [mlpseq.to(device) for mlpseq in seqs[EXP_MODS_MLP_IDX]]])
                # model_result_dates[date].extend( np.power(dfCfg[DATA_FN_KEY]['hfq_close'], y_pred.detach().cpu().numpy()) )
                # targets_dates[date].extend( np.power(dfCfg[DATA_FN_KEY]['hfq_close'], target.detach().cpu().numpy()) )
                # code_dates[date].extend( code )
                # model_result.extend( np.power(dfCfg[DATA_FN_KEY]['hfq_close'], y_pred.detach().cpu().numpy()) )
                # targets.extend( np.power(dfCfg[DATA_FN_KEY]['hfq_close'], target.detach().cpu().numpy()) )
                currbest_pred_target += [(y_pred.detach().cpu().numpy().flatten(), target.detach().cpu().numpy().flatten())]
                model_result_dates[date].extend( y_pred.detach().cpu().numpy() )
                targets_dates[date].extend( target.detach().cpu().numpy() )
                code_dates[date].extend( code )
                model_result.extend( y_pred.detach().cpu().numpy() )
                targets.extend( target.detach().cpu().numpy() )
                if isinstance(info[0], torch.Tensor):
                    infos[date].extend([info[0].item()])
                else:
                    infos[date].extend(info)

            if len(model_result) == 100 and False:
                plt.plot(np.array(model_result).T, np.array(targets).T)
                plt.show()
        print("r2socre", r2score(torch.tensor(model_result), torch.tensor(targets)))
        buy_threshold_calc = pred_stat(currbest_pred_target, stage="test")

        if buy_threshold != buy_threshold:
            buy_threshold = buy_threshold_calc

        idx = 0
        target_means = []
        target_topn_means = []
        target_info_means = []
        target_threshold_means = []
        profit_all = 1.0
        profit_topn = 1.0
        profit_all1 = 1.0
        profit_topn1 = 1.0
        profit_info = 1.0
        profit_threshold = 1.0

        sortedDate = list(model_result_dates.keys())
        sortedDate.sort()
        profit_mean_high = 1.0
        profit_mean_low = 1.0
        profit_mean_drawdown = 0.0
        profit_topn_high = 1.0
        profit_topn_low = 1.0
        profit_topn_drawdown = 0.0
        profit_threshold_high = 1.0
        profit_threshold_low = 1.0
        profit_threshold_drawdown = 0.0

        for date in sortedDate:
            # date = date.item()
            if args["type"] == "MultiLabelLSTM":
                result = calculate_metrics(np.array(model_result)[:-args["multi_steps"]], np.array(targets)[:-args["multi_steps"]])
                for key in result:
                    print(key)
                    print(result[key])
            else:
                topnidx = np.array(model_result_dates[date]).argsort(axis=0)[-NP_TOPN:, :]
                topnclose = np.take(targets_dates[date], topnidx)
                topnpred = np.take(model_result_dates[date], topnidx)
                topninfo = np.take(infos[date], topnidx)

                target_topn_means += [topnclose.mean()]
                target_means += [np.mean(targets_dates[date])]
                target_info_means += [np.power(topnclose, topninfo).mean()]
                target_threshold_means += [topnclose[topnpred > buy_threshold].mean() if topnclose[topnpred > buy_threshold].size>0 else 1.0]

                date_adj = -1 if LSTM_ADJUST_END == "O" else 0

                if idx%(args["multi_steps"]+date_adj) == 0:
                    profit_all *= target_means[-1]
                    profit_mean_high, profit_mean_low, profit_mean_drawdown = update_drawback(profit_mean_high, profit_mean_low, profit_all, profit_mean_drawdown)
                    profit_info *= target_info_means[-1]

                    # if topnpred.mean() > 1.02 and True:  #fixme tobe continue
                    #     profit_topn *= target_topn_means[-1]
                    profit_topn *= target_topn_means[-1]
                    profit_topn_high, profit_topn_low, profit_topn_drawdown = update_drawback(profit_topn_high, profit_topn_low, profit_topn, profit_topn_drawdown)

                    profit_threshold *= target_threshold_means[-1]
                    profit_threshold_high, profit_threshold_low, profit_threshold_drawdown = update_drawback(profit_threshold_high,
                                                                                                             profit_threshold_low, profit_threshold, profit_threshold_drawdown)
                    # if target_threshold_means[-1] == target_threshold_means[-1]:
                    #     profit_threshold *= target_threshold_means[-1]
                else:
                    profit_all1 *= target_means[-1]
                    profit_topn1 *= target_topn_means[-1]

                print(f"\nAverage {NP_TOPN} close is {pandas.to_datetime(date)}:", target_means[-1], target_topn_means[-1], profit_all, profit_topn, profit_info, profit_threshold,
                      # profit_all1, profit_topn1,
                      np.mean(target_means), np.mean(target_topn_means), np.mean(target_info_means),
                      np.mean(np.array(target_threshold_means)[~np.isnan(target_threshold_means)]), profit_mean_drawdown, profit_topn_drawdown, profit_threshold_drawdown)
                topncode = np.take(code_dates[date], topnidx)
                # topnpred = np.take(model_result_dates[date], topnidx)
                topnAll = np.concatenate((topncode, topnclose, topnpred), axis=1)
                # print(f"\ntopn close is:--------{pandas.to_datetime(date)} code close pred--------\r\n{topnAll}\r\n{np.mean(topnclose)}\r\n{topnclose}" )
                idx += 1

        print(datetime.datetime.fromtimestamp(sortedDate[0]/1e9), "--->", datetime.datetime.fromtimestamp(sortedDate[-1]/1e9))

    RPE_FILE_R = "/home/yacc/shares/rpe_pre.xlsx.colored.xlsx"
    XSY_FILE_R = "/home/yacc/shares/xsy_pre.xlsx.colored.xlsx"
    STING_FILE_R = "/home/yacc/shares/sting_pre.xlsx.colored.xlsx"
    BDT_FILE_R = "/home/yacc/shares/bdt_pre.xlsx.colored.xlsx"
    KJYL_FILE_R = "/home/yacc/shares/x_kjyl_pre_111.xlsx.colored.xlsx.xlsx" #"/home/yacc/shares/kjyl_pre.xlsx.colored.xlsx"

    fileDict = {"xsy": XSY_FILE_R, "rpe": RPE_FILE_R, "sting": STING_FILE_R,
            "bdt": BDT_FILE_R, "kjyl": KJYL_FILE_R}
    fileDict = {"kjyl": KJYL_FILE_R}
    filedfList = {}

    for file in fileDict:
        filedf = pandas.read_excel(fileDict[file], dtype={"股票代码":str}).set_index("股票代码")
        filedf["idx"] = 0
        for idx in range(len(filedf.index)):
            filedf.loc[filedf.index[idx], "idx"] = idx
        filedf["type"] = file
        filedfList[file] = filedf

    codes = []
    targets = []
    model_result = []
    stocktypes = []
    for (seq, mlp_in, target, code, date) in last_seq_ts:
        # y=np.append(y,target.numpy(),axis=0)
        # seq = seq.to(device)
        with torch.no_grad():
            inputs = [[lstmseq.to(device) for lstmseq in seqs]]
            if input_size[EXP_MODS_MLP_IDX] > 0:
                inputs += [mlp_in]
            y_pred = model(inputs)
            # y_pred = model(seq)
            # pred=np.append(pred,y_pred.cpu().numpy(),axis=0)
            model_result.extend( y_pred.detach().cpu().numpy() )
            targets.extend( target.detach().cpu().numpy() )
            codes.extend(code)
            bfound = False
            for stocktype in filedfList.keys():
                if code[0] in filedfList[stocktype].index:
                    stocktypes += [stocktype]
                    bfound = True
                    break
            if not bfound:
                stocktypes += ["no"]

    # topnidx = np.array(model_result).argsort(axis=0)[-NP_TOPN:, :]
    # topnclose = np.take(codes, topnidx)
    # print("topn codes")
    # print(topnclose)
    topnidx = np.array(model_result).argsort(axis=0)[-NP_TOPN:, :]
    topnclose = np.take(codes, topnidx)
    print("predicting code", pandas.to_datetime(date.item()))
    for c in topnclose:
        print(c)
    # print(f"\nAverage {NP_TOPN} close is:", topnclose.mean())
    topnpred = np.take(model_result, topnidx)
    topnclose = np.concatenate((topnclose, topnpred), axis=1)
    print("\ntopn close is:\r\n", topnclose)

    typeLists = [["xsy", "rpe", "sting", "kjyl", "bdt"], ["xsy"], ["rpe"], ["sting"], ["kjyl"], ["bdt"]]
    typeLists = [["kjyl"]]


    for typeList in typeLists:
        stockMask = (np.array(stocktypes)==typeList[0])
        for typec in typeList[1:]:
            stockMask = stockMask | ((np.array(stocktypes)==typec))
        maskedCode = np.array(codes)[stockMask]
        maskedPred = np.array(model_result)[stockMask]
        maskedAll = np.concatenate((maskedCode.reshape([-1, 1]), maskedPred), axis=1)
        sortedAll = maskedAll[maskedAll[:, 1].argsort()]
        print("\ntopn close code is:\r\n", typeList, sortedAll[:, 0])
        print("\ntopn close is:\r\n", typeList, sortedAll)

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

    pred_stat(currbest_pred_target, stage="test")

def test_signal(_signo, _stack_frame):
    global TEST_FLAG
    print("prepare to test.")
    TEST_FLAG = True

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
          "input_size":[[BK_SIZE*len(COLS), len(FS_COLS)], 2], #number of input parameters used in predition. you can modify it in data index list.
          "hidden_size":[[int(BK_SIZE*len(COLS)*2), 2*len(FS_COLS)], 4],#number of cells in one hidden layer.
          "num_layers":[[2, 2], 2],  #number of hidden layers in predition module.
          "merged_hidden_size": 2*(len(COLS)+len(FS_COLS)),
          "merged_num_mid_layers": 2,
          "output_size":BK_SIZE, #number of parameter will be predicted.
          "lr":1e-3, #1e-5,
          "weight_decay":0.0, #0001,
          "bidirectional":False,
          "type": "LSTMs", # BiLSTM, LSTM, MultiLabelLSTM, MLPX
          "optimizer":"adam",
          "step_size":500000000000,
          "gamma":0.5,
          "epochs":50000,
          "batch_size":64,#batch of data will be push in model. large batch in multi parameter prediction will be better.
          "seq_len":[EXP_DK1D_SEQ_LEN, EXP_FS1Y_SEQ_LEN], #one contionus series input data will be used to predict the selected parameter.
          "multi_steps":2, #next x days's stock price can be predictions. maybe 1,2,3....
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

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    torch.set_num_threads(os.cpu_count()-1)
    reproducibility_init()

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
    path_file= [
        './model/'+'module-' + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}.ping' + '.pkl', #Module for the next "x" day's stock price prediction.
        './model/'+'module-' + args['type'] + '-bk' + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}.pong' + '.pkl', #Module for the next "x" day's stock price prediction.
    ]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal.signal(signal.SIGUSR1, test_signal)
    print("CUDA or CPU:", device)
    #load data to a dataFrame.
    # Dtr, Val, Dte, last_seq_ts, testdf = nn_stocksdata_seq_split_by_code_n_date(args['batch_size'], args['type'])
    Dtr, Val, Dte, last_seq_ts, testdf = nn_stocksdata_seq(args['batch_size'], args['type'])

    #train it.
    if NO_TRAIN == False:
        buy_threshold = train(args, Dtr, Val, path_file, Dte, last_seq_ts)
    else:
        buy_threshold = math.nan
    #teest it.
    test(args, Dte, path_file, data_pred_index, last_seq_ts, testdf, buy_threshold)
    