import torch
from torch import device, nn
import torch.utils.data as Data
from torch.optim import lr_scheduler
import math
import numpy as np
import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

NO_TRAIN = False
NO_TEST = False

MODULE_LOAD_BEST = NO_TRAIN
MODULE_CURRENT_SAVED = None
MODULE_SAVED_PING = 0
MODULE_SAVED_PONG = 1

LSTM_SEQ_LEN = 7
LSTM_TRAIN_LEN = 3000
LSTM_VAL_LEN = 1000
LSTM_DTE_LEN = 1000

TCH_EARLYSTOP_PATIENCE = 0

class MyDataset(Data.Dataset):
    def __init__(self, data):
        self.data = data
        self.labels = [ x[1].item() for x in data ]

    def __getitem__(self, item):
        #print(len(self.data), item, "\r\n", self.data[item][0].numpy()[::34], self.data[item][1].numpy())
        return self.data[item]

    def __len__(self):
        return len(self.data)

    # def get_labels(self):
    #     return self.labels

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

def do_calc(x):
    # return math.sin(x*0.001*math.pi*math.pi)
    return math.cos(x*0.002*math.pi*math.pi)*math.sin(x*0.001*math.pi*math.pi)

def nn_stocksdata_seq():
    args = get_args()
    print('data processing...')
    seqs = []
    start = 0
    for i in range(start, start+LSTM_TRAIN_LEN):
        inputs = [ [do_calc(i+j)] for j in range(args["seq_len"])]
        target = [ [do_calc(i+args["seq_len"]+args["multi_steps"])]]
        inputs_ts = torch.FloatTensor(np.array(inputs)) #.view(-1)
        target_ts = torch.FloatTensor(np.array(target)).view(-1)
        seqs += [(inputs_ts, target_ts)]
    seq = MyDataset(seqs)
    Dtr = DataLoader(dataset=seq, batch_size=64, num_workers=0, drop_last=True)

    seqs = []
    start = i
    for i in range(start, start+LSTM_VAL_LEN):
        inputs = [ [do_calc(i+j)] for j in range(args["seq_len"])]
        target = [ [do_calc(i+args["seq_len"]+args["multi_steps"])]]
        inputs_ts = torch.FloatTensor(np.array(inputs)) #.view(-1)
        target_ts = torch.FloatTensor(np.array(target)).view(-1)
        seqs += [(inputs_ts, target_ts)]
    seq = MyDataset(seqs)
    Val = DataLoader(dataset=seq, batch_size=64, num_workers=0, drop_last=True)

    seqs = []
    start = i
    for i in range(start, start+LSTM_DTE_LEN):
        inputs = [ [do_calc(i+j)] for j in range(args["seq_len"])]
        target = [ [do_calc(i+args["seq_len"]+args["multi_steps"])]]
        inputs_ts = torch.FloatTensor(np.array(inputs)) #.view(-1)
        target_ts = torch.FloatTensor(np.array(target)).view(-1)
        seqs += [(inputs_ts, target_ts)]
    seq = MyDataset(seqs)
    Dte = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    return Dtr, Val, Dte #, last_seq_ts, pred

def get_module_saved_path(paths, load_best=True):
    ping_mtime = os.path.getmtime(paths[MODULE_SAVED_PING]) if os.path.exists(paths[MODULE_SAVED_PING]) else 0
    pong_mtime = os.path.getmtime(paths[MODULE_SAVED_PONG]) if os.path.exists(paths[MODULE_SAVED_PONG]) else 0

    MODULE_CURRENT_SAVED = ((ping_mtime > pong_mtime)^load_best)
    path = paths[MODULE_CURRENT_SAVED]
    print("load_best:", load_best, "ping is newest:", (ping_mtime > pong_mtime), "load:", path)
    return path, MODULE_CURRENT_SAVED

def get_args():
    return args;

def train(args, Dtr, Val, paths):
    input_size, hidden_size, num_layers = args['input_size'], args['hidden_size'], args['num_layers']
    output_size = args['output_size']
    seq_len = args['seq_len']

    if args["type"] == 'LSTM':
        model = LSTM(  input_size, hidden_size, num_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)

    if args['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'],momentum=0.9, weight_decay=args['weight_decay'])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    path, cur_sav_idx = get_module_saved_path(paths, load_best=MODULE_LOAD_BEST)
    if os.path.exists(path):
        print('loading models...no!')
        # model.load_state_dict(torch.load(path)['models'])

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

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f} best_loss {:.8f} patience {:04d}'.format(epoch, train_loss_all[-1], val_loss_all[-1], best_loss, patience), flush=True)

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

def test(args, Dte, paths):
    args=get_args()
    input_size, hidden_size, num_layers = args['input_size'], args['hidden_size'], args['num_layers']
    output_size = args['output_size']
    seq_len = args['seq_len']

    if args['type'] == "LSTM":
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=1).to(device)

    # path, _ = get_module_saved_path(paths)

    path, cur_sav_idx = get_module_saved_path(paths, load_best=MODULE_LOAD_BEST)
    if os.path.exists(path):
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

    if not NO_TEST:
        targets = []
        model_result = []
        # currbest_pred_target = []
        for (seq, target) in tqdm(Dte):
            y=np.append(y,target.numpy(),axis=0)
            seq = seq.to(device)
            with torch.no_grad():
                y_pred = model(seq)
                # currbest_pred_target += [(y_pred.detach().cpu().numpy().flatten(), target.detach().cpu().numpy().flatten())]
                model_result.extend( y_pred.detach().cpu().numpy() )
                targets.extend( target.detach().cpu().numpy() )

            # if len(model_result) == 100 and False:
            #     plt.plot(np.array(model_result).T, np.array(targets).T)
            #     plt.show()

    if True:
        plt.plot(model_result, linestyle = 'dotted', color='green')
        # x = []
        # for i in range(0,len(targets)):
        #     x.append(i)
        # plt.scatter(x, targets, linestyle = 'dotted')
        plt.plot(targets, linestyle = 'dotted', color='blue')
        plt.grid(axis='y')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    args={
          "input_size":1, #number of input parameters used in predition. you can modify it in data index list.
          "hidden_size":10,#number of cells in one hidden layer.
          "num_layers":3,  #number of hidden layers in predition module.
          "output_size":1, #number of parameter will be predicted.
          "lr":1e-3, #1e-5,
          "weight_decay":0.0, #0001,
          "bidirectional":False,
          "type": "LSTM", # BiLSTM, LSTM, MultiLabelLSTM, MLPX
          "optimizer":"adam",
          "step_size":500000000000,
          "gamma":0.5,
          "epochs":50000,
          "batch_size":64,#batch of data will be push in model. large batch in multi parameter prediction will be better.
          "seq_len":LSTM_SEQ_LEN, #one contionus series input data will be used to predict the selected parameter.
          "multi_steps":6, #next x days's stock price can be predictions. maybe 1,2,3....
          "pred_type":"close",#open price / close price / high price / low price.
          "train_end":1.0,
          "val_begin":0.6,
          "val_end":0.8,
          "test_begin":0.8
        }
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_file= [
        './model/'+'module-' + args['type'] + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}.ping' + '.pkl', #Module for the next "x" day's stock price prediction.
        './model/'+'module-' + args['type'] + f'-{args["input_size"]}-{args["num_layers"]}X{args["hidden_size"]}-{args["output_size"]}.pong' + '.pkl', #Module for the next "x" day's stock price prediction.
    ]

    Dtr, Val, Dte = nn_stocksdata_seq()
    if NO_TRAIN == False:
        train(args, Dtr, Val, path_file)
    test(args, Dte, path_file)
