import torch
from torch import nn
import torch.utils.data as Data
from torch.optim import lr_scheduler
import math
import numpy as np
import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.regression import R2Score
from nn_macro import *
from torchviz import make_dot

NO_TRAIN = False
NO_TEST = False

MODULE_LOAD_BEST = NO_TRAIN
MODULE_CURRENT_SAVED = None
MODULE_SAVED_PING = 0
MODULE_SAVED_PONG = 1

LSTM_SEQ_LEN = 30
LSTM_TRAIN_LEN = 10000
LSTM_VAL_LEN = 3000
LSTM_DTE_LEN = 3000
LSTM_STEPS = 100
LSTM_SEC_MODEL_SEQ_LEN = 25
LSTM_THD_MODEL_MLP_STEP = 10

TCH_EARLYSTOP_PATIENCE = 20

class MyDataset(Data.Dataset):
    def __init__(self, data):
        self.data = data
        self.labels = [ x[2].item() for x in data ]

    def __getitem__(self, item):
        #print(len(self.data), item, "\r\n", self.data[item][0].numpy()[::34], self.data[item][1].numpy())
        return self.data[item]

    def __len__(self):
        return len(self.data)

    # def get_labels(self):
    #     return self.labels
'''
ELU have been shown to produce more accurate results than ReLU and also converge faster
calculate_gain:
identity  : 1.0
sigmoid   : 1.8471702337265015
tanh      : 1.5913989543914795
relu      : 1.4173601865768433
leaky_relu: 1.3895978927612305
selu      : 1.000707983970642
elu       : 1.2470202445983887
'''

EXP_MODS_LSTM_IDX = 0
EXP_MODS_MLP_IDX = 1
EXP_MODS_LAYER_SUB = 0
EXP_MODS_LAYER_SUP = 1
class LSTMs(nn.Module):
    def __init__(self, input_mask, input_sizes, hidden_sizes, dropouts, num_layers, output_layers, merged_hidden_size, merged_num_mid_layers, output_size, batch_size):
        super().__init__()
        self.input_mask = input_mask
        self.input_sizes = input_sizes
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.output_layers = output_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.h0s = None
        self.c0s = None
        self.lstms = None
        self.imlp = None
        # self.lstmLinearSeqs = nn.ModuleList()

        lstmList = []
        lstmLinearSeqs = []
        h0s = []
        c0s = []
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(len(self.input_sizes[EXP_MODS_LSTM_IDX])):
            # if self.output_layers[EXP_MODS_LSTM_IDX][i] == 0:
            #     continue
            if self.input_mask[EXP_MODS_LSTM_IDX][i] == False:
                continue
            h0 = torch.zeros(self.num_directions * self.num_layers[EXP_MODS_LSTM_IDX][i], 1, self.hidden_sizes[EXP_MODS_LSTM_IDX][i]).to(device)
            c0 = torch.zeros(self.num_directions * self.num_layers[EXP_MODS_LSTM_IDX][i], 1, self.hidden_sizes[EXP_MODS_LSTM_IDX][i]).to(device)
            # nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(h0, gain=1.2470202445983887)
            nn.init.xavier_normal_(c0, gain=1.2470202445983887)
            h0s += [nn.Parameter(h0, requires_grad=True)]  # Parameter() to update weights
            c0s += [nn.Parameter(c0, requires_grad=True)]

            lstmList += [nn.LSTM(input_size = self.input_sizes[EXP_MODS_LSTM_IDX][i],
                                 hidden_size = self.hidden_sizes[EXP_MODS_LSTM_IDX][i],
                                 num_layers = self.num_layers[EXP_MODS_LSTM_IDX][i], batch_first=True, dropout=dropouts[EXP_MODS_LAYER_SUB][EXP_MODS_LSTM_IDX][i]),
            ]
            # lstmLinearSeqs += [nn.Sequential(
            #                     nn.Linear(self.hidden_sizes[EXP_MODS_LSTM_IDX][i], self.output_layers[EXP_MODS_LSTM_IDX][i]),
            #                     # nn.ReLU()
            #                 )
            # ]

        self.lstms = nn.ModuleList(lstmList)
        # self.lstmLinearSeqs = nn.ModuleList(lstmLinearSeqs)
        self.h0s = nn.ParameterList(h0s)
        self.c0s = nn.ParameterList(c0s)

        if input_sizes[EXP_MODS_MLP_IDX] > 0 and self.input_mask[EXP_MODS_MLP_IDX]==True: #and self.output_layers[EXP_MODS_MLP_IDX]>0:
            self.imlp = nn.Sequential()
            self.imlp.add_module("iDI", nn.Dropout(dropouts[EXP_MODS_LAYER_SUB][EXP_MODS_MLP_IDX]))
            self.imlp.add_module("iLI", nn.Linear(self.input_sizes[EXP_MODS_MLP_IDX], self.hidden_sizes[EXP_MODS_MLP_IDX]))
            self.imlp.add_module("iRI", nn.ELU())
            for i in range(merged_num_mid_layers):
                self.imlp.add_module(f"iD{i}", nn.Dropout(EXP_NN_DROPOUT))
                self.imlp.add_module(f"iL{i}", nn.Linear(self.hidden_sizes[EXP_MODS_MLP_IDX], self.hidden_sizes[EXP_MODS_MLP_IDX]))
                self.imlp.add_module(f"iR{i}", nn.ELU())
                # self.imlp.add_module(f"iR{i}", nn.ReLU())
            # self.imlp.add_module(f"iLO", nn.Linear(self.hidden_sizes[EXP_MODS_MLP_IDX], self.output_layers[EXP_MODS_MLP_IDX]))
            # self.imlp.add_module(f"RO", nn.ReLU())

        self.mlp = nn.Sequential()
        # self.mlp.add_module("DI", nn.Dropout(EXP_NN_DROPOUT))
        self.mlp.add_module(f"DI", nn.Dropout(dropouts[EXP_MODS_LAYER_SUP]))
        # self.mlp.add_module("LI", nn.Linear(sum(self.output_layers[EXP_MODS_LSTM_IDX])+self.output_layers[EXP_MODS_MLP_IDX], merged_hidden_size))
        self.mlp.add_module("LI", nn.Linear(sum([ self.hidden_sizes[EXP_MODS_LSTM_IDX][i] if self.input_mask[EXP_MODS_LSTM_IDX][i] else 0
                                                  for i in range(len(self.hidden_sizes[EXP_MODS_LSTM_IDX]))])+
                                            (self.hidden_sizes[EXP_MODS_MLP_IDX] if self.input_mask[EXP_MODS_MLP_IDX] else 0), merged_hidden_size))
        self.mlp.add_module("RI", nn.ELU())
        for i in range(merged_num_mid_layers):
            self.mlp.add_module(f"D{i}", nn.Dropout(dropouts[EXP_MODS_LAYER_SUP]))
            self.mlp.add_module(f"L{i}", nn.Linear(merged_hidden_size, merged_hidden_size))
            self.mlp.add_module(f"R{i}", nn.ELU())
        # self.mlp.add_module(f"RO", nn.ReLU())
        self.mlp.add_module(f"LO", nn.Linear(merged_hidden_size, output_size))

    def forward(self, lstm_input_seqs): #, mlp_input_seqs=[]
        batch_size, seq_len = lstm_input_seqs[EXP_MODS_LSTM_IDX][0].shape[0], lstm_input_seqs[EXP_MODS_LSTM_IDX][0].shape[1]
        # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # h_0 = torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).requires_grad_()
        # c_0 = torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).requires_grad_()
        # output(batch_size, seq_len, num_directions * hidden_size)
        outputs = []
        mlpoutput = []
        j = 0
        for i in range(len(self.input_sizes[EXP_MODS_LSTM_IDX])):
            # if self.output_layers[EXP_MODS_LSTM_IDX][i] == 0:
            #     continue
            if self.input_mask[EXP_MODS_LSTM_IDX][i] == False:
                continue
            output, _ = self.lstms[j](lstm_input_seqs[EXP_MODS_LSTM_IDX][i], (self.h0s[j].repeat(1, batch_size, 1), self.c0s[j].repeat(1, batch_size, 1))) # output(5, 30, 64)
            # output = self.linears[i](output)
            # output = self.lstmLinearSeqs[j](output)
            outputs += [output]
            j += 1

        if self.input_sizes[EXP_MODS_MLP_IDX] > 0 and self.input_mask[EXP_MODS_MLP_IDX]==True: #and self.output_layers[EXP_MODS_MLP_IDX]>0:
            mlpoutput += [self.imlp(lstm_input_seqs[EXP_MODS_MLP_IDX])]
            # mlpoutput += [self.imlp(lstm_input_seqs[EXP_MODS_MLP_IDX][0])]

        x = torch.cat([output[:, -1, :] for output in outputs] + [output for output in mlpoutput], dim=1)
        pred = self.mlp(x)  # (5, 24, 1)
        # pred = pred[:, -1, :]  # (5, 1)
        return pred

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

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first=True, dropout=EXP_NN_DROPOUT)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        input_seq = input_seq[EXP_MODS_LSTM_IDX][0]
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
    return math.cos(4*x/(LSTM_TRAIN_LEN*1.0)*math.pi)*math.sin(x*0.001*math.pi*math.pi)

def nn_stocksdata_seq():
    args = get_args()
    print('data processing...')
    plot_data = []
    xs = []
    seqs = []
    start = 0
    for i in range(start, start+LSTM_TRAIN_LEN):
        inputs = [ [do_calc(i+j)] for j in range(args["seq_len"])]
        inputs2 = [ [do_calc(i+args["seq_len"]+j)] for j in range(LSTM_SEC_MODEL_SEQ_LEN)]
        # inputs3 = [ [do_calc(i+args["seq_len"]+LSTM_SEC_MODEL_SEQ_LEN+j) for j in range(LSTM_THD_MODEL_MLP_STEP, LSTM_THD_MODEL_MLP_STEP+1)]]
        inputs3 = [ [i%(LSTM_TRAIN_LEN*0.5)/(LSTM_TRAIN_LEN*0.5)] ]
        target = [ [do_calc(i+args["seq_len"]+args["multi_steps"])]]
        plot_data += target
        xs += [i]
        inputs_ts = torch.FloatTensor(np.array(inputs)) #.view(-1)
        inputs_ts2 = torch.FloatTensor(np.array(inputs2)) #.view(-1)
        inputs_ts3 = torch.FloatTensor(np.array(inputs3)).view(-1)
        target_ts = torch.FloatTensor(np.array(target)).view(-1)
        seqs += [((inputs_ts, inputs_ts2), (inputs_ts3, ), target_ts, i)]
    plotResult(plot_data, plot_data, xs)
    seq = MyDataset(seqs)
    Dtr = DataLoader(dataset=seq, batch_size=64, shuffle=True , num_workers=0, drop_last=True)

    plot_data = []
    xs = []
    seqs = []
    start = i
    for i in range(start, start+LSTM_VAL_LEN):
        inputs = [ [do_calc(i+j)] for j in range(args["seq_len"])]
        inputs2 = [ [do_calc(i+args["seq_len"]+j)] for j in range(LSTM_SEC_MODEL_SEQ_LEN)]
        # inputs3 = [ [do_calc(i+args["seq_len"]+LSTM_SEC_MODEL_SEQ_LEN+j) for j in range(LSTM_THD_MODEL_MLP_STEP, LSTM_THD_MODEL_MLP_STEP+1)]]
        inputs3 = [ [i%(LSTM_TRAIN_LEN*0.5)/(LSTM_TRAIN_LEN*0.5)] ]
        target = [ [do_calc(i+args["seq_len"]+args["multi_steps"])]]
        plot_data += target
        xs += [i]
        inputs_ts = torch.FloatTensor(np.array(inputs)) #.view(-1)
        inputs_ts2 = torch.FloatTensor(np.array(inputs2)) #.view(-1)
        inputs_ts3 = torch.FloatTensor(np.array(inputs3)).view(-1)
        target_ts = torch.FloatTensor(np.array(target)).view(-1)
        seqs += [((inputs_ts, inputs_ts2), (inputs_ts3, ), target_ts, i)]
    plotResult(plot_data, plot_data, xs)
    seq = MyDataset(seqs)
    Val = DataLoader(dataset=seq, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    plot_data = []
    xs = []
    seqs = []
    start = i
    for i in range(start, start+LSTM_DTE_LEN):
        inputs = [ [do_calc(i+j)] for j in range(args["seq_len"])]
        inputs2 = [ [do_calc(i+args["seq_len"]+j)] for j in range(LSTM_SEC_MODEL_SEQ_LEN)]
        # inputs3 = [ [do_calc(i+args["seq_len"]+LSTM_SEC_MODEL_SEQ_LEN+j) for j in range(LSTM_THD_MODEL_MLP_STEP, LSTM_THD_MODEL_MLP_STEP+1)]]
        inputs3 = [ [i%(LSTM_TRAIN_LEN*0.5)/(LSTM_TRAIN_LEN*0.5)] ]
        target = [ [do_calc(i+args["seq_len"]+args["multi_steps"])]]
        plot_data += target
        xs += [i]
        inputs_ts = torch.FloatTensor(np.array(inputs)) #.view(-1)
        inputs_ts2 = torch.FloatTensor(np.array(inputs2)) #.view(-1)
        inputs_ts3 = torch.FloatTensor(np.array(inputs3)).view(-1)
        target_ts = torch.FloatTensor(np.array(target)).view(-1)
        seqs += [((inputs_ts, inputs_ts2), (inputs_ts3, ), target_ts, i)]
    plotResult(plot_data, plot_data, xs)
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
    output_layers = args['output_layers']
    merged_hidden_size, merged_num_mid_layers = args['merged_hidden_size'], args['merged_num_mid_layers']
    output_size = args['output_size']
    seq_len = args['seq_len']

    if args["type"] == 'LSTM':
        model = LSTM(  input_size[EXP_MODS_LSTM_IDX][0], hidden_size[EXP_MODS_LSTM_IDX][0], num_layers[EXP_MODS_LSTM_IDX][0], output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)

    if args["type"] == 'LSTMs':
        model = LSTMs(  args['input_mask'], input_size, hidden_size, args['dropout'], num_layers, output_layers, merged_hidden_size, merged_num_mid_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)

    if args['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'],momentum=0.9, weight_decay=args['weight_decay'])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    path, cur_sav_idx = get_module_saved_path(paths, load_best=MODULE_LOAD_BEST)
    if os.path.exists(path):
        print('loading models...yes!')
        model.load_state_dict(torch.load(path)['models'])

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
        xs = []
        model.eval()
        for seqs in Val:
            label, x = seqs[-2], seqs[-1]
            label = label.to(device)
            # y_pred = model([[lstmseq.to(device) for lstmseq in seqs[EXP_MODS_LSTM_IDX]], [mlpseq.to(device) for mlpseq in seqs[EXP_MODS_MLP_IDX]]])
            y_pred = model([[lstmseq.to(device) for lstmseq in seqs[EXP_MODS_LSTM_IDX]], *[mlpseq.to(device) for mlpseq in seqs[EXP_MODS_MLP_IDX]]])
            # make_dot(y_pred.mean(), params=dict(model.named_parameters())).render("lstm_model_viz", format="png")
            # make_dot(y_pred.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("lstm_param_viz", format="png")
            currbest_pred_target += [(y_pred.detach().cpu().numpy().flatten(), label.detach().cpu().numpy().flatten())]
            model_result.extend( [*(y_pred.detach().cpu().numpy().flatten())] )
            targets.extend( [*(label.detach().cpu().numpy().flatten())] )
            xs += [*(x.detach().cpu().numpy())]
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

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f} best_loss {:.8f} R2 {:.4f} patience {:04d}'.format(epoch, train_loss_all[-1], val_loss_all[-1], best_loss, r2score(torch.tensor(model_result), torch.tensor(targets)), patience), flush=True)

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
            if len(val_loss_all)%1 == 0:
                plotResult(model_result, targets, xs)
        elif val_loss_all[-1]>best_loss:
            patience += 1
            if patience > TCH_EARLYSTOP_PATIENCE:
                break

        train_loss = 0
        num_item=0
        targets = []
        model_result = []
        xs = []
        model.train()
        for seqs in Dtr:
            label, x = seqs[-2], seqs[-1]
            label = label.to(device)
            y_pred = model([[lstmseq.to(device) for lstmseq in seqs[EXP_MODS_LSTM_IDX]], [mlpseq.to(device) for mlpseq in seqs[EXP_MODS_MLP_IDX]]])
            model_result.extend( [*(y_pred.detach().cpu().numpy().flatten())] )
            targets.extend( [*(label.detach().cpu().numpy().flatten())] )
            xs += [*(x.detach().cpu().numpy())]
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
        if epoch%4 == 0:
            plotResult(model_result, targets, xs)

def test(args, Dte, paths):
    args=get_args()
    input_size, hidden_size, num_layers = args['input_size'], args['hidden_size'], args['num_layers']
    output_layers = args['output_layers']
    merged_hidden_size, merged_num_mid_layers = args['merged_hidden_size'], args['merged_num_mid_layers']
    output_size = args['output_size']
    seq_len = args['seq_len']

    if args['type'] == "LSTM":
        model = LSTM(input_size[EXP_MODS_LSTM_IDX][0], hidden_size[EXP_MODS_LSTM_IDX][0], num_layers[EXP_MODS_LSTM_IDX][0], output_size, batch_size=1).to(device)

    if args["type"] == 'LSTMs':
        model = LSTMs(  args['input_mask'], input_size, hidden_size, args['dropout'], num_layers, output_layers, merged_hidden_size, merged_num_mid_layers, output_size, batch_size=args['batch_size']).to(device)
        loss_function = nn.MSELoss().to(device)

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

    r2score = R2Score()
    if not NO_TEST:
        targets = []
        model_result = []
        xs = []
        # currbest_pred_target = []
        for seqs in tqdm(Dte):
            # y=np.append(y,target.numpy(),axis=0)
            target = seqs[-2]
            x = seqs[-1]
            label = target.to(device)
            with torch.no_grad():
                y_pred = model([[lstmseq.to(device) for lstmseq in seqs[EXP_MODS_LSTM_IDX]], [mlpseq.to(device) for mlpseq in seqs[EXP_MODS_MLP_IDX]]])
                # currbest_pred_target += [(y_pred.detach().cpu().numpy().flatten(), target.detach().cpu().numpy().flatten())]
                targets.extend( [*(label.detach().cpu().numpy().flatten())] )
                model_result.extend( [*(y_pred.detach().cpu().numpy().flatten())] )
                xs += [*(x.detach().cpu().numpy())]

            # if len(model_result) == 100 and False:
            #     plt.plot(np.array(model_result).T, np.array(targets).T)
            #     plt.show()
        print("r2socre", r2score(torch.tensor(model_result), torch.tensor(targets)))

    if True:
        plotResult(model_result, targets, xs)
        # plt.plot(model_result, linestyle = 'dotted', color='green')
        # # x = []
        # # for i in range(0,len(targets)):
        # #     x.append(i)
        # # plt.scatter(x, targets, linestyle = 'dotted')
        # plt.plot(targets, linestyle = 'dotted', color='blue')
        # plt.grid(axis='y')
        # plt.legend()
        # plt.show()

def plotResult(pred, targets, x):
    plt.scatter(x, pred, marker=".", s=1, linewidth=0)
    # plt.plot(x, pred, linestyle = 'dotted', color='green')
    # x = []
    # for i in range(0,len(targets)):
    #     x.append(i)
    plt.scatter(x, targets, marker=".", s=1, linewidth=0 )
    # plt.plot(x, targets, linestyle = 'dotted', color='blue')
    plt.grid(axis='y')
    # plt.legend()
    plt.show()

if __name__ == '__main__':
    args={
          "input_mask":[[True, True], True], #number of input parameters used in predition. you can modify it in data index list.
          "input_size":[[1, 1], 1], #number of input parameters used in predition. you can modify it in data index list.
          "hidden_size":[[10, 9], 8],#number of cells in one hidden layer.
          "dropout":[[[0.3, 0.3], 0.0], 0], #[[2, 2], 2],  #number of hidden layers in predition module.
          "num_layers":[[4, 3], 2],  #number of hidden layers in predition module.
          "output_layers": [[1, 1], 1], #[[1, len(FS_COLS)], 2],
          "merged_hidden_size": 8,
          "merged_num_mid_layers": 2,
          "output_size":1, #number of parameter will be predicted.
          "lr":3e-4, #1e-5,
          "weight_decay":0.0, #0001,
          "bidirectional":False,
          "type": "LSTMs", # BiLSTM, LSTM, MultiLabelLSTM, MLPX
          "optimizer":"adam",
          "step_size":500000000000,
          "gamma":0.5,
          "epochs":50000,
          "batch_size":64,#batch of data will be push in model. large batch in multi parameter prediction will be better.
          "seq_len":LSTM_SEQ_LEN, #one contionus series input data will be used to predict the selected parameter.
          "multi_steps":LSTM_STEPS, #next x days's stock price can be predictions. maybe 1,2,3....
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
