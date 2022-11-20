# -*- coding: utf-8 -*-

import pandas as pd
import scipy.optimize
import numpy as np
import os
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.messagebox import showinfo, showerror, askyesno, showwarning
from tkinter.simpledialog import askinteger, askstring, askfloat
from tkinter import *
from tkinter import ttk
from itertools import combinations
import pickle
try:
    from sklearn.cluster import AffinityPropagationCustom
except:
    from sklearn.cluster import AffinityPropagation
from apyori import apriori
import matplotlib.pyplot as plt



def decompose(M, MainFig, maxit=2000, lr=0.003):
    N = M.shape[0]
    core = np.random.rand(N)
    scale = np.max(M)
    M = M / scale
    stable = 0
    up = 0
    lossList = list()
    # 进度条
    progressToplevel = Toplevel(MainFig)
    progressToplevel.geometry('+800+300')
    progressToplevel.title('Coreness')
    progressLabel = Label(progressToplevel, text='Solving the coreness...')
    progressLabel.grid(row=0, column=0)
    progress = ttk.Progressbar(progressToplevel, orient='horizontal',
                               length=1000, mode='determinate')
    progress.grid(row=1, column=0)
    progress['value'] = 0
    progress['maximum'] = maxit
    for iter in range(maxit):
        grad = np.zeros_like(core)
        for i in range(len(grad)):
            crt_value = core[i]
            temp1 = np.delete(core, i)
            temp2 = np.delete(M[i,:],i)
            crt_grad = np.dot(temp1, crt_value * temp1 - temp2)
            grad[i] = crt_grad
        core = core - lr * grad
        crtLoss = loss_cal(M, core)
        lossList.append(crtLoss)
        if len(lossList) > 1:
            crtLossChange = (lossList[-2] - lossList[-1]) / lossList[-2]
            if abs(crtLossChange) < 1e-10:
                stable += 1
            else:
                stable = 0
            if crtLossChange < 0:
                up += 1
            else:
                up = 0
        try:
            progressLabel['text'] = 'Solving the coreness using Gradient Descent, iteration: {}/{}. ' \
                                    'Current loss is {}, the loss has not be improved for {} iterations.' \
                .format(iter + 1, maxit, round(crtLoss,3), stable)
            progress['value'] = iter + 1
            progressToplevel.update()
            if np.isnan(crtLoss) or up >= 50:
                progressToplevel.destroy()
                showwarning('Warning',
                            'The algorithm diverge, you are suggested to lower the learning rate and try again. Please refer to the trend of cost.')
                break
            if stable >= 20:
                break
        except:
            showinfo('Message','The algorithm is interupted, current results are returned.')
            break
    progressToplevel.destroy()
    st_core = core / np.sqrt(np.dot(core, core))
    core = core * np.sqrt(scale)
    expectedMatrix = np.zeros((N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            expectedMatrix[i,j] = core[i] * core[j]
            expectedMatrix[j,i] = core[i] * core[j]
    return st_core, core, expectedMatrix, corr_coef_for_decompose(M,st_core), lossList

def loss_cal(M, core):
    N = M.shape[0]
    loss = 0
    for i in range(N-1):
        for j in range(i+1, N):
            loss = loss + (M[i,j] - core[i]*core[j])**2
    return loss

def corr_coef_for_decompose(M, core):
    obs = list()
    pred = list()
    N = M.shape[0]
    for i in range(N-1):
        for j in range(i+1,N):
            obs.append(M[i,j])
            pred.append(core[i]*core[j])
    obs = obs - np.mean(obs)
    pred = pred - np.mean(pred)
    temp1 = np.dot(obs, pred)
    temp2 = np.sqrt(np.dot(obs, obs)) * np.sqrt(np.dot(pred,pred))
    return temp1 / temp2

def corr_coef(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    temp1 = np.dot(x, y)
    temp2 = np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y))
    return temp1/temp2

def suggestedCore(coreness):
    core = coreness['Coreness']
    N = len(core)
    trace = list()
    for i in range(1, N-1):
        temp = np.zeros(N)
        temp[:i] = 1
        trace.append(corr_coef(core, temp))
    NCore = np.argmax(trace) + 1
    places = np.array(coreness['Place'])
    return NCore, places[:NCore]

def center_window(root, width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    root.geometry(size)

def edit_distance2(list1, list2):
    # 类编辑距离：替换=2
    if type(list1) is str:
        list1 = [ele for ele in list1]
    if type(list2) is str:
        list2 = [ele for ele in list2]
    l1 = len(list1)
    l2 = len(list2)
    v = np.zeros((l1+1, l2+1))
    v[0, :] = [i for i in range(l2+1)]
    v[:, 0] = [i for i in range(l1+1)]
    for i in range(l1):
        for j in range(l2):
            if list1[i] == list2[j]:
                v[i+1, j+1] = v[i,j]
            else:
                v[i+1,j+1] = np.min([v[i+1, j]+1, v[i, j+1]+1, v[i,j]+2])
    V = v[-1, -1]
    return int(V)

class AprioriSetting:
    def __init__(self, MainFig):
        self.minSupport = 0.01
        self.minConfidence = 0.5
        self.minLift = 1
        self.mode = 1 # (1:support, 2:confidence,3:lift)
        # self.maxLength = 5
        master = Toplevel(MainFig)
        master.resizable(0,0)
        master.geometry('+800+300')
        self.master = master
        master.title('Apriori Setting')

        self.instr1 = Label(master, text='Defind the mininum support P(AB)：', height=1, width=50) \
            .grid(row=0, column=1, columnspan=3)
        self.instr2 = Message(master, text='(Support is the joint probability that both A and B exist)', width=450) \
            .grid(row=1, column=1, columnspan=3)
        self.supportEntryVar = StringVar()
        self.supportEntry = Entry(master,textvariable=self.supportEntryVar, width=50)
        self.supportEntryVar.set(str(self.minSupport))
        self.supportEntry.grid(row=3, column=1, columnspan=3)

        self.blank1 = Label(master, text='').grid(row=4, column=1, columnspan=3)
        self.instr3 = Label(master, text='Defind the mininum confidence P(B|A)：', height=1, width=50) \
            .grid(row=5, column=1, columnspan=3)
        self.instr4 = Message(master, text='(Confidence is the conditional probability that B exists given A exists)',
                            width=450) \
            .grid(row=6, column=1, columnspan=3)
        self.confidenceEntryVar = StringVar()
        self.confidenceEntry = Entry(master, textvariable=self.confidenceEntryVar, width=50)
        self.confidenceEntryVar.set(str(self.minConfidence))
        self.confidenceEntry.grid(row=7, column=1, columnspan=3)

        self.blank2 = Label(master, text='').grid(row=8, column=1, columnspan=3)
        self.instr5 = Label(master, text='Defind the mininum lift：', height=1, width=50) \
            .grid(row=9, column=1, columnspan=3)
        self.instr6 = Message(master, text='(Lift measures the information gain by the rule "'"A→B"'" compared to pure "'"B"'")',
                            width=450) \
            .grid(row=10, column=1, columnspan=3)
        self.liftEntryVar = StringVar()
        self.liftEntry = Entry(master, textvariable=self.liftEntryVar, width=50)
        self.liftEntryVar.set(str(self.minLift))
        self.liftEntry.grid(row=11, column=1, columnspan=3)

        self.blank3 = Label(master, text='').grid(row=12, column=1, columnspan=3)
        self.instr10 = Label(master, text='The rules would be sorted in descend order by:', height=1, width=50) \
            .grid(row=13, column=1, columnspan=3)
        modes = [('Support',1), ('Confidence',2), ('Lift',3)]
        self.v = IntVar()
        self.v.set(self.mode)
        temp = 1
        for text, mode in modes:
            self.modeRadio = Radiobutton(master,text=text, variable=self.v, value=mode)
            self.modeRadio.grid(row=14, column=temp)
            temp += 1
        self.blank4 = Label(master, text='').grid(row=15, column=1, columnspan=3)
        self.okbutton = Button(master, text='OK', width=50, command=self.save)
        self.okbutton.grid(row=16, column=1, padx=5, pady=5, columnspan=3)
        self.master.wait_window()
    def save(self):
        try:
            self.minSupport = float(self.supportEntryVar.get())
        except:
            showerror('Error', 'Invalid mininum support. Use the default value.')
            self.minSupport = 0.01
        if self.minSupport  < 0 or self.minSupport > 1:
            showerror('Error', 'Invalid mininum support. Use the default value.')
            self.minSupport = 0.01
        try:
            self.minConfidence = float(self.confidenceEntryVar.get())
        except:
            showerror('Error', 'Invalid mininum confience. Use the default value.')
            self.minConfidence = 0.5
        if self.minConfidence < 0 or self.minConfidence > 1:
            showerror('Error', 'Invalid mininum confience. Use the default value.')
            self.minConfidence = 0.5
        try:
            self.minLift = float(self.liftEntryVar.get())
        except:
            showerror('Error','Invalid minimum lift. Use the default value.')
            self.minLift = 1
        if self.minLift < 0:
            showerror('Error', 'Invalid minimum lift. Use the default value.')
            self.minLift = 1
        try:
            self.mode = float(self.v.get())
        except:
            self.minLift = 1
        self.master.destroy()

def analyzeAprioriResults(results, minSupport=0.01, minConfidence=0.5, minLift=1):
    RulesDF = pd.DataFrame(None, columns=['Rule', 'Support','Confidence', 'Lift'])
    for item in results:
        if len(item[0]) == 1:
            continue
        support = item[1]
        if support < minSupport:
            continue
        for relation in item[2]:
            base = np.unique(list(relation[0]))
            to = list(relation[1])
            confidence = relation[2]
            lift = relation[3]
            if confidence < minConfidence or lift < minLift:
                continue
            rule = ', '.join(base) + ' → ' + to[0]
            RulesDF.loc[RulesDF.shape[0]+1] = {'Rule':rule, 'Support':support, 'Confidence':confidence, 'Lift':lift}
    return RulesDF


class save_csv_class:
    def __init__(self, Mainfig, SaveContent):
        self.SaveContent = SaveContent
        self.output = []
        master = Toplevel(Mainfig)
        master.geometry('+800+300')
        self.master = master
        master.title('Export CSV')
        ListStr = StringVar(value=SaveContent)
        self.instr = Label(master, text='Select the Content to Be Exported：', height=1, width=40) \
            .grid(row=0, column=1, sticky=W)
        self.listbox = Listbox(master, width=40, height=len(SaveContent), listvariable=ListStr,
                                        selectmode="single")
        self.listbox.selection_set(0)
        self.listbox.grid(row=2, column=1, columnspan=2, padx=5, pady=5)
        self.okbutton = Button(master, text='OK', width=40, command=self.save)
        self.okbutton.grid(row=4, column=1, padx=5, pady=5)
        self.master.wait_window()
    def save(self):
        rst = self.listbox.curselection()
        rst = rst[0]
        self.output = self.SaveContent[rst]
        self.master.destroy()

class show_prob:
    def __init__(self,MainFig, resultList, dirname):
        self.dirname = dirname
        master = Toplevel(MainFig)
        master.resizable(0, 0)
        master.geometry('+800+500')
        self.master = master
        master.title('Probabilities')
        objectPlace, conditionPlace, pY, pYwithX, pXY, pYX, pNotYwithX, pYwithNotX, pNotYwithNotX, \
        pYGivenX, pYGivenXAhead, pYGivenXAfter, pYGivenNotX, pNotY, pNotYGivenX, pNotYGivenNotX = resultList
        width = 120
        self.text1 = 'The object place (Y) is {}'.format(objectPlace)
        self.instr1 = Label(master, text=self.text1, height=1, width=width).grid(row=0, column=1, columnspan=2, sticky=W)
        self.text2 = 'The condition place (X) is {}'.format(conditionPlace)
        self.instr2 = Label(master, text=self.text2, height=1, width=width).grid(row=1, column=1, columnspan=2, sticky=W)
        self.instr3 = Label(master, text='', height=1, width=50).grid(row=2, column=1, columnspan=2, sticky=W)
        self.text4 = 'P(Y): the probability that people will go to {} is {}'.format(objectPlace, pY)
        self.instr4 = Label(master, text=self.text4, height=1, width=width).grid(row=3, column=1, columnspan=2, sticky=W)
        self.text5 = 'P(Y&X): the joint probability that people will go to both {} and {} is {}'.format(objectPlace, conditionPlace, pYwithX)
        self.instr5 = Label(master, text=self.text5, height=1, width=width).grid(row=4, column=1, columnspan=2, sticky=W)
        self.text6 = 'P(X→Y): The joint probability that people will first go to {} and then go to {} is {}'.format(conditionPlace, objectPlace, pXY)
        self.instr6 = Label(master, text=self.text6, height=1, width=width).grid(row=5, column=1, columnspan=2, sticky=W)
        self.text7 = 'P(Y→X): The joint probability that people will first go to {} and then go to {} is {}'.format(objectPlace, conditionPlace, pYX)
        self.instr7 = Label(master, text=self.text7, height=1, width=width).grid(row=6, column=1, columnspan=2, sticky=W)
        self.text8 = 'P(Y&no-X): the joint probability that people will go to {} but not go to {} is {}'.format(objectPlace, conditionPlace, pYwithNotX)
        self.instr8 = Label(master, text=self.text8, height=1, width=width).grid(row=7, column=1, columnspan=2, sticky=W)
        self.text9 = 'P(no-Y&X): the joint probability that people will not go to {} but go to {} is {}'.format(objectPlace, conditionPlace, pNotYwithX)
        self.instr9 = Label(master, text=self.text9, height=1, width=width).grid(row=8, column=1, columnspan=2, sticky=W)
        self.text10 = 'P(no-Y&no-X): the joint probability that people will go to neither {} nor {} is {}'.format(objectPlace, conditionPlace, pNotYwithNotX)
        self.instr10 = Label(master, text=self.text10, height=1, width=width).grid(row=9, column=1, columnspan=2, sticky=W)
        self.instr11 = Label(master, text='', height=1, width=width).grid(row=10, column=1, columnspan=2, sticky=W)
        self.text12 = 'P(Y|X): the conditional probability that people will go to {} givien they go to {} is {}'\
            .format(objectPlace, conditionPlace, pYGivenX)
        self.instr12 = Label(master, text=self.text12, height=1, width=width).grid(row=11, column=1, columnspan=2, sticky=W)
        self.text13 = 'P(X→Y|X): the conditional probability that people will go to {} givien they have already gone to {} beforehand is {}'\
            .format(objectPlace, conditionPlace, pYGivenXAhead)
        self.instr13 = Label(master, text=self.text13, height=1, width=width).grid(row=12, column=1, columnspan=2, sticky=W)
        self.text14 = 'P(Y→X|X): the conditional probability that people will go to {} givien they would also go to {} afterwards is {}'\
            .format(objectPlace, conditionPlace, pYGivenXAfter)
        self.instr14 = Label(master, text=self.text14, height=1, width=width).grid(row=13,column=1, columnspan=2, sticky=W)
        self.text15 = 'P(Y|no-X): the conditional probability that people will go to {} givien they do not go to {} is {}'\
            .format(objectPlace, conditionPlace, pYGivenNotX)
        self.instr15 = Label(master,text=self.text15,height=1, width=width).grid(row=14, column=1, columnspan=2, sticky=W)
        self.instr16 = Label(master, text='', height=1, width=50).grid(row=15, column=1, columnspan=2, sticky=W)
        self.text17 = 'P(no-Y): the probability that people will not go to {} is {}'.format(objectPlace, pNotY)
        self.instr17 = Label(master, text=self.text17, height=1, width=width).grid(row=16, column=1, columnspan=2, sticky=W)
        self.text18 = 'P(no-Y|X): the conditional probability that people will not go to {} given they go to {} is {}'\
            .format(objectPlace, conditionPlace, pNotYGivenX)
        self.instr18 = Label(master, text=self.text18, height=1, width=width).grid(row=17, column=1, columnspan=2, sticky=W)
        self.text19 = 'P(no-Y|no-X): the conditional probability that people will not go to {} given they do not go to {} is {}'\
            .format(objectPlace, conditionPlace, pNotYGivenNotX)
        self.instr19 = Label(master,text=self.text19, height=1, width=width).grid(row=18, column=1, columnspan=2, sticky=W)
        self.instr20 = Label(master,text='',height=1).grid(row=19,column=1, columnspan=2)
        self.closeButton = Button(master, text='OK', width=20, command=self.close)
        self.closeButton.grid(row=20, column=1, padx=5, pady=5)
        self.saveButton = Button(master,text='Save', width=20, command=self.save)
        self.saveButton.grid(row=20, column=2, padx=5, pady=5)
        self.master.wait_window()
    def close(self):
        self.master.destroy()
    def save(self):
        if self.dirname:
            WhereToSave = asksaveasfilename(defaultextension='.txt', initialdir=self.dirname, filetypes=[('TXT', 'txt')])
        else:
            WhereToSave = asksaveasfilename(defaultextension='.txt', filetypes=[('TXT', 'txt')])
        if WhereToSave == '':
            return
        with open(WhereToSave, 'w+') as f:
            f.write(self.text1 + '\n')
            f.write(self.text2 + '\n')
            f.write('\n')
            f.write(self.text4 + '\n')
            f.write(self.text5 + '\n')
            f.write(self.text6 + '\n')
            f.write(self.text7 + '\n')
            f.write(self.text8 + '\n')
            f.write(self.text9 + '\n')
            f.write(self.text10 + '\n')
            f.write('\n')
            f.write(self.text12 + '\n')
            f.write(self.text13 + '\n')
            f.write(self.text14 + '\n')
            f.write(self.text15 + '\n')
            f.write('\n')
            f.write(self.text17 + '\n')
            f.write(self.text18 + '\n')
            f.write(self.text19)
        showinfo('Message', 'Finished')
        self.master.destroy()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def print_sample(sample_ix, ix_to_char):
    txt = '→'.join(ix_to_char[ix] for ix in sample_ix)
    print(txt)


def get_initial_loss(place_size, seq_length):
    return -np.log(1.0 / place_size) * seq_length


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



def initialize_parameters(n_a, n_x, n_y):
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
    Waa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
    Wya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
    b = np.zeros((n_a, 1))  # hidden bias
    by = np.zeros((n_y, 1))  # output bias
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    return parameters


def rnn_step_forward(parameters, a_prev, x):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)  # hidden state
    p_t = softmax(
        np.dot(Wya, a_next) + by)  # unnormalized log probabilities for next chars # probabilities for next chars

    return a_next, p_t


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next']  # backprop into h
    daraw = (1 - a * a) * da  # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b'] += -lr * gradients['db']
    parameters['by'] += -lr * gradients['dby']
    return parameters


def rnn_forward(X, Y, a0, parameters, place_size):
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0
    for t in range(len(X)):
        x[t] = np.zeros((place_size, 1))
        if (X[t] != None):
            x[t][X[t]] = 1
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t - 1], x[t])
        loss -= np.log(y_hat[t][Y[t], 0])
    cache = (y_hat, a, x)
    return loss, cache


def rnn_backward(X, Y, parameters, cache):
    gradients = {}
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t - 1])
    return gradients, a

def clip(gradients, maxValue):
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient,-maxValue , maxValue, out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients

def optimize(X, Y, a_prev, parameters, place_size, learning_rate = 0.01):
    loss, cache = rnn_forward(X,Y,a_prev,parameters,place_size = place_size)
    gradients, a = rnn_backward(X,Y,parameters,cache)
    gradients = clip(gradients,5)
    parameters = update_parameters(parameters,gradients,learning_rate)
    return loss, gradients, a[len(X)-1]

def estimate_rnn_model(data, ix_to_place, place_to_ix, place_size, MainFig, lr, routesNum=7, num_iterations = 10000, n_a = 50):
    n_x, n_y = place_size, place_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(place_size, routesNum)
    np.random.shuffle(data)
    a_prev = np.zeros((n_a, 1))
    # 进度条
    progressToplevel = Toplevel(MainFig)
    progressToplevel.geometry('+800+300')
    progressToplevel.title('RNN model')
    progressLabel = Label(progressToplevel, text='Estimating RNN model.')
    progressLabel.grid(row=0, column=0)
    progress = ttk.Progressbar(progressToplevel, orient='horizontal',
                               length=800, mode='determinate')
    progress.grid(row=1, column=0)
    progress['value'] = 0
    progress['maximum'] = num_iterations
    # 进度条
    # Optimization loop
    stable = 0
    up = 0
    lossList = list()
    for j in range(num_iterations):
        index = j%len(data)
        X = [None] + [place_to_ix[place] for place in data[index]]
        Y = X[1:] + [place_to_ix["over"]]
        curr_loss, gradients, a_prev = optimize(X,Y,a_prev,parameters,learning_rate=lr, place_size=place_size)
        loss = smooth(loss, curr_loss)
        lossList.append(loss)
        if len(lossList) > 1:
            crtLossChange = (lossList[-2] - lossList[-1]) / lossList[-2]
            if abs(crtLossChange) < 1e-7:
                stable += 1
            else:
                stable = 0
            if crtLossChange < 0:
                up += 1
            else:
                up = 0

        try:
            progressLabel['text'] = 'Estimating RNN model using Gradient Descent, iteration: {}/{}.\n ' \
                                    'Current loss is {}. The loss is not imporved for {} iterations.' \
                .format(j + 1, num_iterations, round(loss, 3), stable)

            progress['value'] = j + 1
            progressToplevel.update()
            if stable > 20:
                progressToplevel.destroy()
                return parameters
            if up > 50:
                showwarning('Warning', 'The algorithm diverge, you are suggested to lower the learning rate and try again. Please refer to the trend of cost.')
                progressToplevel.destroy()
                return parameters, lossList
        except:
            showinfo('Message','The algorithm is interupted, current parameters is returned.')
            return parameters, lossList
    progressToplevel.destroy()
    return parameters, lossList

def rnn_model_route_probability(parameters, place_to_ix, route):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    place_size = by.shape[0]
    n_a = Waa.shape[1]
    x = np.zeros((place_size, 1))
    a_prev = np.zeros((n_a, 1))
    over_ix = place_to_ix['over']
    route_length = len(route)
    P = 1
    for place in route:
        a = np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+b)
        z = np.dot(Wya,a)+by
        y = softmax(z)
        p = y.ravel()
        crt_chosen_idx = place_to_ix[place]
        P = P * p[crt_chosen_idx]
        # go on with the next step.
        x = np.zeros((place_size,1))
        x[crt_chosen_idx] = 1
        # Update "a_prev" to be "a"
        a_prev = a
    return P

def rnn_model_route_with_next_step_probability(parameters, place_to_ix, route):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    place_size = by.shape[0]
    n_a = Waa.shape[1]
    x = np.zeros((place_size, 1))
    a_prev = np.zeros((n_a, 1))
    over_ix = place_to_ix['over']
    route_length = len(route)
    P = 1
    for placeId in route:
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        p = y.ravel()
        P = P * p[placeId]
        # go on with the next step.
        x = np.zeros((place_size, 1))
        x[placeId] = 1
        # Update "a_prev" to be "a"
        a_prev = a
    # last step
    a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
    z = np.dot(Wya, a) + by
    y = softmax(z)
    p = y.ravel()
    P = P * p
    return P


def rnn_model_beam_search(parameters, place_to_ix, ix_to_place, L, K, useOver, startWith=[]):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    place_size = by.shape[0]
    n_a = Waa.shape[1]
    over_ix = place_to_ix['over']
    # if there is no start part
    if not len(startWith):
        x = np.zeros((place_size, 1))
        a_prev = np.zeros((n_a, 1))
        # The first step
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        p = y.ravel()
        # first step: no over
        p[over_ix] = -1000
        sort_p = -np.sort(-p)
        idx_sort_p = np.argsort(-p)
        indices = [[[xx] for xx in idx_sort_p[:K]]]
        probs = [sort_p[:K]]
    else:
        startWithIdx = [place_to_ix[place] for place in startWith]
        startWithWithNextStepProb = rnn_model_route_with_next_step_probability(parameters, place_to_ix, startWithIdx)
        sort_p = -np.sort(-startWithWithNextStepProb)
        idx_sort_p = np.argsort(-startWithWithNextStepProb)
        indices = [[  startWithIdx + [xx] for xx in idx_sort_p[:K]   ]]
        probs = [sort_p[:K]]
    # useOver = true
    if useOver:
        while 1:
            availableRoutes = []
            correspondingProb = []
            for k in range(min(K,len(indices[-1]))):
                crtRoute = indices[-1][k]
                if crtRoute[-1] != over_ix:
                    crtRouteWithNextStepProb = rnn_model_route_with_next_step_probability(parameters, place_to_ix, crtRoute)
                    for i in range(place_size):
                        availableRoutes.append(crtRoute+[i])
                        correspondingProb.append(crtRouteWithNextStepProb[i])
                else:
                    crtRoutePlace = [ix_to_place[ix] for ix in crtRoute]
                    crtRouteProb = rnn_model_route_probability(parameters, place_to_ix, crtRoutePlace)
                    availableRoutes.append(crtRoute)
                    correspondingProb.append(crtRouteProb)
            correspondingProb = np.array(correspondingProb)
            sort_p = -np.sort(-correspondingProb)
            idx_sort_p = np.argsort(-correspondingProb)
            crtChosenRoute = [availableRoutes[j] for j in idx_sort_p[:K]]
            crtChosenRouteProb = sort_p[:K]
            indices.append(crtChosenRoute)
            probs.append(crtChosenRouteProb)
            overNum = 0
            lenBreak = 0
            for item in crtChosenRoute:
                if len(item) >= K:
                    lenBreak = 1
                if item[-1] == over_ix:
                    overNum += 1
            if overNum == K or lenBreak==1:
                break
    # useOver = false
    else:
        l = 1
        while l < L:
            l += 1
            availableRoutes = []
            correspondingProb = []
            for k in range(min(K,len(indices[-1]))):
                crtRoute = indices[-1][k]
                crtRouteWithNextStepProb = rnn_model_route_with_next_step_probability(parameters, place_to_ix,
                                                                                      crtRoute)
                crtRouteWithNextStepProb[over_ix] = -1000
                for i in range(place_size):
                    availableRoutes.append(crtRoute + [i])
                    correspondingProb.append(crtRouteWithNextStepProb[i])
            correspondingProb = np.array(correspondingProb)
            sort_p = -np.sort(-correspondingProb)
            idx_sort_p = np.argsort(-correspondingProb)
            crtChosenRoute = [availableRoutes[j] for j in idx_sort_p[:K]]
            crtChosenRouteProb = sort_p[:K]
            indices.append(crtChosenRoute)
            probs.append(crtChosenRouteProb)
    return indices, probs

def rnn_accuracy(parameters, place_to_ix, data):
    routes = data['Route'].tolist()
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    place_size = by.shape[0]
    n_a = Waa.shape[1]
    N = 0
    N_1bet = 0
    N_3bet = 0
    N_5bet = 0
    for route1 in routes:
        route = route1 + ['over']
        x = np.zeros((place_size, 1))
        a_prev = np.zeros((n_a, 1))
        over_ix = place_to_ix['over']
        route_length = len(route)
        for place in route:
            N += 1
            a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
            z = np.dot(Wya, a) + by
            y = softmax(z)
            p = y.ravel()
            crt_argsort_idx = np.argsort(-p)
            crt_chosen_idx = place_to_ix[place]
            if crt_chosen_idx == crt_argsort_idx[0]:
                N_1bet += 1
            if crt_chosen_idx in crt_argsort_idx[:3]:
                N_3bet += 1
            if crt_chosen_idx in crt_argsort_idx[:5]:
                N_5bet += 1
            # go on with the next step.
            x = np.zeros((place_size, 1))
            x[crt_chosen_idx] = 1
            # Update "a_prev" to be "a"
            a_prev = a
    accuracy1 = N_1bet / N
    accuracy3 = N_3bet / N
    accuracy5 = N_5bet / N
    return accuracy1, accuracy3, accuracy5

class beamSearchSetting:
    def __init__(self, MainFig, uniquePlaces):
        self.uniquePlaces = uniquePlaces
        self.K = 1
        self.maxLength = 5
        self.useOver = True
        self.StartWith = []
        width = 120
        master = Toplevel(MainFig)
        master.resizable(0,0)
        master.geometry('+800+300')
        self.master = master
        master.title('Beam Search Setting')

        text1 = 'Please set the K parameter.'
        text2 = '(In each step, K routes with the largest probabilities would kept for searching at the next step)'
        text3 = 'Please set the maxinum length of the route.'
        self.instr1 = Label(master, text='', height=1, width=width).grid(row=0, column=1)
        self.instr2 = Label(master, text=text1, height=1, width=width).grid(row=1, column=1)
        self.instr3 = Label(master, text=text2, height=1, width=width).grid(row=2, column=1)
        self.KEntryVar = StringVar()
        self.KEntryVar.set(str(self.K))
        self.KEntry = Entry(master, textvariable=self.KEntryVar, width=50).grid(row=3, column=1)
        Label(master, text='', height=1, width=50).grid(row=4, column=1)
        self.instr4 = Label(master, text=text3,height=1, width=width).grid(row=5, column=1)
        self.maxLengthEntryVar = StringVar()
        self.maxLengthEntryVar.set(str(self.maxLength))
        self.maxLengthEntry = Entry(master, textvariable=self.maxLengthEntryVar,width=50).grid(row=6, column=1)
        Label(master, text='', height=1, width=50).grid(row=7, column=1)
        self.useOverCheckVar = IntVar()
        self.useOverCheck = Checkbutton(master, text='Stop searching when "'"over"'" is chosen?', variable=self.useOverCheckVar)
        self.useOverCheck.grid(row=8,column=1)
        if self.useOver:
            self.useOverCheck.select()
        else:
            self.useOverCheck.deselect()
        text5 = '(Optional) You could set the starting part of the route by inputing a sequence of places separated by comma (,).'
        Label(master, text='', height=1, width=50).grid(row=9, column=1)
        self.instr5 = Label(master, text=text5,height=1, width=width).grid(row=10, column=1)
        self.StartWithEntryVar = StringVar()
        self.StartWithEntry = Entry(master, textvariable=self.StartWithEntryVar, width=50).grid(row=11, column=1)
        Label(master, text='', height=1, width=50).grid(row=12, column=1)
        self.okbutton = Button(master, text='OK', width=30, command=self.save)
        self.okbutton.grid(row=13, column=1)
        self.master.wait_window()
    def save(self):
        try:
            self.K = int(self.KEntryVar.get())
        except:
            showerror('Error', 'Invalid K. Use the default value.')
            self.K = 1
        if self.K  <= 0:
            showerror('Error', 'Invalid K. Use the default value.')
            self.K = 1
        try:
            self.maxLength = int(self.maxLengthEntryVar.get())
        except:
            showerror('Error', 'Invalid maximum route length. Use the default value.')
            self.maxLength = 5
        if self.maxLength <= 0 :
            showerror('Error', 'Invalid maximum route length. Use the default value.')
            self.maxLength = 5
        self.useOver = self.useOverCheckVar.get()
        try:
            temp = self.StartWithEntryVar.get()
            startWithList = temp.split(',')
        except:
            showerror('Error','Invalid start part of the route.')
            self.StartWith = []
        if len(temp) == 0:
            self.StartWith = []
        else:
            self.StartWith = startWithList
            for place in startWithList:
                if not place in self.uniquePlaces:
                    showerror('Error', '{} is not a valid place. Start part of the route would be omited.'.format(place))
                    self.StartWith = []
                    break
                if place == 'over':
                    showerror('Error', 'You can not set "'"over"'". Start part of the route would be omited.')
                    self.StartWith = []
                    break
                if len(startWithList) >= self.maxLength:
                    showerror('Error','The maximum length of the route is {}, but the length of its starting part is already {}'.format(
                        self.maxLength, len(startWithList)
                    ))
                    self.StartWith = []
                    break
        self.master.destroy()

class MainFig:
    def __init__(self, root):
        center_window(root, 500, 0)
        self.root = root
        self.dirname = None
        self.purefilename = None
        self.data = []
        self.uniquePlaces = []
        self.LR = []
        self.meanLR = []
        self.APRoutesList = []
        self.CoOccList = []
        self.Rules = []
        self.expectOccMatrix = []
        self.coreness = []
        self.rnnModelList = []
        self.beamList = []
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label='Read Route Data', command=self.read_data)
        filemenu.add_command(label='Export CSV Results', command=self.export_csv)
        menubar.add_cascade(label="File", menu=filemenu)

        clustermenu = Menu(menubar, tearoff=0)
        clustermenu.add_command(label='Analyze Route Similarity', command = self.calc_levenstein_ratio_matrix)
        clustermenu.add_command(label='Mean Similarity with All Routes', command=self.mean_similarity_with_all_routes)
        clustermenu.add_command(label='Affinity Propagation Clustering', command=self.ap_clustering)
        menubar.add_cascade(label='Clustering', menu=clustermenu)

        associationmenu = Menu(menubar, tearoff=0)
        associationmenu.add_command(label='Co-occurrence Matrix', command=self.CoOccuranceMatrix)
        associationmenu.add_command(label='Core/Periphery: Coreness', command=self.coreness_cal)
        associationmenu.add_command(label='Probabilities Concerning 2 Places', command=self.prob_cal)
        associationmenu.add_command(label='Apriori', command=self.place_apriori)
        menubar.add_cascade(label='Association', menu=associationmenu)

        predictmenu = Menu(menubar, tearoff=0)
        predictmenu.add_command(label='Recurrent Neural Network', command=self.rnn_model)
        predictmenu.add_command(label='Evaluate Accuracy of RNN', command=self.accuracy_rnn)
        predictmenu.add_command(label='Route Probability', command=self.route_probability)
        predictmenu.add_command(label='Step Probability', command=self.step_prbability)
        predictmenu.add_command(label='Beam Search for Typical Route', command=self.beam_search)
        menubar.add_cascade(label='Predict', menu=predictmenu)

        root.config(menu=menubar)


    def read_data(self, save_path=None):
        # showinfo('Help', 'Open the csv file')
        filename = askopenfilename(defaultextension='.csv', filetypes=[('CSV', 'csv')])
        if filename == '':
            return
        self.dirname = os.path.dirname(filename)
        self.purefilename = os.path.basename(filename).split('.')[0]
        use_existed = False
        raw_data_pkl_filename = os.path.splitext(filename)[0] + '_raw_route_pkl'
        if os.path.exists(raw_data_pkl_filename):
            use_existed = askyesno('File', 'Preprocessing file exists, use it?')
        if use_existed:
            with open(raw_data_pkl_filename, 'rb') as f:
                data, uniquePlaces = pickle.load(f)
        else:
            data = pd.DataFrame(None, columns=['Individual', 'Route'])
            places = list()
            file = pd.read_csv(filename, dtype=np.str_)
            ncol = file.shape[1]
            for item in range(file.shape[0]):
                path_list_raw = file.iloc[item, 1:ncol + 1].tolist()
                path_list = [ele for ele in path_list_raw if str(ele) != 'nan']
                for place in path_list:
                    places.append(place)
                # print(path_list)
                data.loc[data.shape[0] + 1] = {'Individual': file.iloc[item, 0], 'Route': path_list}
            places = np.array(places)
            uniquePlaces = np.unique(places.flatten())
            with open(raw_data_pkl_filename,'wb') as f:
                pickle.dump([data, uniquePlaces], f)
        self.data = data
        self.uniquePlaces = uniquePlaces
        msg = 'Finished.'
        if data.shape[0] < 5:
            msg += ' Routes are as follows.'
            k = data.shape[0]
        else:
            msg += ' The first 5 routes are as follows.'
            k = 5
        for i in range(k):
            msg += '\n' + data.iloc[i,0] + ': ' + '→'.join(data.iloc[i,1])
        showinfo('Message', msg)

    def calc_levenstein_ratio_matrix(self, save_path=None):
        use_existed = False
        dirname = self.dirname
        purefilename = self.purefilename
        if not dirname or not purefilename:
            showerror('Error', 'There is no valid raw data')
            return
        LR_pkl_filename = os.path.join(dirname, purefilename+'_LR_pkl')
        if os.path.exists(LR_pkl_filename):
            use_existed = askyesno('File', 'Pairwise similarity file already exists, use it?')
        if use_existed:
            with open(LR_pkl_filename, 'rb') as f:
                result = pickle.load(f)
        else:
            data = self.data
            if not data.size:
                showerror('Error', 'There is no valid raw data')
                return
            N = data.shape[0]
            result = np.ones((N,N))
            # 进度条
            progressToplevel = Toplevel(root)
            progressToplevel.geometry('+800+300')
            progressToplevel.title('Route Similarity')
            progressLabel = Label(progressToplevel, text='Calculating Levenstein Ratio Matrix……')
            progressLabel.grid(row=0, column=0)
            progress = ttk.Progressbar(progressToplevel, orient='horizontal',
                                       length=800, mode='determinate')
            progress.grid(row=1, column=0)
            progress['value'] = 0
            progress['maximum'] = N * (N-1) / 2
            k = 0
            # 进度条
            for i in range(N-1):
                for j in range(i+1, N):
                    k += 1
                    progressLabel['text'] = 'Calculating the similarity betweeen the ' + str(k + 1) + 'th pair of routes, totally ' \
                                            + str(round(N * (N-1) / 2)) + 'pairs，' + \
                                            str(round((k + 1) / (N * (N-1) / 2) * 100, 2))+ '% finished.'
                    progress['value'] = k + 1
                    progressToplevel.update()
                    if i == j:
                        continue
                    list1 = data.iloc[i, 1]
                    list2 = data.iloc[j, 1]
                    l1 = len(list1)
                    l2 = len(list2)
                    dist = edit_distance2(list1,list2)
                    result[i, j] = (l1+l2-dist) / (l1+l2)
                    result[j, i] = (l1+l2-dist) / (l1+l2)

            result = pd.DataFrame(result, index=data['Individual'])
            with open(LR_pkl_filename, 'wb') as f:
                pickle.dump(result, f)
            progressToplevel.destroy()
        self.LR = result
        showinfo('Message', 'Finished')


    def mean_similarity_with_all_routes(self):
        LR = self.LR
        if not LR.size:
            showerror('Error', 'You must generate a Levenstein Ratoi Matrix')
            return
        temp = LR.mean(axis=1)
        meanLR = pd.DataFrame(temp, columns= ['MeanSimilarity'])
        data = self.data
        data.index = data['Individual']
        ind = meanLR.index
        for i in range(ind.shape[0]):
            meanLR.loc[ind[i], 'Route'] = '→'.join(data.loc[ind[i], 'Route'])
        meanLR = meanLR.sort_values(by='MeanSimilarity', ascending=False)
        self.meanLR = meanLR
        showinfo('Message', 'The route with the largest similarity with all other routes is "'"{}"'"'
                 .format(meanLR.ix[0,'Route']))


    def ap_clustering(self):
        use_existed = False
        dirname = self.dirname
        purefilename = self.purefilename
        if not dirname or not purefilename:
            showerror('Error', 'There is no valid raw data')
            return
        AP_pkl_filename = os.path.join(dirname, purefilename + '_AP_pkl')
        if os.path.exists(AP_pkl_filename):
            use_existed = askyesno('File', 'Affinity Propagation clustering result already exists, use it?')
        if use_existed:
            with open(AP_pkl_filename, 'rb') as f:
                APRoutesList = pickle.load(f)
                APRoutes, APCluster = APRoutesList
        else:
            data = self.data
            if not data.size:
                showerror('Error', 'There is no valid raw data')
            data.index = data['Individual']
            ind = self.LR.index
            LR = np.array(self.LR)
            if not LR.size:
                showerror('Error', 'You must generate a Levenstein Ratoi Matrix')
                return
            # 进度条
            medianS = round(np.median(LR),2)
            prefFlag = askyesno('Setting', 'Use default preference (the median of the similarity matrix: {}) for all routes?\n\n'
                                           '(It is recommended if you have no idea about which route should be given priority to be an exemplar '
                                           'and how many exemplars you expect.)'.format(medianS))
            if not prefFlag:
                prefFlag2 = askyesno('Setting', 'Use a general preference for all routes? Select "'"No"'" to set route-specific preferences.')
                if prefFlag2:
                    pref = askfloat('Setting', 'Please input the general preference.\n The default value is the median of similarity matrix ({}).\n'
                                               'Higher value is expected to output more clusters, vice versa.'.format(medianS))
                    if not pref:
                        return
                    try:
                        pref = float(pref)
                    except:
                        showerror('Invalid input')
                        return
                else:
                    prompt = '\n'.join(['Please open a CSV File containning preference value for each route.\n','The file should have 2 columns: ',
                                                          'the 1st column is the index of the route, which should be same as the input data; '
                                                          'the 2nd column is the preference value of the route, which should be a number; '
                                                          'the file should use "'"Individual"'" and  "'"Preference"'" as the header. \n',
                                        'The larger the preference of a route, the more likely it is selected to be an examplar.',
                                        'Reference: the default value of general preference is the median of similaritiy matrix: {}'.format(medianS)])
                    showinfo('Setting', prompt)
                    prefFile = askopenfilename(defaultextension='.csv', filetypes=[('CSV', 'csv')])
                    if not prefFile:
                        return
                    with open(prefFile) as f2:
                        pref = pd.read_csv(prefFile)
                        try:
                            prefInd = pref['Individual'].tolist()
                            prefIndStr = [str(x) for x in prefInd]
                            if data['Individual'].tolist() == prefIndStr:
                                prefList = pref['Preference'].tolist()
                                try:
                                    pref = [float(mmm) for mmm in prefList]
                                except:
                                    showerror('Error','There exists invalid preference value.')
                                    return
                            else:
                                showerror('Error', 'Indices of routes do not keep the same.')
                                return
                        except:
                            showerror('Error', 'The file is invalid.')
                            return
                        if len(pref) != data.shape[0]:
                            showerror('Error', 'The are {} routes in the data, but only {} valid preference values.'.format(data.shape[0], len(pref)))
                            return
            progressToplevel = Toplevel(root)
            progressToplevel.geometry('+800+300')
            progressToplevel.title('AP Clustering')
            progressLabel = Label(progressToplevel, text='Running Affinity Propagation Clustering')
            progressLabel.grid(row=0, column=0)
            progress = ttk.Progressbar(progressToplevel, orient='horizontal',
                                       length=600, mode='determinate')
            progress.grid(row=1, column=0)
            progress['value'] = 0
            # 进度条
            if prefFlag:
                pref = None

            try:
                AFInstance = AffinityPropagationCustom(affinity='precomputed', verbose=True, max_iter=1900,
                                                       convergence_iter=10, processToplevel=progressToplevel,
                                                       processLabel=progressLabel, processBar=progress, preference=pref)
            except:
                AFInstance = AffinityPropagation(affinity='precomputed', verbose=True, max_iter=1000,
                                                 convergence_iter=10, preference=pref)
            AFInstance.fit(LR)
            progressToplevel.destroy()
            Labels = AFInstance.labels_
            Centers = AFInstance.cluster_centers_indices_
            CenterCount = np.bincount(Labels)
            SortedIndexofCenterCount = np.argsort(-CenterCount)
            # ExtractIndex = SortedIndexofCenterCount[:NoTerm]
            ExtractInds = [ind[Centers[i]] for i in SortedIndexofCenterCount]
            ExtractRoutes = ['→'.join(data.loc[ind, 'Route']) for ind in ExtractInds]
            ExtractWeights = [CenterCount[i] for i in SortedIndexofCenterCount]
            APRoutes =  pd.DataFrame(ExtractInds, columns= ['Individual'])
            APRoutes['Route'] = ExtractRoutes
            APRoutes['Representativeness'] = ExtractWeights
            APCluster = pd.DataFrame(None, columns=['Individual', 'Route','Cluster','Exemplar'])
            APCluster['Individual'] = data['Individual']
            for i in range(data.shape[0]):
                APCluster.ix[i, 'Route'] = '→'.join(data.iloc[i,1])
                APCluster.ix[i, 'Cluster'] = Centers[Labels[i]] + 1
                APCluster.ix[i,'Exemplar'] = '→'.join(data.iloc[Centers[Labels[i]],1])
            APRoutesList = [APRoutes, APCluster]
            with open(AP_pkl_filename,'wb') as f:
                pickle.dump(APRoutesList, f)
        self.APRoutesList = APRoutesList
        showinfo('Message', 'The route with the largest representativeness is "'"{}"'", it represents {} routes'
                     .format(APRoutes.ix[0, 'Route'], APRoutes.ix[0, 'Representativeness']))

    def CoOccuranceMatrix(self):
        use_existed = False
        dirname = self.dirname
        purefilename = self.purefilename
        if not dirname or not purefilename:
            showerror('Error', 'There is no valid raw data')
            return
        CoOcc_pkl_filename = os.path.join(dirname, purefilename + '_CoOcc_pkl')
        if os.path.exists(CoOcc_pkl_filename):
            use_existed = askyesno('File', 'Co-occurance matrix already exists, use it?')
        if use_existed:
            with open(CoOcc_pkl_filename, 'rb') as f:
                CoOccList = pickle.load(f)
        else:
            data = self.data
            if not data.size:
                showerror('Error', 'There is no valid raw data')
            CoOccDict = dict()
            for i in range(data.shape[0]):
                route = data.iloc[i, 1]
                routeCombinations = combinations(route, 2)
                for item in routeCombinations:
                    if item in CoOccDict.keys():
                        CoOccDict[item] += 1
                    else:
                        CoOccDict[item] = 1
            places = np.array([item for item in CoOccDict.keys()])
            uniquePlaces = np.unique(places.flatten())
            OcOccMatrix = np.zeros((uniquePlaces.size, uniquePlaces.size))
            for key,value in CoOccDict.items():
                rowIndex = np.where(uniquePlaces == key[0])
                columnIndex = np.where(uniquePlaces == key[1])
                OcOccMatrix[rowIndex, columnIndex] = value
            OcOccMatrix = OcOccMatrix + OcOccMatrix.T
            CoOccList = [uniquePlaces, OcOccMatrix, CoOccDict]  #最后保存的是唯一场所名，基于该名的共现矩阵，以及单独的共现字典
            with open(CoOcc_pkl_filename,'wb') as f:
                pickle.dump(CoOccList, f)
        self.CoOccList = CoOccList
        showinfo('Message', 'Finished')

    def prob_cal(self):
        data = self.data
        if not data.size:
            showerror('Error', 'There is no valid raw data')
            return
        dirname = self.dirname
        purefilename = self.purefilename
        if not dirname or not purefilename:
            showerror('Error', 'There is no valid raw data')
            return
        uniquePlaces = self.uniquePlaces
        objectPlace = askstring('Object', 'Please input the object place:')
        if not objectPlace:
            return
        if objectPlace not in uniquePlaces:
            showerror('Error', 'This place does not exist')
            return
        conditionPlace = askstring('Condition', 'The probabilities of goint to {} are conditioned on which place?'
                                   .format(objectPlace))
        if not conditionPlace:
            return
        if conditionPlace not in uniquePlaces:
            showerror('Error', 'This place does not exist')
            return
        if conditionPlace == objectPlace:
            showerror('Error', 'The probabilities of object place can not be conditioned on itself.')
            return
        N = data.shape[0]
        numY = 0
        numX = 0
        numNotX = 0
        numYandX = 0
        numYandXAhead = 0
        numYandXAfter = 0
        numYandNotX = 0
        numNotY = 0
        numNotYandX = 0
        numNotYandNotX = 0
        for i in range(data.shape[0]):
            route = data.iloc[i, 1]
            if conditionPlace in route:
                numX += 1
            else:
                numNotX += 1
            if objectPlace in route:
                numY += 1
                if conditionPlace in route:
                    numYandX += 1
                    firstXIndex = np.where(np.array(route)==conditionPlace)[0][0]
                    firstYIndex = np.where(np.array(route)==objectPlace)[0][0]
                    if firstXIndex < firstYIndex:
                        numYandXAhead += 1
                    else:
                        numYandXAfter += 1
                else:
                    numYandNotX += 1
            else:
                numNotY += 1
                if conditionPlace in route:
                    numNotYandX += 1
                else:
                    numNotYandNotX += 1
        if N > 0:
            pY = numY / N
            pNotY = numNotY / N
        else:
            pY = 'Invalid'
            pNotY = 'Invalid'
        if numX > 0:
            pYGivenX = numYandX / numX
            pYwithX = numYandX / N
            pXY =  numYandXAhead / N
            pYX = numYandXAfter / N
            pNotYwithX = numNotYandX / N
            pYGivenXAhead = numYandXAhead / numX
            pYGivenXAfter = numYandXAfter / numX
            pNotYGivenX = numNotYandX / numX
        else:
            pYGivenX = 'Invalid'
            pYGivenXAhead = 'Invalid'
            pYGivenXAfter = 'Invalid'
            pNotYGivenX = 'Invalid'
        if numNotX > 0:
            pYwithNotX = numYandNotX / N
            pNotYwithNotX = numNotYandNotX / N
            pYGivenNotX = numYandNotX / numNotX
            pNotYGivenNotX = numNotYandNotX / numNotX
        else:
            pYGivenNotX = 'Invalid'
            pNotYGivenNotX = 'Invalid'
        resultList = [objectPlace, conditionPlace, pY, pYwithX, pXY, pYX, pNotYwithX, pYwithNotX, pNotYwithNotX,
                      pYGivenX, pYGivenXAhead, pYGivenXAfter, pYGivenNotX, pNotY, pNotYGivenX, pNotYGivenNotX]
        show_probInstance = show_prob(root, resultList, dirname)


    def place_apriori(self):
        use_existed = False
        dirname = self.dirname
        purefilename = self.purefilename
        if not dirname or not purefilename:
            showerror('Error', 'There is no valid raw data')
            return
        apriori_pkl_filename = os.path.join(dirname, purefilename + '_apriori_pkl')
        if os.path.exists(apriori_pkl_filename):
            use_existed = askyesno('File', 'Apriori results already exists, use it?')
        if use_existed:
            with open(apriori_pkl_filename, 'rb') as f:
                Rules = pickle.load(f)
        else:
            data = self.data
            if not data.size:
                showerror('Error', 'There is no valid raw data')
            routeList = list()
            for i in range(data.shape[0]):
                routeList.append(data.iloc[i, 1])
            AprioriSettingInstance = AprioriSetting(root)
            minSupport = AprioriSettingInstance.minSupport
            minConfidence = AprioriSettingInstance.minConfidence
            minLift = AprioriSettingInstance.minLift
            mode = AprioriSettingInstance.mode
            apriori_results = apriori(routeList, min_support=minSupport, min_confidence=minConfidence, min_lift=minLift)
            Rules = analyzeAprioriResults(apriori_results, minSupport, minConfidence, minLift)
            if mode == 1:
                Rules = Rules.sort_values(by='Support', ascending=False)
            elif mode == 2:
                Rules = Rules.sort_values(by='Confidence', ascending=False)
            elif mode == 3:
                Rules = Rules.sort_values(by='Lift', ascending=False)
            with open(apriori_pkl_filename, 'wb') as f:
                pickle.dump(Rules, f)
        self.Rules = Rules
        if len(Rules) == 0:
            showinfo('Message', 'Can not find any rule, please lower the support/confidence/lift.')
        else:
            showinfo('Message', 'Finished. The first rule is {}, with support = {}, confidence = {}, lift = {}'.format(
                Rules.iloc[0,0], Rules.iloc[0,1], Rules.iloc[0,2],Rules.iloc[0,3]
            ))

    def coreness_cal(self):
        use_existed = False
        dirname = self.dirname
        purefilename = self.purefilename
        if not dirname or not purefilename:
            showerror('Error', 'There is no valid raw data')
            return
        coreness_pkl_filename = os.path.join(dirname, purefilename + '_coreness_pkl')
        if os.path.exists(coreness_pkl_filename):
            use_existed = askyesno('File', 'Coreness results already exists, use it?')
        if use_existed:
            with open(coreness_pkl_filename, 'rb') as f:
                corenessResults = pickle.load(f)
                coreness, expectOccMatrix, gof, NCore, SuggestCorePlace = corenessResults
        else:
            CoOccList = self.CoOccList
            if not len(CoOccList):
                showerror('Error', 'You should generate Co-occurance matrix first.')
                return
            uniquePlaces = CoOccList[0]
            OcOccMatrix = CoOccList[1]
            maxit = askinteger('Setting', 'Please set the maximum iterations.', initialvalue = 5000)
            if not maxit:
                return
            instr = 'Please set the learning rate. \n\n' \
                    '(Larger learning rate will do it quicker, \nbut the algorithm might fail to converge for too large value.|'
            lr = askfloat('Learning Rate', instr,initialvalue=0.001)
            try:
                lr = float(lr)
            except:
                showerror('Error','Invalid input')
                return
            if lr < 0:
                showerror('Error', 'Learning rate must be larger than 0.')
                return
            result = decompose(OcOccMatrix, root, lr=lr, maxit=maxit)
            if not len(result):
                return
            st_core, unst_core, expectOccMatrix, gof, lossList = result
            coreness = pd.DataFrame(None, columns=['Place', 'Coreness', 'Coreness(Unstandardized)', 'SuggestedCore'])
            coreness['Place'] = uniquePlaces
            coreness['Coreness'] = st_core
            coreness['Coreness(Unstandardized)'] = unst_core
            coreness = coreness.sort_values(by='Coreness', ascending=False)
            NCore, SuggestCorePlace = suggestedCore(coreness)
            temp = np.zeros(coreness.shape[0])
            temp[:NCore] = 1
            coreness['SuggestedCore'] = temp
            expectOccMatrix = pd.DataFrame(expectOccMatrix, columns=uniquePlaces, index=uniquePlaces)
            corenessResults = [coreness, expectOccMatrix, gof, NCore, SuggestCorePlace]
            with open(coreness_pkl_filename, 'wb') as f:
                pickle.dump(corenessResults, f)
        self.coreness = coreness
        self.expectOccMatrix = expectOccMatrix
        if NCore <= 10:
            mainMessage =  'Finished. The correlation coefficient between observed and expected co-occurance matrix is {}. ' \
                           'The top {} places with the largest coreness are suggested as cores: {}.'.format(
                gof, NCore, SuggestCorePlace
            )
        else:
            mainMessage = 'Finished. The correlation coefficient between observed and expected co-occurance matrix is {}. ' \
                          'The top {} places with the largest coreness are suggested as cores.'.format(
                gof, NCore
            )

        if 'lossList' in vars():
            Message = mainMessage + '\n\n Trend of loss would show as a reference.'
            showinfo('Message', Message)
            plt.plot(np.array(lossList))
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.title('Trend of Cost')
            plt.show()
        else:
            showinfo('Message', mainMessage)

    def rnn_model(self):
        use_existed = False
        dirname = self.dirname
        purefilename = self.purefilename
        if not dirname or not purefilename:
            showerror('Error', 'There is no valid raw data')
            return
        rnn_pkl_filename = os.path.join(dirname, purefilename + '_rnn_pkl')
        if os.path.exists(rnn_pkl_filename):
            use_existed = askyesno('File', 'A RNN model has already existed, use it?')
        if use_existed:
            with open(rnn_pkl_filename, 'rb') as f:
                rnnModelList = pickle.load(f)
                rnnParameter, ix_to_place, place_to_ix = rnnModelList
        else:
            data = self.data
            if not data.size:
                showerror('Error', 'There is no valid raw data')
                return
            uniquePlaces = self.uniquePlaces.tolist()
            uniquePlaces.append('over')
            routes = data['Route'].tolist()
            data_size, place_size = len(data), len(uniquePlaces)
            place_to_ix = {place: i for i, place in enumerate(sorted(uniquePlaces))}
            ix_to_place = {i: place for i, place in enumerate(sorted(uniquePlaces))}
            maxit = askinteger('Setting', 'Please set the maximum iterations.', initialvalue=10000)
            if not maxit:
                return
            if maxit <= 0:
                showerror('The maximum iterations must be larger than 0.')
                return
            n_a = askinteger('Setting', 'Please set the number of hidden units.', initialvalue=50)
            if not n_a:
                return
            if n_a <= 0:
                showerror('The number of hidden units must be larger than 0.')
                return
            instr = '\n'.join(['Please set the learning rate. ',
                    '(Larger learning rate will do it quicker,','but the algorithm might fail to converge for too large value.)'])
            lr = askfloat('Learning Rate', instr, initialvalue=0.01)
            try:
                lr = float(lr)
            except:
                showerror('Error', 'Invalid input')
                return
            if lr < 0:
                showerror('Error', 'Learning rate must be larger than 0.')
                return
            rnnParameter, lossList = estimate_rnn_model(routes, ix_to_place, place_to_ix, len(uniquePlaces), self.root, lr, num_iterations=maxit, n_a=n_a)
            if not len(rnnParameter):
                return
            rnnModelList = [rnnParameter, ix_to_place, place_to_ix]
            with open(rnn_pkl_filename, 'wb') as f:
                pickle.dump(rnnModelList, f)
        self.rnnModelList = rnnModelList
        if 'lossList' in vars():
            showinfo('Message', 'Finished. Trend of cost would show as a reference.')
            plt.plot(np.array(lossList))
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.title('Trend of Cost')
            plt.show()
        else:
            showinfo('Message', 'Finished.')

    def accuracy_rnn(self):
        uniquePlaces = self.uniquePlaces.tolist()
        if not uniquePlaces:
            showerror('Error', 'There is no valid raw data')
            return
        uniquePlaces.append('over')
        uniquePlaces = [i.strip() for i in uniquePlaces]
        rnnModelList = self.rnnModelList
        if not rnnModelList:
            showerror('Error', 'You must have a RNN model first.')
            return
        rnnParameter, ix_to_place, place_to_ix = rnnModelList
        whichData = askyesno('Data','Evaluate RNN on current dataset?')
        if whichData:
            data = self.data
        else:
            filename = askopenfilename(defaultextension='.csv', filetypes=[('CSV', 'csv')])
            if filename == '':
                return
            data = pd.DataFrame(None, columns=['Individual', 'Route'])
            places = list()
            file = pd.read_csv(filename, dtype=np.str_)
            ncol = file.shape[1]
            for item in range(file.shape[0]):
                path_list_raw = file.iloc[item, 1:ncol + 1].tolist()
                path_list = [ele for ele in path_list_raw if str(ele) != 'nan']
                for place in path_list:
                    places.append(place)
                data.loc[data.shape[0] + 1] = {'Individual': file.iloc[item, 0], 'Route': path_list}
            places = np.array(places)
            uniquePlacesNewData = np.unique(places.flatten())
            for newPlace in uniquePlacesNewData:
                if newPlace not in uniquePlaces:
                    showerror('Error', 'The new dataset contains unknow place(s).')
                    return
        accuracy1, accuracy3, accuracy5 = rnn_accuracy(rnnParameter, place_to_ix, data)
        showinfo('Accuracy', '\n'.join(['The accuracy is {}'.format(accuracy1),
                                        'The accuracy for top-3 bet is {}'.format(accuracy3),
                                        'The accuracy for top-5 bet is {}'.format(accuracy5),
                                        '\n The accuracy for randomly guess is {}'.format(1/len(uniquePlaces))]))

    def route_probability(self):
        uniquePlaces = self.uniquePlaces.tolist()
        if not uniquePlaces:
            showerror('Error', 'There is no valid raw data')
            return
        uniquePlaces.append('over')
        uniquePlaces = [i.strip() for i in uniquePlaces]
        rnnModelList = self.rnnModelList
        if not rnnModelList:
            showerror('Error', 'You must have a RNN model first.')
            return
        rnnParameter, ix_to_place, place_to_ix = rnnModelList
        routeStr = askstring('Input','\n'.join(['Please input the object route: a sequence of places separated by comma (,). ',
                             'You can use "'"over"'" to indicate the end of the route.']))
        if not routeStr:
            return
        route = routeStr.split(',')
        route = [i.strip() for i in route]
        for i in route:
            if i not in uniquePlaces:
                showerror('Error', '"'"{}"'" is not a valid place.'.format(i))
                return
        P = rnn_model_route_probability(rnnParameter, place_to_ix, route)
        routeLen = len(route)
        placeLen = len(uniquePlaces)
        P0 = (1/placeLen)**routeLen

        resultList = ['The probability for route "'"{}"'" is {}.'.format('→'.join(route), P),
                      'which is {} times of the probability of a random route with the same length.'.format(round(P/P0,3))]
        showinfo('Result', '\n\n'.join(resultList))

    def step_prbability(self):
        uniquePlaces = self.uniquePlaces.tolist()
        if not uniquePlaces:
            showerror('Error', 'There is no valid raw data')
            return
        uniquePlaces.append('over')
        uniquePlaces = [i.strip() for i in uniquePlaces]
        rnnModelList = self.rnnModelList
        if not rnnModelList:
            showerror('Error', 'You must have a RNN model first.')
            return
        rnnParameter, ix_to_place, place_to_ix = rnnModelList
        routeStr = askstring('Input',
                             '\n'.join(['Please input a route: a sequence of places separated by comma (,). ',
                                        'You can use "'"over"'" to indicate the end of the route.',
                                        'The object "'"from→to"'" is contained at the last of the route.',
                                        'For instance, given a route "'"...→M→P→A→B"'", the step probability of "'"A→B"'" would be calculated.']))
        if not routeStr:
            return
        route = routeStr.split(',')
        if len(route) < 2:
            showerror('Error', 'The length of the route should be at least 2.')
            return
        route = [i.strip() for i in route]
        for i in route:
            if i not in uniquePlaces:
                showerror('Error', '"'"{}"'" is not a valid place.'.format(i))
                return
        PA = rnn_model_route_probability(rnnParameter, place_to_ix, route[:-1])
        PB = rnn_model_route_probability(rnnParameter, place_to_ix, route)
        PAB = PB / PA
        placeLen = len(uniquePlaces)
        P0 = 1 / placeLen
        formorRoute = '→'.join(route[:-2])
        if len(formorRoute):
            formorRoute = ' given the previous route ' + "'" + formorRoute + "'"
        crtRoute = '→'.join(route[-2:])
        resultList = ['The probability for the step choice "'"{}"'"'.format(crtRoute) + formorRoute + ' is {}'.format(PAB),
                      'which is {} times of the probability of a random choice.'.format(
                          round(PAB / P0, 3))]
        showinfo('Result', '\n\n'.join(resultList))

    def beam_search(self):
        uniquePlaces = self.uniquePlaces.tolist()
        if not uniquePlaces:
            showerror('Error', 'There is no valid raw data')
            return
        uniquePlaces.append('over')
        uniquePlaces = [i.strip() for i in uniquePlaces]
        rnnModelList = self.rnnModelList
        if not rnnModelList:
            showerror('Error', 'You must have a RNN model first.')
            return
        rnnParameter, ix_to_place, place_to_ix = rnnModelList
        use_existed = False
        dirname = self.dirname
        purefilename = self.purefilename
        if not dirname or not purefilename:
            showerror('Error', 'There is no valid raw data')
            return
        beam_pkl_filename = os.path.join(dirname, purefilename + '_beam_pkl')
        if os.path.exists(beam_pkl_filename):
            use_existed = askyesno('File', 'Typcial routes found by beam search have already existed, use it?')
        if use_existed:
            with open(beam_pkl_filename, 'rb') as f:
                beamList = pickle.load(f)
                indices, probs, beamRouteDF = beamList
        else:
            beamSearchSettingInstance = beamSearchSetting(root, uniquePlaces)
            K = beamSearchSettingInstance.K
            L = beamSearchSettingInstance.maxLength
            useOver = beamSearchSettingInstance.useOver
            startWith = beamSearchSettingInstance.StartWith
            indices, probs = rnn_model_beam_search(rnnParameter, place_to_ix, ix_to_place, L, K, useOver, startWith)
            beamRouteDF = pd.DataFrame(None, columns=['Route', 'Probability'])
            lastIndices = indices[-1]
            lastProbs = probs[-1]
            for i in range(len(lastIndices)):
                crtRouteIdx = lastIndices[i]
                crtRoute = [ix_to_place[j] for j in crtRouteIdx]
                beamRouteDF.loc[beamRouteDF.shape[0] + 1] = {
                    'Route': '→'.join(crtRoute), 'Probability':lastProbs[i]
                }
            with open(beam_pkl_filename, 'wb') as f:
                beamList = [indices, probs, beamRouteDF]
                pickle.dump(beamList, f)
        self.beamList = beamList
        finalRouteIdx = indices[-1][0]
        finalRoute = [ix_to_place[i] for i in finalRouteIdx]
        if finalRoute[-1] == 'over':
            finalRoute = finalRoute[:-1]
        finalProb = probs[-1][0]
        showinfo('Result','\n'.join(['Finished. The typical route with the largest probability found by Beam Search is ',
                                     '"'"{}"'", '.format('→'.join(finalRoute)),
                                     'with the probability of {}'.format(finalProb)]))

    def export_csv(self):
        dirname = self.dirname
        purefilename = self.purefilename
        # data
        LR = self.LR
        meanLR = self.meanLR
        APRoutesList = self.APRoutesList
        CoOccList = self.CoOccList
        Rules = self.Rules
        coreness = self.coreness
        expectOccMatrix = self.expectOccMatrix
        beamList = self.beamList

        SaveContent = []
        if len(LR):
            SaveContent.append('Levenstein Ratio Matrix (Similarity Matrix)')
        if len(meanLR):
            SaveContent.append('Mean Similarity with All Routes')
        if len(APRoutesList):
            SaveContent.append('AP Cluster Result: Representative Routes')
            SaveContent.append('AP Cluster Result: Cluster Membership')
        if len(CoOccList):
            SaveContent.append('Co-occurance Matrix')
        if len(Rules):
            SaveContent.append('Apriori Rules')
        if len(coreness):
            SaveContent.append('Core/Periphery: Coreness')
        if len(expectOccMatrix):
            SaveContent.append('Core/Periphery: Expected Co-occurance Matrix')
        if len(beamList):
            SaveContent.append('Typical Routes Found by RNN and Beam Search')

        if not len(SaveContent):
            showerror('Error', 'There is nothing you can export now.')
            return
        SaveCSVInstance = save_csv_class(root, SaveContent)
        if not len(SaveCSVInstance.output):
            return
        if dirname:
            WhereToSave = asksaveasfilename(defaultextension='.csv', initialdir=dirname, filetypes=[('CSV', 'csv')])
        else:
            WhereToSave = asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', 'csv')])
        if WhereToSave == '':
            return
        SaveContent = SaveCSVInstance.output
        UseIndex = False
        if SaveContent == 'Levenstein Ratio Matrix (Similarity Matrix)':
            LR2 = LR
            LR2.columns = LR2.index
            SaveFile = LR2
            UseIndex = True
        elif SaveContent == 'Mean Similarity with All Routes':
            SaveFile = meanLR
        elif SaveContent == 'AP Cluster Result: Representative Routes':
            SaveFile = APRoutesList[0]
        elif SaveContent == 'AP Cluster Result: Cluster Membership':
            SaveFile = APRoutesList[1]
        elif SaveContent == 'Co-occurance Matrix':
            SaveFile = pd.DataFrame(CoOccList[1], columns=CoOccList[0], index=CoOccList[0])
            UseIndex = True
        elif SaveContent == 'Apriori Rules':
            SaveFile = Rules
        elif SaveContent == 'Core/Periphery: Coreness':
            SaveFile = coreness
        elif SaveContent == 'Core/Periphery: Expected Co-occurance Matrix':
            SaveFile = expectOccMatrix
            UseIndex = True
        elif SaveContent == 'Typical Routes Found by RNN and Beam Search':
            indices, probs, beamRouteDF = beamList
            SaveFile = beamRouteDF

        try:
            SaveFile.to_csv(WhereToSave, index=UseIndex, sep=',')
            showinfo('Message', 'Finished.')
        except:
            showerror('Error', 'Can not export the result.')





if __name__ == '__main__':
    root = Tk()
    root.resizable(0, 0)
    root.title('Utilities for Spatial Route Analysis')
    MainFig = MainFig(root=root)
    root.mainloop()