import tkinter as tk
from tkinter import filedialog
from matplotlib.pyplot import text
import numpy as np
import random
import matplotlib.pyplot as plt

def file_path(file_entry):
    file_path = filedialog.askopenfilename()
    file_entry.delete(0,"end")
    file_entry.insert(0, file_path)

def read_file(file_path):
    file = open(file_path,mode='r')
    return file

def threshod_fun(x, Group):
        if(x >= 0):
            return Group[1]
        return Group[0]

def rate(weight,identify_data,data,Group):
    N = len(identify_data)
    count = 0
    for i in identify_data:
        V = sum(weight * i)
        Y = threshod_fun(V, Group)
        rows = np.where((data[:,0] == i[1]) & (data[:,1] == i[2]))
        if(len(rows[0]) == 0):
                continue
        else:
            rows = rows[0][0]
            d = data[rows, 2]
            if(d == Y):
                count+=1
            else:
                continue
    
    identify_rate = (count / N ) * 100
    return identify_rate
def train(data,train_data,learn_rate,train_round,weight,Group):
    for turn in range(train_round):
        for i in train_data:
            V = sum(weight * i)
            Y = threshod_fun(V, Group)
            rows = np.where((data[:,0] == i[1]) & (data[:,1] == i[2]))
            if(len(rows[0]) == 0):
                continue
            else:
                rows = rows[0][0]
                d = data[rows, 2]
                if(d == Y):
                    continue
                else:
                    if (d < Y):
                        weight = weight - (learn_rate * i)
                    elif(d > Y):
                        weight = weight + (learn_rate * i)
            temp_rate = rate(weight,train_data,data,Group)
            if(temp_rate >= 90):
                return weight
                break
    return weight

def scatter_plot(data,train_data,weight,Group):
        x = train_data[:,1]
        y = (-1*x*weight[1]+weight[0])/weight[2]
        plt.Figure(figsize= (2,2), dpi = 20)
        for i in data:
            if(i[2] == Group[0]):
                plt.scatter(i[0], i[1], c= 'red')
            elif(i[2] == Group[1]):
                plt.scatter(i[0], i[1], c = 'green')
            else:
                plt.scatter(i[0], i[1], c = 'blue')
        plt.plot(x,y, c= 'orange')
        plt.show()
        return
def perceptron(file_entry,learn_entry,round_entry,result1_label,result2_label,weight_label):
    try:
        learn_rate = float(learn_entry.get())
        train_round = int(round_entry.get())
        file_path = file_entry.get()
        data = np.loadtxt(read_file(file_path))
    except:
        weight_label.configure(text='請填寫完整資訊')
        return
    
    Group = []
    for i in data:
        if(i[2] not in  Group):
            Group.append(i[2])
    Group.sort()
    
    tr_len = round((len(data) * 2) / 3) + 1
    train_data = np.full((1, 3), -1)
    test_data = np.full((1, 3), 0)

    #input training data
    while True:
        inputdata = random.choice(data)
        inputdata = np.array([(inputdata)])
        inputdata = np.delete(inputdata, 2, axis= 1)
        inputdata = np.insert(inputdata,[0],-1,axis=1)
        train_data = np.append(train_data, inputdata, axis=0)
        train_data = np.unique(train_data, axis=0)
        if(len(train_data) == tr_len) :
            train_data = np.delete(train_data, 0,axis=0)
            break
    # input testing data
    for i in data:
        rows = np.where((i[0] == train_data[:,1]) & (i[1] == train_data[:,2]))
        if(len(rows[0]) == 0):
            test_data = np.append(test_data, [i], axis = 0)
    
    test_data = np.delete(test_data, 2, axis= 1)
    test_data = np.insert(test_data,[0],-1,axis=1)
    test_data = np.delete(test_data, 0, axis= 0)
    
    weight = np.array([-1, 0, 1])
    train_weight = train(data,train_data,learn_rate,train_round,weight,Group)

    train_rate = round(rate(train_weight,train_data,data,Group))
    test_rate = rate(train_weight,test_data,data,Group)
    
    result1_label.configure(text = '訓練辨識率:'+ str(train_rate) + '%')
    result2_label.configure(text = '測試辨識率:'+ str(test_rate) + '%')
    weight_label.configure(text='訓練後鍵結值:'+str(train_weight))
    scatter_plot(data,train_data,train_weight,Group)    

window = tk.Tk()
window.title('Perception')
window.geometry('320x160')

file_label = tk.Label(window,text='訓練資料(.txt)')
file_label.grid(row = 1, column = 1)
file_entry = tk.Entry(window)
file_entry.grid(row = 1, column = 2)
file_btn = tk.Button(window, text='...',command=lambda: file_path(file_entry))
file_btn.grid(row=1, column=3)

learn_label = tk.Label(window, text='學習率')
learn_label.grid(row=2, column=1)
learn_entry = tk.Entry(window)
learn_entry.grid(row=2, column=2)


round_label = tk.Label(window, text='訓練次數(收斂條件)')
round_label.grid(row=3, column=1)
round_entry= tk.Entry(window)
round_entry.grid(row=3, column=2)

weight_label = tk.Label(window)
weight_label.grid(row=6, column=1,columnspan=2)

result1_label = tk.Label(window)
result1_label.grid(row=4, column=1,columnspan=2)

result2_label = tk.Label(window)
result2_label.grid(row=5, column=1,columnspan=2)

check_btn = tk.Button(window, text='學習',command=lambda: perceptron(file_entry,learn_entry,round_entry,result1_label,result2_label,weight_label))
check_btn.grid(row=7, column=2)

window.mainloop()
