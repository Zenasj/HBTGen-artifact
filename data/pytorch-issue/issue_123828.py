import multiprocessing
import torch 
import torch.nn as nn 
 
# Change the hidden from 128 to 6 will erase the stuck?
strange_hidden = 128

lstm = nn.LSTM(input_size=2, hidden_size=strange_hidden, num_layers=1, batch_first=True)
torch.save(lstm.state_dict(),"./plain.pt") 

def run(inputs): 
    model = nn.LSTM(input_size=2, hidden_size=strange_hidden, num_layers=1, batch_first=True)
    model.load_state_dict(torch.load("./plain.pt")) 
    x=model(inputs) 
    return x 

state=torch.rand(1,1,2) 
print("CHECKPOINT 0 PASS")

# Comment the line below will erase the stuck
lstm.load_state_dict(torch.load("./plain.pt"))
# Comment the line above will erase the stuck
# What is weird, if I change the strange hidden_size from 128 to 6, it would never stuck!


x = lstm(state)
print(x[0].shape)
print("CHECKPOINT 1 PASS")

pool = multiprocessing.Pool(5)
for i in range(5):
    pool.apply_async(func=run,args=(state,))
pool.close()
pool.join()
print("CHECKPOINT 2 PASS")

# WINDOWS VERSION
import multiprocessing
import torch 
import torch.nn as nn 

def run(inputs): 
        model = nn.LSTM(input_size=2, hidden_size=strange_hidden, num_layers=1, batch_first=True)
        model.load_state_dict(torch.load("./plain.pt")) 
        x=model(inputs) 
        return x 


if __name__=="__main__":
    strange_hidden = 128

    lstm = nn.LSTM(input_size=2, hidden_size=strange_hidden, num_layers=1, batch_first=True)
    torch.save(lstm.state_dict(),"./plain.pt") 

    state=torch.rand(1,1,2) 
    print("CHECKPOINT 0 PASS")

    lstm.load_state_dict(torch.load("./plain.pt"))


    x = lstm(state)
    print(x[0].shape)
    print("CHECKPOINT 1 PASS")

    pool = multiprocessing.Pool(5)
    for i in range(5):
        pool.apply_async(func=run,args=(state,))
    pool.close()
    pool.join()
    print("CHECKPOINT 2 PASS")