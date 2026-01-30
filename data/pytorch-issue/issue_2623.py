import torch
import torch.nn as nn
import math

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.Softmax()
  
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        output = output.add(1e-8)
        output = output.log()

        return output, hidden
        

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size).cuda())

n_iters = 40000
print_every = 200
plot_every = 200

# # Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

tp = 0
tn = 0
fp = 0
fn = 0

precision = 0
recall = 0
fmeasure = 0

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)

    if loss != -1:
        current_loss += loss

        guess, guess_i = categoryFromOutput(output)
        if guess == -1 and guess_i == -1:
            continue
        else:                
            correct = '1' if guess == category else '0 (%s)' % category
            if guess == 'class1' and category == 'class1':
                tp += 1
            elif guess == 'class2' and category == 'class2':
                fn += 1
            elif guess == 'class1' and category == 'class2':
                fp += 1
            else:
                tn += 1
            
            if iter % print_every == 0:
                loss = current_loss / print_every
                print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
                all_losses.append(current_loss / plot_every)
                current_loss = 0

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    if(line_tensor.dim() == 0):
        return line_tensor
    else:
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        return output

def predict(input_line, category, n_predictions=1):
    output = evaluate(Variable(lineToTensor(input_line)).cuda())
    global total
    global indian
    global nonindian

    total += 1
    if(output.dim() != 0):
        topv, topi = output.data.topk(1, 1, True)

        for i in range(0, n_predictions):
            value = topv[0][i]
            category_index = topi[0][i]

            if category_index <= 1:
                if all_categories[category_index] == 'indian':
                    indian += 1
                else:
                    nonindian += 1
                predictions.append([value, all_categories[category_index], category])