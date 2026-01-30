import torch
import math

logger.info('Training....')
best_val_loss = []
stored_loss = 100000000

def train(train_dataloader=train_dataloader):
    model.train()
    start_time = datetime.now()
    batch_losses = 0
    batch_metrics = 0  
    predict_proba = []
    for questions,answers, targets in train_dataloader:

        if args.use_gpu:
            questions,answers , targets = Variable(questions.cuda()), Variable(answers.cuda()), Variable(targets.cuda())
        else:
            questions,answers, targets = Variable(questions), Variable(answers), Variable(targets)

        optimizer.zero_grad()
        model.zero_grad()
        outputs = model(questions,answers)
        batch_loss = criterion(outputs, targets)
        batch_metric = accuracy(outputs, targets)
        batch_loss.backward(retain_graph=True)
        optimizer.step()

        batch_losses += batch_loss.data
        batch_metrics += batch_metric.data
    train_data_size = len(train_dataloader.dataset)
    epoch_loss = batch_losses / train_data_size
    epoch_metric = batch_metrics / train_data_size
    elapsed = datetime.now() - start_time
    print('| epoch {:3d} | lr {:05.5f} | ms/epoch {} | '
            'loss {:5.2f} | acc {:8.3f}  | ppl {:8.3f} | bpc {:8.3f}'.format(
        epoch,  optimizer.param_groups[0]['lr'],
        str(elapsed) , epoch_loss, epoch_metric , math.exp(epoch_loss), epoch_loss / math.log(2)))
    

def evaluate(dataloader):
    # validation
    model.eval()
    val_batch_losses = 0
    val_batch_metrics = 0
    predict_proba = []
    for val_questions,val_answers, val_targets in dataloader:

        if args.use_gpu:
            val_questions,val_answers , val_targets = Variable(val_questions.cuda()), Variable(val_answers.cuda()), Variable(val_targets.cuda())
        else:
            val_questions,val_answers, val_targets = Variable(val_questions), Variable(val_answers), Variable(val_targets)

        val_outputs = model(val_questions,val_answers)
        val_batch_loss = criterion(val_outputs, val_targets)
        val_batch_metric = accuracy(val_outputs, val_targets)
        val_batch_losses += val_batch_loss.data
        val_batch_metrics += val_batch_metric.data        
        proba = [x[1] for x in torch.exp(val_outputs).cpu().data.numpy()]
        predict_proba.extend(proba)
    return val_batch_losses/len(dataloader.dataset), val_batch_metrics/len(dataloader.dataset),predict_proba

def ks_stat(outputs, labels,good_cnts,bad_cnts):
    _, argmax = outputs.max(dim=1)
    good = (argmax.data == 1).cpu().float()
    bad = (argmax.data == 0).cpu().float()
    good[0] += good_cnts
    bad[0] += bad_cnts 
    good_cnt = torch.cumsum(good, dim=0)
    bad_cnt = torch.cumsum(bad, dim=0)
    val = torch.abs(bad_cnt/preprocessor.total_bad - good_cnt/preprocessor.total_good).max().cpu().float().data
    
    return good_cnt[-1],bad_cnt[-1],val

def accuracy(outputs, labels):
    maximum, argmax = outputs.max(dim=1)
    corrects = argmax == labels # ByteTensor
    n_corrects = corrects.float().sum() # FloatTensor
    return n_corrects

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)
        
def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)
        
        
from scipy.stats import ks_2samp

ks_statistics = lambda y_pred, y_true: ks_2samp(y_pred[y_true==1],y_pred[y_true != 1]).statistic

for epoch in range(args.epochs + 1):
    start_time = datetime.now()

    train()
    val_epoch_loss, val_epoch_metric,predict_proba = evaluate(test_dataloader)
    print('-' * 130)
    print('| end of epoch {:3d} | time: {} | valid loss {:5.3f} | valid acc {:5.3f} | '
        'valid ppl {:8.3f} | valid bpc {:8.3f}'.format(
      epoch, str(datetime.now() - start_time), val_epoch_loss,val_epoch_metric  ,math.exp(val_epoch_loss), val_epoch_loss / math.log(2)))
    print('-' * 130)
    
    if val_epoch_loss < stored_loss:
        model_save('model.best')
        print('Saving model (new best validation)')
        stored_loss = val_epoch_loss

    if epoch %2 ==0 :
        print('Saving model before learning rate decreased')
        model_save('{}.e{}'.format(start_time.strftime('%m%d-%H%M%S'), epoch))
        print('recuce learning rate by multiply 0.7')
        optimizer.param_groups[0]['lr'] * 0.7

    best_val_loss.append(val_epoch_loss)