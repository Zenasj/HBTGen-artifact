import torch
import numpy as np

bestValidationLoss = float("inf")
def EvaluateModel(trial):
  bs = 16
  #bs = trial.suggest_categorical('BatchSize',[1,2,4,8,16,32,64])
  rd = trial.suggest_uniform('RecurrentDropout',0.25,0.45)
  #rd = trial.suggest_uniform('RecurrentDropout',0.1,0.5)
  ts = 'ordinary'
  #ts = trial.suggest_categorical('TrainingSet',['ordinary','adjusted'])
  print("trial:",trial.number," batchSize:",bs," recurrentDropout:",rd," ts:",ts)

  if ts=='ordinary':
    trainLoader = DataLoader(trainDataset,batch_size=bs,shuffle=True)
  elif ts=='adjusted':
    trainLoader = DataLoader(adjustedTrainDataset,batch_size=bs,shuffle=True)

  model = BuildModel.Model(sourceZSlices,sourceBands,sourceHeight,sourceWidth,rd)
  model.to(device)

  optimizer = optim.Adam(model.parameters(), lr=0.0001)

  pmodel = Model(model,optimizer,criterion)
  pmodel.to(device)

  timer.tic()
  history = pmodel.fit_generator(trainLoader,valid_generator=validateLoader,callbacks=[bestModelRestore,reduceLROnPlateau,earlyStopping],epochs=numberOfEpochs)
  timer.toc()
  print("pmodel.fit_generator took " + '{:.1f}'.format(timer.elapsed) + " s")

  losses = [epoch['loss'] for epoch in history]
  validationLosses = [epoch['val_loss'] for epoch in history]
  validationLoss = min(validationLosses)

  global bestValidationLoss
  if validationLoss < bestValidationLoss:
    modelName = 'model-' + str(trial.number) + '-' + str(bs) + '-' + '{:0.4f}'.format(rd) + '-' + '{:.3f}'.format(validationLoss)
    plt.plot(losses)
    plt.plot(validationLosses)
    validationLoss = min(validationLosses)
    validationLossStr = '{:.4f}'.format(validationLoss)
    validationLossEpoch = str(np.argmin(validationLosses))
    plt.title('Model loss' + '(best: ' + validationLossStr + ', epoch:' + validationLossEpoch + ')')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper right')
    #plt.show()
    plt.savefig(modelName + ' losses.png', bbox_inches='tight')
    plt.close()

    pmodel.model.eval()
    print("saving model.state_dict()",modelName + '.ptsd')
    torch.save(pmodel.model.state_dict(),modelName + '.ptsd')
    try:
      print("saving to ",modelName + '.pt')
      torch.save(pmodel.model,modelName + '.pt')
    except:
      print("problems saving to ",modelName + '.pt')
    try:
      print("creating torchscript intermediate")
      intermediate = torch.jit.script(pmodel.model)
      print("saving torchscript intermediate to",modelName + '.ptj')
      torch.jit.save(intermediate,modelName + '.ptj')
    except:
      print("problems saving to ",modelName + '.ptj')

    bestValidationLoss = validationLoss

  return validationLoss