import torch

def objective(trial):
    N_TRAIN_EXAMPLES = CONFIG['train_batch_size'] * 30
    N_VALID_EXAMPLES = CONFIG['valid_batch_size'] * 10
    device = CONFIG['device']

    # Generate the model.
    model = AnimalsModel(CONFIG['model_name'], labels)

    # Generate the optimizers.
    optimizer_number = trial.suggest_categorical('optimizer number (0: Adam; 1: SGD)', [0, 1])
    lr = trial.suggest_float('learning rate', 1e-2, 1e-1, log=True)
    
    optimizer_map = {
        0: 'Adam',
        1: 'SGD',
    }
    
    optimizer = getattr(optim, optimizer_map[optimizer_number])(model.parameters(), lr=lr)
    
    config = {
        'optimizer (0: Adam; 1: SGD)': optimizer_number,
        'learning rate': lr,
    }
    
    run = wandb.init(project='Animals10',
                     name=f'trial_{trial.number + 1}',
                     group='optuna research',
                     config=config,
                     anonymous='must')

    # Training of the model.
    for epoch in range(CONFIG['epochs']):
        model.train()
    
        dataset_size = 0
        running_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * CONFIG['train_batch_size'] > N_TRAIN_EXAMPLES:
                break

            images = data['image'].to(device, dtype=torch.float)
            targets = data['target'].to(device, dtype=torch.long)
            targets = torch.reshape(targets, (-1,))

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # zero the parameter gradients
            optimizer.zero_grad()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * CONFIG['valid_batch_size'] > N_VALID_EXAMPLES:
                    break
                
                images = data['image'].to(device, dtype=torch.float)
                targets = data['target'].to(device, dtype=torch.long)
                targets = torch.reshape(targets, (-1,))

                batch_size = images.size(0)

                outputs = model(images)
                loss = criterion(outputs, targets)

                running_loss += (loss.item() * batch_size)
                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        run.log({'Cross Entropy Loss': epoch_loss})

        trial.report(epoch_loss, epoch)

        # handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    run.finish()
    
    return epoch_loss

study = optuna.create_study()
study.optimize(objective, n_trials=20, timeout=1800, show_progress_bar=True)

print('Number of finished trials: {}'.format(len(study.trials)))

print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))

print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
          
# Create the summary run.
summary = wandb.init(project='Animals10',
                     group='optuna summary',
                     name='summary')

# Getting the study trials.
trials = study.trials

# WandB summary.
for step, trial in enumerate(trials):
    # Logging the loss.
    summary.log({'Cross Entropy Loss': trial.value}, step=step, commit=False)

    # Logging the parameters.        
    summary.log(trial.params, commit=True)
    
summary.finish()