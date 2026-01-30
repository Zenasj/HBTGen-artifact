with tqdm(loader, total=len(train_dataset)//hps.train.batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()

                lengths = batch['lengths'].to('mps')
                vec = batch['vec'].to('mps')
                pit = batch['pit'].to('mps')
                spk = batch['spk'].to('mps')
                mel = batch['mel'].to('mps')

                prior_loss, diff_loss, mel_loss, spk_loss = model.compute_loss(
                    lengths, vec, pit, spk,
                    mel, out_size=out_size,
                    skip_diff=skip_diff_train)
                
                loss = sum([prior_loss, diff_loss, mel_loss, spk_loss])
                loss.backward()