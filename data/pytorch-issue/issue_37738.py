writer = SummaryWriter()
writer.add_hparams({"lr" : 0.001}, {"accuracy" : 0})
for epoch in range(epochs):
    ...
    writer.add_scalar("accuracy", get_accuracy(epoch), epoch)

with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

3
with SummaryWriter(log_dir=logdir) as w_hp:
    w_hp.file_writer.add_summary(exp)
    w_hp.file_writer.add_summary(ssi)
    w_hp.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        w_hp.add_scalar(k, v, global_step)