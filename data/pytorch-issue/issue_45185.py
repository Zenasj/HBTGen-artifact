import torch

for i in range(10):    # <----- Added loop
  per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
  per_channel_quantized_model.eval()
  per_channel_quantized_model.fuse_model()
  per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
  #print(per_channel_quantized_model.qconfig)

  torch.quantization.prepare(per_channel_quantized_model, inplace=True)
  evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
  torch.quantization.convert(per_channel_quantized_model, inplace=True)
  top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
  print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)