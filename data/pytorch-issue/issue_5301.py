def create_dataloader():
      # some function to create a dataloader
      return dataloader

# in main function
dataloader = create_dataloader()
dataloader_iter = iter(dataloader)
for step in range(len(dataloader)):
  while(true):
     try:
        data = next(dataloader_iter)
     except:
        dataloader = create_dataloader()
        dataloader_iter = iter(dataloader)

   #perform training