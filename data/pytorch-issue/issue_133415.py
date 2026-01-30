for i, data in enumerate(train_loader, 0):
    # Get the inputs; data is a tuple of (inputs, labels)
    inputs, labels = data

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = loss_function(outputs, labels)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Periodically checkpoint
    
# Load checkpoint   
 
# Resume training from ckpt
for i, data in enumerate(train_loader, 0):  
    # Get the inputs; data is a tuple of (inputs, labels)
    inputs, labels = data

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = loss_function(outputs, labels)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Periodically checkpoint