stream = cuda.Stream()
with cuda.stream(stream):
    while True:
        image, label = retrieve_data()
        image.to('cuda:0')
        label.to('cuda:0')
        cuda.synchronize()
        queue.put((image, label))
        queue.join()

while True:
    image, label = queue.get()
    queue.task_done()
    output = model(image)
    ...