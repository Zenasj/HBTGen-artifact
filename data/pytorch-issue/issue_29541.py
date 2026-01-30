with open(r"cat.jpg", 'rb') as f:
    image_bytes = f.read()
tensor = []
t = preprocess.transform_image(image_bytes)
tensor.append(t)
res = predict.do_predict(tensor)