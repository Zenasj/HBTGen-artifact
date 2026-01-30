import torch

def data_transorm(input):
    b,n,c = input.shape
    input = input.reshape(b*n,c)
    offset = [(i+1)*n for i in range(b)]
    return [input[:,:3].contiguous(),input[:,3].unsqueeze(-1).contiguous(),torch.IntTensor(offset)]

input_xyz,input_r,offset = data_transorm(input)
input_xyz = Variable(input_xyz,requires_grad=True)
input_r = Variable(input_r,requires_grad=True)
logits = model([input_xyz,input_r,offset])
loss = criterion(logits, target.squeeze())
loss.backward()

def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)

        b = o0.shape[0]
        n,c = x.shape
        x = x.reshape(b,n//b,c)
        return x

input = Variable(input_xyz,requires_grad=True)
logits = model(input)
loss = criterion(logits, target.squeeze())
loss.backward()

def forward(self, pc):
        n,c = pc.shape
        p0, x0, o0 = pc[...,:3].clone().contiguous(),pc,torch.IntTensor([n]).cuda() # (n, 3), (n, c), (b)
        # x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)

        # b = o0.shape[0]
        # n,c = x.shape
        # x = x.reshape(b,n//b,c)
        return x