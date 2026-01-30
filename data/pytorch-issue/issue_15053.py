import torch.nn.functional as F

def forward(self, x): 
        residual = x 
        out = self.conv1(x)
        out = F.dropout(out, training=True)
        out = self.relu(out)
        out = self.conv2(out)
        out = F.dropout(out, training=True)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def forward(self, x): 
        residual = x 

        out = self.conv1(x)
        out = F.dropout(out, training=True)
        out = self.relu(out)

        out = self.conv2(out)
        out = F.dropout(out, training=True)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out