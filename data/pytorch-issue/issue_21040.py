def forward(self):
        x = self.array
        x.append(0)
        self.array = x
        return self.array