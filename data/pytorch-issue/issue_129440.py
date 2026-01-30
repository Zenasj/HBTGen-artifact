def forward(self, x):
      c1 = self.count
      self.count += 1
      c2 = self.count
      return x + c1 + c2

def forward(self, inp):
  x = self.buffer
  self.buffer += 1
  y = self.buffer
  return x + y + inp

def forward(self, x):
  if pred:
      self.w += 1
  else:
      self.w += 2
  return x + self.w