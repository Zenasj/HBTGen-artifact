# suppose self.print == True
def forward(self):
    if self.print:
        print("log line")
        self.print = False

# suppose self.knob == True 
def forward(self):
    if self.knob:
        self.submodule_0(x)
    else:
        self.knob = True
        self.submodule_1(x)