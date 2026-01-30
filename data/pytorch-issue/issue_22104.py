@jit.script
def forward(
    languages: Optional[Dict[int, str]]  
):
    if languages is None:
        print(languages)