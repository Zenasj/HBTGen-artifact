import torch

from dataclasses import dataclass

# Define the dataclass
@dataclass
class MyDataClass:
    __slots__ = ["x", "y"]
    x: int
    y: str
# Create an instance of the dataclass
my_data = MyDataClass(x=2, y=3)
# Save the dataclass to a file
torch.save(my_data, "my_data.pt")
with torch.serialization.safe_globals([MyDataClass]):
    loaded_my_data = torch.load("my_data.pt", weights_only=True)
# AttributeError: 'MyDataClass' object has no attribute '__dict__'