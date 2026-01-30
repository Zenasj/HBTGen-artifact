from monai.networks.nets import SwinUNETR    
import torch                                 
                                             
if __name__ == '__main__':                   
                                             
    model = SwinUNETR(img_size=(96, 96, 96), 
                      in_channels=1,         
                      out_channels=5,        
                      feature_size=48,       
                      drop_rate=0.0,         
                      attn_drop_rate=0.0,    
                      dropout_path_rate=0.0, 
                      use_checkpoint=True,   
                      )                      
    inputs = [torch.randn([1,1,96,96,96])]
    input_names = ['input']                          
    output_names = ['output']                        
                                                     
    torch.onnx.export(                               
        model,                                       
        tuple(inputs), 'model.onnx',                 
        verbose=False,                               
        input_names=input_names,                     
        output_names=output_names,                   
        dynamic_axes=None,                           
        opset_version=11,                            
    )

from monai.networks.nets import SwinUNETR
import torch

if __name__ == '__main__':

    model = SwinUNETR(img_size=(96, 96, 96),
                      in_channels=1,
                      out_channels=5,
                      feature_size=48,
                      drop_rate=0.0,
                      attn_drop_rate=0.0,
                      dropout_path_rate=0.0,
                      use_checkpoint=True,
                      )
    inputs = [torch.randn([1,1,96,96,96])]
    input_names = ['input']
    output_names = ['output']

    torch.onnx.export(
        model,
        tuple(inputs), 'model.onnx',
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
        opset_version=17,
    )