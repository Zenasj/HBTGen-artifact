import torch

class Compile:
    def __init__(self, configs):

        self.configs = configs
    


        onnx_folder = os.path.join(self.configs.paths.result_path, self.configs.paths.onnx_folder)
        if not os.path.exists(onnx_folder):
            os.makedirs(onnx_folder)


        self.onnx_path = os.path.join(onnx_folder, f"{self.mode}_{self.configs.paths.onnx_path}")
   
        self.__load_model()
        self.convert()
        self.__create_session()

    def __load_model(self):
       
        self.device = 'cuda'
        self.original_model = MAXIM(num_stages=2, num_supervision_scales=1).to(self.device)
  
        with torch.no_grad():
            self.original_model.eval()
          
            self.scripted_model = torch.jit.script(self.original_model)
            logger.info("model was loaded successfully")

    def __check_onnx(self):
        model = onnx.load(self.onnx_path)
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            logger.error(f"{e}")
        else:
            logger.info("passed successsfully")

    def convert(self):
        logger.info("converting to onnx ...")
        input_names = ["input"]
        output_names = ["output"]
        self.dummy_input = torch.rand(1, 3, self.configs.compile.w_input, self.configs.compile.h_input)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        with torch.no_grad():
            self.scripted_model.eval()
            dynamic_axes = {"input": {0: 'batch', 1: 'channels', 2: 'width', 3: 'height'},
                            "output": {0: 'batch', 1: 'channels', 2: 'width', 3: 'height'}}  # adding names for better debugging

            torch_onnx.export(self.scripted_model, self.dummy_input.to("cuda"), self.onnx_path, verbose=True,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes,
                              do_constant_folding=True,
                              opset_version=13,
                              training=TrainingMode.EVAL
                              )
            self.__check_onnx()
        logger.info("______________module was converted to onnx successfully___________ ")


    def __create_session(self):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider_options = None
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_path, sess_options=session_options,
                                            provider_options=provider_options, providers=providers)