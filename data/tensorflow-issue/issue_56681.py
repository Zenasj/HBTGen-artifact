class Model(keras.Model):
    def __init__(self,cfg='/models/yolov5s.yaml',ch=3,nc=None):
        super(Model,self).__init__()
        if isinstance(cfg,dict):
            yaml = cfg
        else :
            import yaml as y
            yaml_file=Path(cfg).name
            with open(cfg) as f:
                yaml = y.load(f,Loader=y.FullLoader)