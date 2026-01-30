class DataClassVariable(ConstDictVariable):
    """             
    This is a bit of a hack to deal with
    transformers.file_utils.ModelOutput() from huggingface.
        
    ModelOutput causes trouble because it a a mix of a dataclass and a
    OrderedDict and it calls super() methods implemented in C.
    """ 
        
    # ModelOutput() excludes None, though generic datclasses don't
    include_none = False
                
    @staticmethod
    @functools.lru_cache(None)
    def _patch_once():
        from transformers.file_utils import ModelOutput
                
        for obj in ModelOutput.__dict__.values():
            if callable(obj):
                skip_code(obj.__code__)
            
    @staticmethod
    def is_matching_cls(cls):
        try:
            from transformers.file_utils import ModelOutput
    
            return issubclass(cls, ModelOutput)
        except ImportError:
            return False