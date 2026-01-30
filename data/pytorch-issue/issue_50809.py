from fastai.text import *
import multifit

exp = multifit.from_pretrained("name of the model")
fa_config =  exp.pretrain_lm.tokenizer.get_fastai_config(add_open_file_processor=True)
data_lm = (TextList.from_folder(imdb_path, **fa_config)
            .filter_by_folder(include=['train', 'test', 'unsup']) 
            .split_by_rand_pct(0.1)
            .label_for_lm()           
            .databunch(bs=bs))
learn = exp.finetune_lm.get_learner(data_lm)  
# learn is a preconfigured fastai learner with a pretrained model loaded
learn.fit_one_cycle(10)