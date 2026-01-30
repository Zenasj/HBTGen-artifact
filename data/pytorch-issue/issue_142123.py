import stanza

pos_pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=True, device='xpu')
sentence = "This caused particular problems in cases involving abused women who killed their abusive husbands as case law demonstrates that they often involved a lapse in time between the provocation and the killing. Subsequent cases sought to extend the defence to battered women by accepting the notion of cumulative provocation and acknowledging that provocation might still be operating on the defendant’s mind, even after a lapse of time. Irrespective of this, there were still concerns that the requirement of a sudden and temporary loss of control made it too difficult for battered women to rely on this defence."
pos_pipeline(sentence)