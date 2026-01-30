import torch
import torch.nn as nn

class BERT_CUSTOM(nn.Module):
    
    
    def __init__(self, bert_model,id2label,num_labels):
        
        
        
        super(BERT_CUSTOM, self).__init__()
        self.bert = bert_model
        self.config=self.bert.config
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(768, num_labels)
        self.crf = CRF(num_labels, batch_first = True)
        
    
    def forward(self, input_ids, attention_mask,  labels=None, token_type_ids=None):
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = torch.stack((outputs[1][-1], outputs[1][-2], outputs[1][-3], outputs[1][-4])).mean(dim=0)
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output) # [32,256,21] logits
        
        if labels is not None:
            
            labels=labels.reshape(attention_mask.size()[0],attention_mask.size()[1])
            loss = -self.crf(log_soft(emission, 2), labels, mask=attention_mask.type(torch.uint8), reduction='mean')
            prediction = self.crf.decode(emission, mask=attention_mask.type(torch.uint8))
            return [loss, prediction]
                
        else:
            
            prediction = self.crf.decode(emission, mask=attention_mask.type(torch.uint8))
            prediction=[id2label[k] for k in prediction]
            return prediction