import torch.nn as nn

import onnx
import torch
from transformers import MultitaskRobertaForIntentClassification, RobertaTokenizer

PYTORCH_MODEL_PATH = './best_Distilroberta_4_layers_AdvMultitask_with_attentions'
ONNX_MODEL_PATH = './4_layers.onnx'

pytorch_model =  MultitaskRobertaForIntentClassification.from_pretrained(PYTORCH_MODEL_PATH)
tokenizer = RobertaTokenizer.from_pretrained(PYTORCH_MODEL_PATH)
dummy_query = '<s> dummy query </s>';
tokenized_text = tokenizer.tokenize(dummy_query)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
torch.onnx.export(pytorch_model, tokens_tensor, ONNX_MODEL_PATH, opset_version=10) #, opset_version=11

onnx_model = onnx.load(ONNX_MODEL_PATH)
onnx.checker.check_model(onnx_model)
print('Model :\n\n{}'.format(onnx.helper.printable_graph(onnx_model.graph)))

class MultitaskRobertaForIntentClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"


    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_intent_labels = config.num_intent_labels
        self.num_topic_labels = config.num_topic_labels
        self.num_ability_labels = config.num_ability_labels

        self.roberta = RobertaModel(config)
        self.dp_prob = config.hidden_dropout_prob
        self.intent_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.topic_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ability_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, self.num_intent_labels)
        self.topic_classifier = nn.Linear(config.hidden_size, self.num_topic_labels)
        self.ability_classifier = nn.Linear(config.hidden_size, self.num_ability_labels)

        self.init_weights()

        if "adv_lr" in kwargs:
            print('*********************using ADV training******************************')
            self.use_adv = True
            self.adv_lr = kwargs['adv_lr']
            self.adv_init_mag = kwargs['adv_init_mag']
            self.adv_steps = kwargs['adv_steps']
        else:
            self.use_adv = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        intent_labels=None,
        topic_labels=None,
        ability_labels=None,
        dp_masks=None
    ):

        outputs, encoder_dp_masks = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            dp_masks=dp_masks[1] if dp_masks is not None else None
        )

        pooled_output = outputs[1]

        if self.dp_prob > 0 and self.training:
            if dp_masks is None:
                mask1 = torch.zeros_like(pooled_output).bernoulli_(1 - self.dp_prob) / (1 - self.dp_prob)
                # mask2 = torch.zeros_like(pooled_output).bernoulli_(1 - self.dp_prob) / (1 - self.dp_prob)
                # mask3 = torch.zeros_like(pooled_output).bernoulli_(1 - self.dp_prob) / (1 - self.dp_prob)
            else:
                mask1 = dp_masks[0]
                # mask1, mask2, mask3 = dp_masks[0]
            pooled_output = mask1 * pooled_output
            # intent_output = mask1 * pooled_output
            # topic_output = mask2 * pooled_output
            # ability_output = mask3 * pooled_output
        else:
            mask1 = None  # just to maintain the structure

        intent_logits = self.intent_classifier(pooled_output)

        topic_logits = self.topic_classifier(pooled_output)

        ability_logits = self.ability_classifier(pooled_output)

        outputs = (intent_logits, ability_logits, topic_logits) + outputs[2:]  # add hidden states and attention if they are here

        if intent_labels is not None:
            assert(ability_labels is not None and topic_labels is not None)
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1))
            topic_loss = loss_fct(topic_logits.view(-1, self.num_topic_labels), topic_labels.view(-1))
            ability_loss = loss_fct(ability_logits.view(-1, self.num_ability_labels), ability_labels.view(-1))
            loss = intent_loss + topic_loss + ability_loss
            outputs = (loss, ) + outputs

        if self.training:
            return outputs, [mask1, encoder_dp_masks] # replace mask1 with (mask1, mask2, mask3)  # (loss), logits, (hidden_states), (attentions)
        else:
            return outputs  # (loss), intent logits, ability logits, topic logits, (hidden_states), (attentions)