py
import torch

from torchtext.models import RobertaEncoderConf, RobertaModel, RobertaClassificationHead

roberta_encoder_conf = RobertaEncoderConf(vocab_size=250002)
classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
classifier = RobertaModel(encoder_conf=roberta_encoder_conf, head=classifier_head)

input = torch.randint(250002, (7,13))

classifier = torch.compile(classifier)

print(classifier(input))

typing._GenericAlias

typing.Mapping

torch.storage