import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

class trainGenSeq_short_YesNo(tf.keras.utils.Sequence, ):
    def __init__(self, batchSize, sentenceLength):
        self.batchSize = batchSize
        self.trainFiles = os.listdir('D:/Python/Datasets/v1.0/train/')
        self.trainingSamples = 307372 * 2
        self.sentenceLength = sentenceLength
        
        #Load Vocab
        slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        save_path = "bert_base_uncased/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        slow_tokenizer.save_pretrained(save_path)
        self.tokenizer = BertTokenizer('vocab.txt', lowercase = True)
        self.vocabSize = len(self.tokenizer.vocab)
        
    
    def __len__(self):
        return int(self.trainingSamples // self.batchSize)
    
    def getLen(self):
        return int(self.trainingSamples // self.batchSize)
    
    def attentionMasks(self,input_dims):
        return [int(id > 0) for id in input_dims]
        
    def inputDims(self, dims):
        return pad_sequences([dims], maxlen = self.sentenceLength, dtype="long", value=0, truncating="post", padding="post")[0]
    
    def encode_sentence(self, sentence):
        sentence = sent_tokenize(sentence)
        ans = []
        for i in range(len(sentence)):
            encode_sent = self.tokenizer.encode(sentence[i],add_special_tokens = True)
            ans += encode_sent

        ans = pad_sequences([ans], maxlen = self.sentenceLength, dtype = "long", value = 0, truncating = "post", padding = "post")
        return ans[0]
    
    def __getitem__(self, _):
        documentStack = np.array([])
        questionStack = np.array([])
        answerStack = np.array([])
        
        document_AttStack = np.array([])
        question_AttStack = np.array([])

        document_SegStack = np.array([])
        question_SegStack = np.array([])

        First = True
        
        for file in self.trainFiles:
            for line in open('D:/Python/Datasets/v1.0/train/' + file):
                file = json.loads(line)
                #annotations
                if file.get('annotations')[0].get('short_answers'):
                    s_Start = file.get('annotations')[0].get('short_answers')[0].get('start_token')
                    s_End = file.get('annotations')[0].get('short_answers')[0].get('end_token')
                    l_Start = file.get('annotations')[0].get('long_answer').get('start_token')
                    l_End = file.get('annotations')[0].get('long_answer').get('end_token')

                    #Question and Title
                    question = file.get('question_text')

                    #document
                    document = []
                    for indexs in file.get('document_tokens')[l_Start:l_End]:
                        if indexs.get('html_token') == False:
                            document.append(indexs.get('token'))
                    
                    #Fake Document OR No document
                    fake = []
                    randomNumber = random.randint(7500, 9000)
                    front = random.choice([True, False])
                    
                    if front:
                        try:
                            for indexs in range(max(0, l_Start - randomNumber), min(len(file.get('document_tokens')),l_End - randomNumber)):
                                if file.get('document_tokens')[indexs].get('html_token') == False:
                                    fake.append(file.get('document_tokens')[indexs].get('token'))
                                else:
                                    indexs -= 1
                        except:
                            for indexs in range(max(0, l_Start + randomNumber), min(len(file.get('document_tokens')),l_End + randomNumber)):
                                if file.get('document_tokens')[indexs].get('html_token') == False:
                                    fake.append(file.get('document_tokens')[indexs].get('token'))
                                else:
                                    indexs -= 1
                    else:
                        try:
                            for indexs in range(max(0, l_Start + randomNumber), min(len(file.get('document_tokens')),l_End + randomNumber)):
                                if file.get('document_tokens')[indexs].get('html_token') == False:
                                    fake.append(file.get('document_tokens')[indexs].get('token'))
                                else:
                                    indexs -= 1
                        except:
                            for indexs in range(max(0, l_Start - randomNumber), min(len(file.get('document_tokens')),l_End - randomNumber)):
                                if file.get('document_tokens')[indexs].get('html_token') == False:
                                    fake.append(file.get('document_tokens')[indexs].get('token'))
                                else:
                                    indexs -= 1
                    
                    document = ' '.join(document)
                    fake = ' '.join(document)

                    document = self.encode_sentence(document)
                    fake = self.encode_sentence(fake)
                    question = self.encode_sentence(question)
                    
                    fake_AttentionMask = self.attentionMasks(fake)
                    document_AttentionMask = self.attentionMasks(document)
                    question_AttentionMask = self.attentionMasks(question)
                    
                    fake_SegID = [0 for _ in range(len(fake))]
                    document_SegID = [0 for _ in range(len(document))]
                    question_SegID = [0 for _ in range(len(question))]

                    if First:
                        #Document
                        documentStack = np.array([document])
                        documentStack = np.append(documentStack, np.array([fake]), axis = 0)
                        document_AttStack = np.array([document_AttentionMask])
                        document_AttStack = np.append(document_AttStack, np.array([fake_AttentionMask]), axis = 0)
                        document_SegStack = np.array([document_SegID])
                        document_SegStack = np.append(document_SegStack, np.array([fake_AttentionMask]), axis = 0)
                        
                        #Add Question Again
                        questionStack = np.array([question])
                        questionStack = np.append(questionStack, np.array([question]), axis = 0)
                        
                        question_AttStack = np.array([question_AttentionMask])
                        question_AttStack = np.append(question_AttStack, np.array([question_AttentionMask]), axis = 0)
                        question_SegStack = np.array([question_SegID])
                        question_SegStack = np.append(question_SegStack, np.array([question_SegID]), axis = 0)
                        
                        #Add Answer
                        answerStack = np.array([np.array([1,0])])
                        answerStack = np.append(answerStack, np.array([np.array([0,1])]), axis = 0)
                        
                        First = False
                    else:
                        documentStack = np.append(documentStack, np.array([document]), axis = 0)
                        documentStack = np.append(documentStack, np.array([fake]), axis = 0)
                        questionStack = np.append(questionStack, np.array([question]), axis = 0)
                        questionStack = np.append(questionStack, np.array([question]), axis = 0)
                        answerStack = np.append(answerStack, np.array([np.array([1,0])]), axis = 0)
                        answerStack = np.append(answerStack, np.array([np.array([0,1])]), axis = 0)
                        
                        #Attention Mask
                        document_AttStack = np.append(document_AttStack, np.array([document_AttentionMask]), axis = 0)
                        document_AttStack = np.append(document_AttStack, np.array([fake_AttentionMask]), axis = 0)
                        question_AttStack = np.append(question_AttStack, np.array([question_AttentionMask]), axis = 0)
                        question_AttStack = np.append(question_AttStack, np.array([question_AttentionMask]), axis = 0)
                        
                        #SegmentIDs
                        document_SegStack = np.append(document_SegStack, np.array([document_AttentionMask]), axis = 0)
                        document_SegStack = np.append(document_SegStack, np.array([fake_AttentionMask]), axis = 0)
                        question_SegStack = np.append(question_SegStack, np.array([question_SegID]), axis = 0)
                        question_SegStack = np.append(question_SegStack, np.array([question_SegID]), axis = 0)
                
                if documentStack.shape[0] == self.batchSize:
                    documentStack = np.reshape(documentStack, (documentStack.shape[0], 1, documentStack.shape[1]))
                    questionStack = np.reshape(questionStack, (questionStack.shape[0], 1, questionStack.shape[1]))
                    answerStack = np.reshape(answerStack, (answerStack.shape[0], 1, answerStack.shape[1]))
                    First = True

                    #print(type(documentStack), type(questionStack), type(answerStack))
                    return [np.squeeze(documentStack), np.squeeze(document_AttStack), np.squeeze(document_SegStack), 
                            np.squeeze(questionStack), np.squeeze(question_AttStack), np.squeeze(question_SegStack)], np.squeeze(answerStack)
                    
                    documentStack = None
                    titleStack = None
                    questionStack = None
                    answerStack = None

trainGen = trainGenSeq_short_YesNo(BatchSize, SeqLength)

model = createModel_YesNo(trainGen.vocabSize, BatchSize, SeqLength)
model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
model.fit(trainGen, epochs = 10, steps_per_epoch = trainGen.getLen(),verbose = 1)