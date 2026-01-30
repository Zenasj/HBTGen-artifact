t_word_attention= self.word_attention(packed_sentences.data,
                                                     packed_words_per_sentence.data)  
t_word_attention= self.dropout(t_word_attention)
input_packed_sentences = PackedSequence(data=t_word_attention,
                                   batch_sizes=packed_sentences.batch_sizes,
                                    sorted_indices=packed_sentences.sorted_indices,
                                    unsorted_indices=packed_sentences.unsorted_indices)
out, _ = self.sentence_rnn(input_packed_sentences)