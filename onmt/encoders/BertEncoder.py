import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertTokenizer, BertModel

from onmt.utils.misc import aeq
from onmt.encoders.encoder import EncoderBase

class BertEncoder(EncoderBase):

	def __init__(self, bert_vocab):
		super(BertEncoder, self).__init__()
		#print(bert_vocab)
		#self.tokenizer = BertTokenizer.from_pretrained(bert_vocab)
		self.BertModel = BertModel.from_pretrained('bert-base-uncased')
		self.BertModel.eval()
		#self.BertModel.to('cuda')
			
		
	@classmethod
	def from_opt(cls, opt, embeddings=None):
		return cls(opt.bert_vocab)
		
	def _check_args(self, src, lengths=None, hidden=None):
		_, n_batch, _ = src.size()
		if lengths is not None:
			n_batch_, = lengths.size()
			aeq(n_batch, n_batch_)
	
	def forward(self, src, lengths=None):
	
		#print('input:')
		#print(src.shape)
		
		tokens_tensor = src[:,:,0].t()
		segments_tensors = torch.zeros_like(tokens_tensor)
				
		#tokens_tensor = tokens_tensor.to('cuda')
		#segments_tensors = segments_tensors.to('cuda')
		
		with torch.no_grad():
			encoded_layers, _ = self.BertModel(tokens_tensor, segments_tensors)
						
		final_layer = encoded_layers[-1].permute(1, 0, 2)
		memory = final_layer[1:]
		
		encoder_final = final_layer[0].unsqueeze(0)
		#print('encoder final:')
		#print(encoder_final.shape)
		
		#print('memory bank:')
		#for m in memory:
		#	print(m.shape)
		#print()
			
		return (encoder_final, encoder_final), memory, lengths