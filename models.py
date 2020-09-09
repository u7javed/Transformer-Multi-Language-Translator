import torch 
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout, device):
        super().__init__()
        
        assert hidden_size % n_heads == 0
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        
        self.fc_query = nn.Linear(hidden_size, hidden_size)
        self.fc_key = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
    
        self.dp = nn.Dropout(dropout)
        
        self.coefficient = torch.sqrt(torch.FloatTensor([self.head_size])).to(device)
        
    def forward(self, query, key, value, mask=None):
        b_size = query.shape[0]
   
        query_output = self.fc_query(query)
        key_output = self.fc_key(key)
        value_output = self.fc_value(value)
     
        query_output = query_output.view(b_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        key_output = key_output.view(b_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        value_output = value_output.view(b_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
      
        energy = torch.matmul(query_output, key_output.permute(0, 1, 3, 2)) / self.coefficient
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)    
        output = torch.matmul(self.dp(attention), value_output)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(b_size, -1, self.hidden_size)  
        output = self.fc_out(output)
        return output, attention



class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout):
        super().__init__()

        self.ff_layer = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size)
        )
        
    def forward(self, input):
        output = self.ff_layer(input)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_size,  dropout, device):
        super().__init__()
        
        self.self_atten = MultiHeadAttentionLayer(hidden_size, n_heads, dropout, device)
        self.self_atten_norm = nn.LayerNorm(hidden_size)
        self.ff_layer = FeedForwardLayer(hidden_size, ff_size, dropout)
        self.dp = nn.Dropout(dropout)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, input, input_mask):
        #self attention
        atten_result, _ = self.self_atten(input, input, input, input_mask)
        
        atten_norm = self.self_atten_norm(input + self.dp(atten_result))
        ff_result = self.ff_layer(atten_norm)
        
        output = self.ff_layer_norm(atten_norm + self.dp(ff_result))
        return output

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_heads, ff_size,dropout, device, MAX_LENGTH=100):
        super().__init__()

        self.device = device
        
        self.te = nn.Embedding(input_size, hidden_size)
        self.pe = nn.Embedding(MAX_LENGTH, hidden_size)
        
        encoding_layers = []
        for _ in range(n_layers):
            encoding_layers.append(EncoderLayer(hidden_size, n_heads, ff_size, dropout, device))
        self.encode_sequence = nn.Sequential(*encoding_layers)
        
        self.dp = nn.Dropout(dropout)
        
        self.coefficient = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)
        
    def forward(self, input, input_mask):
        b_size = input.shape[0]
        input_size = input.shape[1]
        
        pos = torch.arange(0, input_size).unsqueeze(0).repeat(b_size, 1).to(self.device)
        input = self.dp((self.te(input) * self.coefficient) + self.pe(pos))

        for layer in self.encode_sequence:
            input = layer(input, input_mask)
  
        return input

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_size, dropout, device):
        super().__init__()
        
        self.self_atten = MultiHeadAttentionLayer(hidden_size, n_heads, dropout, device)
        self.self_atten_norm = nn.LayerNorm(hidden_size)
        self.encoder_atten = MultiHeadAttentionLayer(hidden_size, n_heads, dropout, device)
        self.encoder_atten_norm = nn.LayerNorm(hidden_size)
        self.ff_layer = FeedForwardLayer(hidden_size, ff_size, dropout)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        self.dp = nn.Dropout(dropout)
        
    def forward(self, target, encoded_input, target_mask, input_mask):
        #self attention
        atten_result, _ = self.self_atten(target, target, target, target_mask)
        
        atten_norm = self.self_atten_norm(target + self.dp(atten_result))

        atten_encoded, attention = self.encoder_atten(atten_norm, encoded_input, encoded_input, input_mask)
        
        encoded_norm = self.encoder_atten_norm(atten_norm + self.dp(atten_encoded))

        ff_result = self.ff_layer(encoded_norm)

        output = self.ff_layer_norm(encoded_norm + self.dp(ff_result))

        return output, attention

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers, n_heads, ff_size, dropout, device, MAX_LENGTH=100):
        super().__init__()
        
        self.device = device
        
        self.te = nn.Embedding(output_size, hidden_size)
        self.pe = nn.Embedding(MAX_LENGTH, hidden_size)

        decoding_layers = []
        for _ in range(n_layers):
            decoding_layers.append(DecoderLayer(hidden_size, n_heads, ff_size, dropout, device))
        
        self.decode_sequence = nn.Sequential(*decoding_layers) 
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.dp = nn.Dropout(dropout)
        
        self.coefficient = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)
        
    def forward(self, target, encoded_input, target_mask, input_mask):    
        b_size = target.shape[0]
        target_size = target.shape[1]
        
        pos = torch.arange(0, target_size).unsqueeze(0).repeat(b_size, 1).to(self.device)
        target = self.dp((self.te(target) * self.coefficient) + self.pe(pos))
        for layer in self.decode_sequence:
            target, attention = layer(target, encoded_input, target_mask, input_mask)

        output = self.fc_out(target)
        return output, attention

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, device, padding_index=0):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.padding_index = padding_index
        self.device = device
        
    def make_input_mask(self, input):

        input_mask = (input != self.padding_index).unsqueeze(1).unsqueeze(2)
        return input_mask
    
    def make_target_mask(self, target):

        target_pad_mask = (target != self.padding_index).unsqueeze(1).unsqueeze(2)
        target_sub_mask = torch.tril(torch.ones((target.shape[1], target.shape[1]), device = self.device)).bool()
        target_mask = target_pad_mask & target_sub_mask
        return target_mask

    def forward(self, input, target):   
        input_mask = self.make_input_mask(input)
        target_mask = self.make_target_mask(target)

        #encoder feed through
        encoded_input = self.encoder(input, input_mask)

        #decoder feed_through
        output, attention = self.decoder(target, encoded_input, target_mask, input_mask)

        return output, attention