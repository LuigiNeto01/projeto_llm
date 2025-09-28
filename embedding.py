import torch
input_ids = torch.tensor([2,3,5,1])  # exemplo de token ids

vocab_size = 6
output_dim = 3

torch.manual_seed(123) # para garantir que os valores aleatorios sejam os mesmos em cada execucao
embeddings_layer = torch.nn.Embedding(vocab_size, output_dim) # vocab size = tamanho do vocabulario, output dim = tamanho do vetor de embedding

print(embeddings_layer.weight) # pesos iniciais aleatorios