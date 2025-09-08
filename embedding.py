'''
para criar token embeddings basicamente vamos criar um vetor numerico para cada token
cat -> id 3
o vetor dele sera algo assim [0,0,0,1,0,....,0,0,0,0,0] (tamanho do vetor = tamanho do vocabulario)

o problema de usar numeros aleatorios eh que o modelo nao vai entender a relacao entre os tokens
o one hot encoding nao captura relacoes semanticas entre palavras
dog -> [0,0,0,1,0,....,0,0,0,0,0]
puppy -> [0,0,0,0,0,....,0,1,0,0,0]

como colocar significado?
palavras com semantica similar devem ter vetores similares

                dog     cat    apple   banana
has_a_tail  |   23      31      1       2
is_eatable  |   2       3       22      38
has_4_legs  |   19      21      0       0
makes_sound |   12      18      0.5     0.2
is_a_pet    |   35      31      5       7

nesse caso conseguimos ver que a banan e a maça estao proximos nos resultados, assim como maça e banana

entao podemos treinar uma rede neural para criar o vector embeding
'''

import gensim.downloader as api
model = api.load('word2vec-google-news-300') # fazendo download do modelo pre treinado

word_vector=model

# vamos ver como fica a representacao vetorial de algumas palavras
print(word_vector['cat'])# representacao vetorial de cat

# agora esse seria o tamanho do vetor
print(word_vector['cat'].shape)

# king + woman - man = ?
# exemplo de uso de most_similar
print(word_vector.most_similar(positive=['king', 'woman'], negative=['man']))
# output = [('queen', 0.7118193507194519), ('monarch', 0.6189674139022827), ('princess', 0.5902431011199951), ('crown_prince', 0.5499460697174072), ('prince', 0.5377321839332581), ('kings', 0.5236844420433044), ('Queen_Consort', 0.5235945582389832), ('queens', 0.5181134343147278), ('sultan', 0.5098593831062317), ('monarchy', 0.5087411999702454)]


# conseguimos ver as similaridades entre palavras
print(word_vector.similarity('cat', 'dog')) # similaridade entre cat e dog
# output = 0.76094574

# para criar um modelo do zero
# passo 1 inicializar embeddings com pesos com valores aleatorios
# passo 2 essa inicializaçao serve como ponto de partida para o aprendizado da llm
# passo 3 o peso do embedding sera ajustado durante o treinamento da llm

# create tokens embeddings
import torch
input_ids = torch.tensor([2,3,5,1])  # exemplo de token ids

vocab_size = 6
output_dim = 3

torch.manual_seed(123) # para garantir que os valores aleatorios sejam os mesmos em cada execucao
embeddings_layer = torch.nn.Embedding(vocab_size, output_dim) # vocab size = tamanho do vocabulario, output dim = tamanho do vetor de embedding

print(embeddings_layer.weight) # pesos iniciais aleatorios
'''
output:
Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)
'''

# apos inicializados vamos aplicar a ele o token id para obter o vetor de embedding
# ele busca uma linha de acordo com o token id
print(embeddings_layer(torch.tensor([3])))
'''
output:
tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
'''

print(embeddings_layer(input_ids)) # aplicando o vetor de token ids para obter os vetores de embedding
'''
output:
tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
'''
# embeddings sao basicamente a lookup operation que retorna uma linha da embeddings layer wheight matrix usando o token id como indice
'''
na hora de criar usamos a linha:
embeddings_layer = torch.nn.Embedding(vocab_size, output_dim) 
basicamente o vocab_size define o numero de linhas e o output_dim define o numero de colunas

basicamente em uma rede neural com os inputs
[2,3,1]
ele seria algo assim
[0,0,1,0] = 2
[0,0,0,1] = 3
[0,1,0,0] = 1

basicamente sendo 4 pontos (os 0 e 1) conectando em 5 dimensoes (5 colunas), tendo cada coluna 4 linhas conectadas nelas
[0,0,1,0] -> [w11,w12,w13,w14,w15]
[0,0,0,1] -> [w21,w22,w23,w24,w25]
[0,1,0,0] -> [w31,w32,w33,w34,w35]
onde w11,w12...w35 sao os pesos que serao ajustados durante o treinamento

W^T = [
[w11,w12,w13,w14,w15]
[w21,w22,w23,w24,w25]
[w31,w32,w33,w34,w35]
]
sendo o output = X.W^T

o que esta fazendo realmente:
eh exatamente o que a camada linear de rede neural faz, ela recebe inputs. 
vamos dizer que temos 3 tokens ids que vao ser convertidos em one hot representation, eles são alimentados em uma rede neural
com 5 neuronios, pq 5? pq o output_dim = 5, ou seja, cada token id sera convertido em um vetor de 5 dimensoes
em seguida temos uma camada linear cuja a saida eh X.W^T ( x em w transposto ) que dara a mesma saida que a embeddings_layer

ambos a embedding layer e neural network os metodos dao o mesmo resultado, porem a camada de embeddings eh mais computacionalmente eficiente
pq a camada linear da rede neural tem mais multiplicacoes desnecessarias com zero
'''