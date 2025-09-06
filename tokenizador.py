'''
tokenizador eh basicamente quebrar palavras em pedaços menores, chamados tokens.

1. vamos preparar a entrada de textos para treinar as llms
como prepara?

primeiro passo: dividir o texto em letras invididuais e  subword tokens

segundo passo: coverter tokens em tokens ids

terceiro passo: encode tokens ids em representação de vetores

exemplo

this is an example -> texto tokenizado [this | is | an | example] -> token ids [ 4013 | 201 | 302 | 1134] -> token embeddings
'''


# passo 1 criar tokens
with open('dataset\Dataset1.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read().replace('\n', ' ')

print(f"Total de caracteres: {len(raw_text)}")
print(f"{raw_text[:100]}...")

# dividir todo o texto em palavras
import re

preprocessed = re.split(r'[*,.:;?!\-()"\'\s]|--', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(f"Total de palavras: {len(preprocessed)}")

# passo 2 criar token ids
''' 
agora vamos criar um vocabulario unico
um vocabulario eh uma lista de todas as palavras em um corpus de texto
vamos criala de modo alfabetico

tokens unicos = id unicos
'''

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(f"Tamanho do vocabulario: {vocab_size}")

# vamos criar o vocabulario
# cada elemento tem seu id
# encoder seria buscar o id pelo token
# decoder seria o inverso do encoder, vamos buscar um token pelo id
vocab = {token:integer for integer, token in enumerate(all_words)}
#salvar vocabulario em um arquivo
import json
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)
    
class tokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        # s = token
        # i = token id
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'[*,.:;?!\-()"\'\s]|--', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = ' '.join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([.,!?"()\'])', r'\1', text)
        return text

# teste de encode do tokenizer
tokenizer = tokenizerV1(vocab)
text = 'Amadurecido pela leitura atenta dos teóricos da linguagem'
ids = tokenizer.encode(text)
print(f"Token ids: {ids}")
# Token ids: [1191, 21335, 18304, 7977, 12964, 26278, 11456, 18532]

# teste de decode do tokenizer
print(f"Decoded text: {tokenizer.decode(ids)}")

'''
caso nao tenha o token no vocabulario, o encode vai gerar um erro
para resolver isso, podemos adicionar um token especial <unk> (unknown)
e vamos adicionar um endoftext token <eot> (end of text) para indicar o fim do texto no vocabulario ele basicamante vai ser o ultimo token do vocabulario para indicar o fim do texto
'''

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(['<|endoftext|>', '<|unk|>'])

vocab = {token:integer for integer, token in enumerate(all_tokens)}
len(vocab)

class tokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        # s = token
        # i = token id
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'[*,.:;?!\-()"\'\s]|--', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else '<|unk|>' for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = ' '.join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([.,!?"()\'])', r'\1', text)
        return text

# teste de encode do tokenizer
tokenizer = tokenizerV2(vocab)
text = 'Olá meu amigo, como você está?'
ids = tokenizer.encode(text)
print(f"Token ids: {ids}")

# teste de decode do tokenizer
print(f"Decoded text: {tokenizer.decode(ids)}")

'''
conteudo adicional:

[BOS] = begin of sentence: indica o inicio de uma sentença, que significa que o modelo deve começar a gerar texto a partir desse ponto.
[SEP] = separator: usado para separar diferentes partes de um texto, como perguntas e respostas em um diálogo.
[PAD] = padding: usado para preencher sequências de texto para que todas tenham o mesmo comprimento em um lote (batch) de dados.
[CLS] = classification: usado em tarefas de classificação de texto, onde o modelo precisa classificar uma sentença ou um par de sentenças.
[EOS] = end of sentence: indica o fim de uma sentença, semelhante ao <|endoftext|> que usamos.
'''

'''
O GPT utiliza um tokenizador baseado em Byte Pair Encoding (BPE), que é uma técnica de tokenização subword. O BPE divide palavras em subunidades menores, permitindo que o modelo lide melhor com palavras raras ou desconhecidas. Além disso, o tokenizador do GPT inclui tokens especiais como <|endoftext|> para indicar o fim do texto, o que é crucial para o funcionamento do modelo.
'''

# -------------------------------- = -------------------------------

'''
Byte Pair Encoding 

Algoritimos de tokenização
- word based
- subword based
- character based

# word based
eh basicamente o que fiz a cima, o problema eh, o que fazer com palavras desconhecidas?

# character based
cada caractere eh um token, o problema eh que a sequencia de tokens fica muito grande mas resolve o problema de palavras desconhecidas
- perde o significado das palavras
- a sequencia de tokens fica muito grande

# subword based
combina o melhor dos dois mundos
- nao divide palavras usadas com frequencia em caracteres melhores
- divide palavras raras em caracteres ou subwords menores
exemplo:
boys -> ['boy', 's']

ele ajuda o modelo a aprender que diferentes palavras vem da mesma raiz -> token, tokens e tokenization teriam significado similar

ele ajuda o modelo a entender que tokenization e modernization tem raizes diferentes porem compartilham o sufixo 'ization' e são usadas nas mesmas situacoes sintaticas

BPE -> Byte Pair Encoding
Algoritmo BPE (1994) -> um algoritmo de compressão de dados que pode ser adaptado para tokenização de texto. 
Ele funciona substituindo pares de bytes mais frequentes por um byte único que não aparece no texto original. 
No contexto da tokenização, o BPE começa com um vocabulário inicial de caracteres individuais e itera para combinar os pares de tokens mais frequentes em novos tokens, criando assim um vocabulário de subwords.

exemplo de algoritmo BPE:
original data: "aaabdaaabac"
1. contar pares de bytes:
   aa: 4
   ab: 2
   ba: 2
   ad: 1
   da: 1
   ac: 1
2. substituir o par mais frequente (aa) por um novo token (X):
   XabdXabac
3. repetir o processo:
   contar pares:
    ab: 2
    ba: 2
    Xb: 2
    ad: 1
    da: 1
    ac: 1
    substituir o par mais frequente (ab) por um novo token (Y):
    XYdXYac

'''

import tiktoken
import importlib

tokenizer = tiktoken.get_encoding("gpt2")

text = ('hello, do you like tea? <|endoftext|> In the sunlit terraces' 'of sumeunknowPlace')
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)