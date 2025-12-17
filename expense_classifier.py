import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Dataset Fictício
sentences = [
    # 0: Combustível
    "abasteci 50 reais", "tanque cheio de gasolina", "gasolina está cara", "posto de gasolina", "abastecimento diesel",
    "coloquei etanol", "preço do combustivel", "abastecer o carro", "nota do posto", "gasto com combustivel",
    "fui no posto ipiranga", "gasolina aditivada", "combustivel barato", "encher o tanque", "posto shell",

    # 1: Manutenção
    "troca de óleo do motor", "pneu furou na estrada", "visita ao mecânico", "revisão do carro", "conserto do freio",
    "bateria nova", "alinhamento e balanceamento", "limpador de parabrisa", "troca de pastilha", "manutencao preventiva",
    "trocar pneu", "motor fazendo barulho", "embreagem ruim", "suspensão quebrada", "freio fazendo barulho",

    # 2: Alimentação
    "almoço no restaurante", "comprei água mineral", "lanche da tarde", "jantar após o trabalho", "cafezinho na padaria",
    "sanduíche natural", "comida por quilo", "refrigerante e salgado", "gastei com comida", "almoco self service",
    "fome na estrada", "marmita", "churrasco", "pizza", "salgado",

    # 3: Receita
    "corrida uber finalizada", "pagamento 99 pop", "dinheiro de passageiro", "recebi em dinheiro", "ganhos do dia",
    "corrida particular", "pix de cliente", "gorjeta do passageiro", "faturamento semanal", "corrida longa",
    "dinheiro extra", "pagamento via pix", "corrida dinheiro", "receita do dia", "lucro hoje"
]

# Labels correspondentes (15 exemplos para cada classe)
labels = np.array(
    [0] * 15 +
    [1] * 15 +
    [2] * 15 +
    [3] * 15
)

# 2. Pré-processamento
vocab_size = 100
max_length = 10
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# 3. Arquitetura do Modelo
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# 4. Compilação e Treinamento
# Ajustando learning_rate para garantir convergência em 50 épocas com poucos dados
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print("Iniciando treinamento (50 épocas)...")
history = model.fit(padded, labels, epochs=50, verbose=0)
print("Treinamento concluído.")

# 5. Função de Teste
def classificar_despesa(frase):
    # Categorias mapeadas
    categorias = {
        0: "Combustível",
        1: "Manutenção",
        2: "Alimentação",
        3: "Receita"
    }

    # Pré-processar a nova frase
    sequence = tokenizer.texts_to_sequences([frase])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Predição
    prediction = model.predict(padded_sequence, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    categoria_nome = categorias[class_idx]

    print(f"Frase: '{frase}'")
    print(f"Categoria Prevista: {categoria_nome} (Confiança: {confidence:.2f})\n")
    return categoria_nome, confidence

# Teste final conforme solicitado
if __name__ == "__main__":
    test_phrase = "gastei no posto Ipiranga"
    classificar_despesa(test_phrase)
