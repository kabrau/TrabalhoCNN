import numpy as np
import matplotlib.pyplot as plt
from rncvc.classifiers.neural_net import NeuralNet
from rncvc.data_utils import load_CIFAR10, save_model, load_model

import winsound

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def beep():
    Freq = 2000 # Set Frequency To 2500 Hertz
    Dur = 500 # Set Duration To 1000 ms == 1 second
    winsound.Beep(Freq,Dur)

def final_music():
    beatlength = 300
    frequency = 1 #.7

    winsound.Beep(int(262*frequency), beatlength) # C
    winsound.Beep(int(262*frequency), beatlength) # C
    winsound.Beep(int(294*frequency), beatlength) # D
    winsound.Beep(int(330*frequency), beatlength) # E

    winsound.Beep(int(262*frequency), beatlength) # C
    winsound.Beep(int(330*frequency), beatlength) # E
    winsound.Beep(int(294*frequency), 2*beatlength) # D (double length)

    winsound.Beep(int(262*frequency), beatlength) # C
    winsound.Beep(int(262*frequency), beatlength) # C
    winsound.Beep(int(294*frequency), beatlength) # D
    winsound.Beep(int(330*frequency), beatlength) # E

    winsound.Beep(int(262*frequency), 2*beatlength) # C (double length)
    winsound.Beep(int(247*frequency), 2*beatlength) # B (double length)

    winsound.Beep(int(262*frequency), beatlength) # C
    winsound.Beep(int(262*frequency), beatlength) # C
    winsound.Beep(int(294*frequency), beatlength) # D
    winsound.Beep(int(330*frequency), beatlength) # E

    winsound.Beep(int(349*frequency), beatlength) # F
    winsound.Beep(int(330*frequency), beatlength) # E
    winsound.Beep(int(294*frequency), beatlength) # D
    winsound.Beep(int(262*frequency), beatlength) # C

    winsound.Beep(int(247*frequency), beatlength) # B
    winsound.Beep(int(196*frequency), beatlength) # G
    winsound.Beep(int(220*frequency), beatlength) # A
    winsound.Beep(int(247*frequency), beatlength) # B

    winsound.Beep(int(262*frequency), 2*beatlength) # C (double length)
    winsound.Beep(int(262*frequency), 2*beatlength) # C (double length)    

def rel_error(x, y):
  """ retorna erro relativo """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def get_CIFAR10_data():
    """
    Carregando o CIFAR-10 e efetuando pre-processamento para preparar os dados
    para entrada na Rede Neural.     
    """
    # Carrega o CIFAR-10
    cifar10_dir = 'rncvc/datasets/cifar-10-batches-py'    
    X_train, y_train, X_valid, y_valid = load_CIFAR10(cifar10_dir)   

    # Normalizacao dos dados: subtracao da imagem media
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_valid -= mean_image
    
    # Imagens para linhas 
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_valid = X_valid.reshape(X_valid.shape[0], -1)    

    return X_train, y_train, X_valid, y_valid


# Utiliza a funcao acima pra carregar os dados.
X_train, y_train, X_valid, y_valid = get_CIFAR10_data()
input_size = 32 * 32 * 3
num_classes = 10
earlyStopping = 10

epochs = 200
batch_size = 200
hidden_size = 50
learning_rate=1e-3
learning_rate_decay=0.95
reg = 0.5
optimizer = "SGD"

model_name = ""
prefixo = ""
sufixo = ""

# primeira rodada
# hidden_range = [50,100,200,400,800,1600,3200,4000]
# regRange = [0.5, 0.75, 1]
# learnRange = [1e-3, 2e-3, 3e-3]

# segunda rodada
# vou tentar carregar o treinamento de: model_0.570300_800_0.500000_0.003000_95.pickle
# e fazer o seguinte range:
#hidden_range = [800]
#regRange = np.random.uniform(0.4, 0.6, 5)
#learnRange = np.random.uniform(1e-3,1e-5, 5)
#model_name = "model_0.570300_800_0.500000_0.003000_95.pickle"
#sufixo = "R2"

# terceira rodada
# vou tentar carregar o treinamento de: model_0.575500_3200_0.750000_0.002000_135.pickle
# e fazer o seguinte range:
hidden_range = [3200]
regRange = np.random.uniform(0.5, 0.75, 5)
learnRange = np.random.uniform(1e-3,1e-4, 5)
model_name = "model_0.575500_3200_0.750000_0.002000_135.pickle"
sufixo = "R3"

# quarta rodada
# vou tentar carregar o treinamento de: R3-model_0.580100_3200_0.678687_0.000673_66.pickle
# e fazer o seguinte range:
hidden_range = [3200]
regRange = np.random.uniform(0.6, 0.8, 5)
learnRange = np.random.uniform(8e-4,1e-5, 5)
model_name = "R3-model_0.580100_3200_0.678687_0.000673_66.pickle"
sufixo = "R4"

# quinta rodada
# vou tentar carregar o treinamento de: model_0.580800_3200_0.768194_0.000031_21-R4.pickle
# e fazer o seguinte range:
hidden_range = [3200]
regRange = np.random.uniform(0.65, 0.85, 5)
learnRange = np.random.uniform(8e-4,1e-6, 5)
model_name = "model_0.580800_3200_0.768194_0.000031_21-R4.pickle"
sufixo = "R5"


# sexta rodada, mudando otimizador
# vou tentar carregar o treinamento de: model_0.580800_3200_0.768194_0.000031_21-R4.pickle
# e fazer o seguinte range:
hidden_range = [3200]
regRange = np.random.uniform(0.65, 0.85, 5)
learnRange = np.random.uniform(8e-4,1e-6, 5)
model_name = "model_0.580800_3200_0.768194_0.000031_21-R4.pickle"
sufixo = "R6-AdaGrad"
optimizer = "AdaGrad"
learning_rate_decay=0.95

# setima rodada, mudando otimizador para ADAM
# vou tentar carregar o treinamento de: model_0.581500_3200_0.689206_0.000664_75-R6-AdaGrad.pickle
# e fazer o seguinte range:
hidden_range = [3200]
regRange = np.random.uniform(0.5, 0.9, 5)
learnRange = np.random.uniform(1e-3,1e-6, 5)
model_name = "model_0.581500_3200_0.689206_0.000664_75-R6-AdaGrad.pickle"
sufixo = "R7-Adam"
optimizer = "Adam"
learning_rate_decay=0.95

if (model_name):
    model = load_model(model_name)

for hidden_size in hidden_range:
    for learning_rate in learnRange:
        for reg in regRange: 

            print "Inicio Hidden:",hidden_size, ", LearnRate:", learning_rate, ", Reg:",reg

            net = NeuralNet(input_size, hidden_size, num_classes)

            if (model_name):
                print "lendo ", model_name
                net.params = model


            # Treina a rede
            stats = net.trainMarcelo(X_train, y_train, X_valid, y_valid,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate, 
                                    learning_rate_decay=learning_rate_decay,
                                    reg=reg, 
                                    verbose=True, 
                                    earlyStopping=earlyStopping)

            # Efetua predicao no conjunto de validacao
            val_acc = (net.predict(X_valid) == y_valid).mean()
            print 'Neuronios: %d , acuracia de validacao: %f, epoca %d' % (hidden_size, val_acc, net.params['Epoch'])

            # Salva o modelo da rede treinada
            model_path = '%smodel_%f_%d_%f_%f_%d-%s.pickle' % (prefixo,val_acc, hidden_size, reg, learning_rate, net.params['Epoch'],sufixo)
            save_model(model_path, net.params)
            beep()

final_music()
