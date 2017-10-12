import numpy as np
from sys import stdout


class NeuralNet(object):
  """
  Uma Rede Neural totalmente conectada de duas camadas. 
  
  Dimensoes: 
  N: Entrada da rede
  H: Numero de neuronios na camada escondida 
  C: Numero de classes
  
  Treinamento ocorre com a funcao de custo entropia cruzada + softmax.
  Utilize o ReLU como ativacao da primeira camada oculta.  

  Em resumo a arquitetura da rede eh: 

  entrada - camada totalmente conectada - ReLU - camada totalmente conectada - softmax

  As saidas da segunda camada sao as predicoes para as classes. 
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4, dropout=0.0):
    """
    
    Inicializacao do modelo. Os pesos iniciais sao pequenos valores aleatorios.  
    Valores de bias sao inicializados com zero. 
    Pesos e bias sao armazenados na variavel self.params, 
    que e um dicionario e tem as seguintes chaves:    

    W1: Pesos da primeira camada; shape (D, H)
    b1: Biases da primeira camada; has shape (H,)
    W2: Pesos da segunda camada; shape (H, C)
    b2: Biases da segunda camada; shape (C,)

    Inputs:
    - input_size: Dimensao D dos dados de entrada.
    - hidden_size: Numero de neuronios H na camada oculta.
    - output_size: Numero de classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0, dropout=0.0):
    """
    Calcula a funcao de custo e os gradientes.

    Entradas:
    - X: instancias de entrada (N, D). Cada X[i] e uma instancia de treinamento.
    - y: classes das instancias de treinamento. y[i] e a classe de X[i],
         y[i] e um valor inteiro onde 0 <= y[i] < C. 
         Este parametro e opcional: se nao for passado, serao retornados apenas os valores de predicao. 
         Passe o parametro se quiser retornar o valor da funcao de custo e os gradientes.      
    - reg: regularizacao L2.

    Retorna:
    Se y e None: retorna matriz de predicoes de shape (N, C), onde scores[i, c] e
    a predicao da classe c relativa a entrada X[i].

    Se y nao for None, retorne uma tupla com:
    - loss: valor da funcao de custo para este batch de treinamento
      samples.
    - grads: dicionario contendo os gradientes relativos a cada camada de pesos
      com respeito a funcao de custo; assume as mesmas chaves que self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Calcula a etapa forward
    scores = None
    #############################################################################
    # TODO: Implemente a etapa forward, calculando as predicoes das entradas.   #
    # Armazene o resultado na variavel scores cujo shape deve ser (N, C).       #
    #############################################################################
    hidden_layer = np.maximum(0,np.dot(X, W1) + b1) # calcula hidden layer com ReLU
    
    if dropout>0:
      D1 = (np.random.random_sample(hidden_layer.shape) < dropout) / dropout
      hidden_layer *= D1


    scores = np.dot(hidden_layer, W2) + b2          # saida
    
    #############################################################################
    #                              FIM DO SEU CODIGO                            #
    #############################################################################
    
    # Sem passar as classes por parametros retorna
    if y is None:
      return scores

    # Calcula o custo
    loss = None
    #############################################################################
    # TODO: Implemente a etapa forward e calcule o custo. Armazene o resultado  #
    # na variavel loss (escalar). Use a funcao de custo do Softmax.             #
    #############################################################################

    # SOFTMAX
    exp_scores = np.exp(scores) # para calcular apenas uma vez
    softmax_score = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
    loss = -np.log(softmax_score[range(N), y])
    loss = np.sum(loss)/N

    # Regularizacao
    loss += 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))

    #############################################################################
    #                              FIM DO SEU CODIGO                            #
    #############################################################################

    # Etapa de backward : Calcular os gradientes
    grads = {}
    #############################################################################
    # TODO: Calcule os gradientes dos pesos e dos biases. Armazene os           #
    # resultados no dicionario grads. Por exemplo, grads['W1'] deve armazenar   #
    # os gradientes relativos a W1, sendo uma matriz do mesmo tamanho de W1.    #
    #############################################################################
    
    # derivada, se entendi bem
    score_ajustado = softmax_score  # so para trocar o nome
    score_ajustado[range(N),y] -= 1 # diminuir o valor do score de cada targets
    score_ajustado /= N             # normaliza
    
    # pesos camada final, W2 e b2
    grads['W2'] = np.dot(hidden_layer.T, score_ajustado)
    grads['b2'] = np.sum(score_ajustado, axis=0)

    # Hidden Layer
    hidden_ajustado = np.dot(score_ajustado, W2.T)
    hidden_ajustado[hidden_layer <= 0] = 0 # ReLU do backpropragation
    
    # pesos da Hidden Layer
    grads['W1'] = np.dot(X.T, hidden_ajustado)
    grads['b1'] = np.sum(hidden_ajustado, axis=0)

    # Regularizacao
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1

    #############################################################################
    #                              FIM DO SEU CODIGO                            #
    #############################################################################

    return loss, grads

  def trainMarcelo(self, X, y, X_val, y_val,
            epochs=100,
            batch_size=200, 
            learning_rate=1e-3, 
            learning_rate_decay=0.95,
            reg=1e-5, 
            verbose=False, 
            earlyStopping=0, 
            optimizer="SGD", 
            dropout=0.0):
    """
    Treine uma rede neural usando SGD (Stochastic Gradient Descent)

    Inputs:
    - X: numpy array (N, D) com dados de treinamento. 
    - y: numpy array (N,) com as classes. Onde y[i] = c significa que
      X[i] tem a classe c, onde 0 <= c < C.
    - X_val: numpy array (N_val, D) com dados de validacao.
    - y_val: numpy array (N_val,) com as classes da validacao. 
    - learning_rate: taxa de aprendizado (escalar). 
    - learning_rate_decay: reducao da taxa de aprendizado por epoca. 
    - reg: parametro para controlar a forca da regularizacao.
    - epochs: numero de epocas.
    - batch_size: numero de instancias em cada batch.
    - verbose: boolean; se verdadeiro imprime informacoes durante treinamento.
    """
    num_train = X.shape[0]  
    batchs_per_epoch = max(num_train / batch_size, 1)
    num_iters = batchs_per_epoch * epochs

    # Use SGD para otimizar os parametros em self.model
    loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []   

    # Marcelo - earlyStopping
    best_loss = 99999999
    best_loss_train = 99999999
    best_acc = 0
    epocas_para_best_loss = 0

    pesos = ["W1","W2","b1","b2"]

    # AdaGrad
    AdaGradCache = {}

    # Adam
    AdamM = {}
    AdamV = {}
    AdamBeta1 = 0.98
    AdamBeta2 = 0.98
    
    for peso in pesos:
      AdaGradCache[peso] = 0
      AdamM[peso] = 0
      AdamV[peso] = 0



    for epoca in xrange(epochs):

      batch_indexes = np.arange(batchs_per_epoch)
      np.random.shuffle(batch_indexes)

      indices_random = np.arange(num_train)
      np.random.shuffle(indices_random)

      qtd_it = 0
      for it in batch_indexes:
        X_batch = None
        y_batch = None

        #########################################################################
        # TODO: Carregue um minibatch de instancias em X_batch e y_batch.       #      
        #########################################################################

        part_indice = np.split(indices_random,batchs_per_epoch)[it]
        X_batch = X[part_indice, :]
        y_batch = y[part_indice]

        qtd_it += X_batch.shape[0]

        #########################################################################
        #                             FIM DO SEU CODIGO                         #
        #########################################################################

        # Calcule a funcao de custo e os gradientes usando o minibatch atual
        loss, grads = self.loss(X_batch, y=y_batch, reg=reg, dropout=dropout)
        loss_history.append(loss)

        #########################################################################
        # TODO: use os gradientes no dicionario grads para atualizar os         #
        # parametros da rede (armazenados no dicionario self.params)            #
        # usando gradiente descendente estocastico. 
        #########################################################################
        
        if (optimizer == "AdaGrad"):

          for peso in pesos: 
            AdaGradCache[peso] += grads[peso] ** 2
            self.params[peso] += -learning_rate * grads[peso] / (np.sqrt(AdaGradCache[peso]) + 1e-7)

        elif (optimizer == "Adam"): 

          for peso in pesos:
            AdamM[peso] = AdamBeta1 * AdamM[peso] + (1 + AdamBeta1) * grads[peso]
            AdamV[peso] = AdamBeta2 * AdamV[peso] + (1 + AdamBeta1) * (grads[peso] ** 2.0)
            mb = AdamM[peso] / ( 1 - AdamBeta1 ** epoca)
            vb = AdamV[peso] / ( 1 - AdamBeta2 ** epoca)
            self.params[peso] += -learning_rate * mb / (np.sqrt(vb) + 1e-7)


        else: #SGD

          for peso in pesos:
            self.params[peso] += -learning_rate*grads[peso]

        #########################################################################
        #                             FIM DO SEU CODIGO                         #
        #########################################################################

        stdout.write('Epoch %d -> %d/%d (%d%%) loss %f %s\r' % (epoca, qtd_it, num_train, qtd_it*100/num_train, loss, optimizer))
        stdout.flush()

        # Every epoch, check train and val accuracy and decay learning rate.
        if qtd_it == num_train:
          # Check accuracy
          train_acc = (self.predict(X_batch) == y_batch).mean()
          val_acc = (self.predict(X_val) == y_val).mean()
          val_loss, _ = self.loss(X_val, y=y_val, reg=reg)

          train_acc_history.append(train_acc)
          val_acc_history.append(val_acc)
          val_loss_history.append(val_loss)

          # Marcelo - earlyStopping
          epocas_para_best_loss += 1
          best = ''
          if val_loss < best_loss:
            best += ' LossVal'
            best_loss = val_loss  
            epocas_para_best_loss = 0

          if loss < best_loss_train:
            best += ' LossTrain'
            best_loss_train = loss  
            epocas_para_best_loss = 0

          if val_acc > best_acc:
            best += ' AccVal'
            best_acc = val_acc
            epocas_para_best_loss = 0

          print 'Epoch %d -> loss %f acc %f val_loss %f val_acc %f %s' % (epoca, loss, train_acc, val_loss, val_acc, best)

          #=============================================================================
          # Ajusta atualizacao dos pesos do MarceloSGD - Decai apenas se nao mudar o loss e acc da validacao
          if (best == ''):
            # Decay learning rate
            #if (optimizer == "SGD"):  
              learning_rate *= learning_rate_decay
          else:
            self.params['Epoch'] = epoca
            self.params['BestW1'] = self.params['W1'].copy()
            self.params['Bestb1'] = self.params['b1'].copy()
            self.params['BestW2'] = self.params['W2'].copy()
            self.params['Bestb2'] = self.params['b2'].copy()


          # Marcelo - earlyStopping
          if earlyStopping > 0 and epocas_para_best_loss == earlyStopping:
            break
          
      if earlyStopping > 0 and epocas_para_best_loss == earlyStopping:
         break


    self.params['W1'] = self.params['BestW1'].copy()
    self.params['b1'] = self.params['Bestb1'].copy()
    self.params['W2'] = self.params['BestW2'].copy()
    self.params['b2'] = self.params['Bestb2'].copy()
    self.params['BestW1'] = []
    self.params['Bestb1'] = []
    self.params['BestW2'] = []
    self.params['Bestb2'] = []

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }


  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Treine uma rede neural usando SGD (Stochastic Gradient Descent)

    Inputs:
    - X: numpy array (N, D) com dados de treinamento. 
    - y: numpy array (N,) com as classes. Onde y[i] = c significa que
      X[i] tem a classe c, onde 0 <= c < C.
    - X_val: numpy array (N_val, D) com dados de validacao.
    - y_val: numpy array (N_val,) com as classes da validacao. 
    - learning_rate: taxa de aprendizado (escalar). 
    - learning_rate_decay: reducao da taxa de aprendizado por epoca. 
    - reg: parametro para controlar a forca da regularizacao.
    - num_iters: numero de iteracoes.
    - batch_size: numero de instancias em cada batch.
    - verbose: boolean; se verdadeiro imprime informacoes durante treinamento.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)    

    # Use SGD para otimizar os parametros em self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Carregue um minibatch de instancias em X_batch e y_batch.       #      
      #########################################################################
   
      # dos {num_train} samples possiveis, escolhe {batch_size} indices randomicamente
      indices_random = np.random.choice(np.arange(num_train), batch_size)
      X_batch = X[indices_random, :]
      y_batch = y[indices_random]

      #########################################################################
      #                             FIM DO SEU CODIGO                         #
      #########################################################################

      # Calcule a funcao de custo e os gradientes usando o minibatch atual
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: use os gradientes no dicionario grads para atualizar os         #
      # parametros da rede (armazenados no dicionario self.params)            #
      # usando gradiente descendente estocastico. 
      #########################################################################

      # acho que aqui eh soh isso mesmo     
      self.params['W1'] += -learning_rate*grads['W1']
      self.params['W2'] += -learning_rate*grads['W2']
      self.params['b1'] += -learning_rate*grads['b1']
      self.params['b2'] += -learning_rate*grads['b2']

      #########################################################################
      #                             FIM DO SEU CODIGO                         #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f ' % (it, num_iters, loss)        

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay
       

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use os pesos treinados para efetuar predicoes das instancias de teste. 
    Para cada instancia faca a predicao dos valores para cada uma das C classes.
    A classe com maior score sera a classe predita.     

    Entradas:
    - X: numpy array (N, D) com N D-dimensional instancias para classificar.

    Retorna:
    - y_pred: numpy array (N,) com as classes preditas para cada um dos elementos em X
      Para cada i, y_pred[i] = c significa c e a classe predita para X[i], onde 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implemente esta funcao. Provavelmente ela sera bastante simples   #
    ###########################################################################

    # Aqui eh isso, mas eu reduzi a funcao ....

    #layer_hidden = X.dot(self.params['W1']) + self.params['b1']
    #layer_relu = np.maximum(0, layer_hidden) # relu
    #scores = layer_relu.dot(self.params['W2']) + self.params['b2']
    #y_pred = np.argmax(scores, axis=1)

    # Funcao reduzida
    y_pred = np.argmax(np.maximum(X.dot(self.params['W1']) + self.params['b1'], 0).dot(self.params['W2'])+self.params['b2'], axis=1)

    ###########################################################################
    #                              FIM DO SEU CODIGO                          #
    ###########################################################################

    return y_pred


  def predictMarcelo(self, X):
      """
      Use os pesos treinados para efetuar predicoes das instancias de teste. 
      Para cada instancia faca a predicao dos valores para cada uma das C classes.
      A classe com maior score sera a classe predita.     

      Entradas:
      - X: numpy array (N, D) com N D-dimensional instancias para classificar.

      Retorna:
      - y_pred: numpy array (N,) com as classes preditas para cada um dos elementos em X
        Para cada i, y_pred[i] = c significa c e a classe predita para X[i], onde 0 <= c < C.
      """
      y_pred = None

      ###########################################################################
      # TODO: Implemente esta funcao. Provavelmente ela sera bastante simples   #
      ###########################################################################

      # Aqui eh isso, mas eu reduzi a funcao ....

      #layer_hidden = X.dot(self.params['W1']) + self.params['b1']
      #layer_relu = np.maximum(0, layer_hidden) # relu
      #scores = layer_relu.dot(self.params['W2']) + self.params['b2']
      #y_pred = np.argmax(scores, axis=1)

      # Funcao reduzida
      y_pred = np.argmax(np.maximum(X.dot(self.params['BestW1']) + self.params['Bestb1'], 0).dot(self.params['BestW2'])+self.params['Bestb2'], axis=1)

      ###########################################################################
      #                              FIM DO SEU CODIGO                          #
      ###########################################################################

      return y_pred
