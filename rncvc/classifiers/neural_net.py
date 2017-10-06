import numpy as np


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

  def __init__(self, input_size, hidden_size, output_size, std=1e-4, dropout=0):
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

  def loss(self, X, y=None, reg=0.0):
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

    # Log-verossimilhanca
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
    
    dscores = softmax_score
    dscores[range(N),y] -= 1
    dscores /= N

    #print softmax_score
    #print "/n"
    #print dscores
    #print "/n"
    #print "Novo/n"


    # W2 and b2
    grads['W2'] = np.dot(hidden_layer.T, dscores)
    grads['b2'] = np.sum(dscores, axis=0)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    grads['W1'] = np.dot(X.T, dhidden)
    grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1

    #############################################################################
    #                              FIM DO SEU CODIGO                            #
    #############################################################################

    return loss, grads

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
      pass

      shuffle_indexes = np.arange(num_train)
      np.random.shuffle(shuffle_indexes)
      shuffle_indexes = shuffle_indexes[0:batch_size-1]
      X_batch = X[shuffle_indexes, :]
      y_batch = y[shuffle_indexes]

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
      pass

      self.params['W1'] += -learning_rate*grads['W1']
      self.params['W2'] += -learning_rate*grads['W2']
      self.params['b1'] += -learning_rate*grads['b1']
      self.params['b2'] += -learning_rate*grads['b2']

      #########################################################################
      #                             FIM DO SEU CODIGO                         #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

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
    pass

    l1 = np.dot(X, self.params['W1']) + self.params['b1']
    l1[l1<=0] = 0 # ReLu
    scores = np.dot(l1, self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)

    ###########################################################################
    #                              FIM DO SEU CODIGO                          #
    ###########################################################################

    return y_pred


