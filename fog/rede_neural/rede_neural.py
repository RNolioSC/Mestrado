from numpy import genfromtxt
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
import time

if __name__ == '__main__':
    
    tempoi = time.time()

    
    # Accuracy: 79.30
    dataset = genfromtxt(r'../../simulacao/csv/cenario_1-1.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    X = dataset[:15300, 0:3]
    Y = dataset[:15300, 3]
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 1-1")

    # Accuracy: 71.73
    dataset = genfromtxt(r'../../simulacao/csv/cenario_1-2.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:3060, 0:3]
    #Y = dataset[:3060, 3]
    X = numpy.append(X, dataset[:3060, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:3060, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 1-2")
                
    # Accuracy: 85.97
    dataset = genfromtxt(r'../../simulacao/csv/cenario_1-3.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:4590, 0:3]
    #Y = dataset[:4590, 3]
    X = numpy.append(X, dataset[:4590, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:4590, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 1-3")
    
    # Accuracy: 67.14
    dataset = genfromtxt(r'../../simulacao/csv/cenario_1-4.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:13515, 0:3]
    #Y = dataset[:13515, 3]
    X = numpy.append(X, dataset[:13515, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:13515, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 1-4")
                
    # Accuracy: 56.37    
    dataset = genfromtxt(r'../../simulacao/csv/cenario_2-1.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:16065, 0:3]
    #Y = dataset[:16065, 3]
    X = numpy.append(X, dataset[:16065, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:16065, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 2-1")
    
    # Accuracy: 78.05
    dataset = genfromtxt(r'../../simulacao/csv/cenario_2-2.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:20400, 0:3]
    #Y = dataset[:20400, 3]
    X = numpy.append(X, dataset[:20400, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:20400, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 2-2")

    
    # Accuracy: 78.10
    dataset = genfromtxt(r'../../simulacao/csv/cenario_3-1.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:21420, 0:3]
    #Y = dataset[:21420, 3]
    X = numpy.append(X, dataset[:21420, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:21420, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 3-1")
    
    # Accuracy: 52.19
    dataset = genfromtxt(r'../../simulacao/csv/cenario_3-2.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:6120, 0:3]
    #Y = dataset[:6120, 3]
    X = numpy.append(X, dataset[:6120, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:6120, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 3-2")
   
    # Accuracy: 63.99
    dataset = genfromtxt(r'../../simulacao/csv/cenario_3-3.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:4590, 0:3]
    #Y = dataset[:4590, 3]
    X = numpy.append(X, dataset[:4590, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:4590, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 3-3")
   
    # Accuracy: 50.38
    dataset = genfromtxt(r'../../simulacao/csv/cenario_3-4.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:4335, 0:3]
    #Y = dataset[:4335, 3]
    X = numpy.append(X, dataset[:4335, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:4335, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 3-4")
    
    # Accuracy: 86.39
    dataset = genfromtxt(r'../../simulacao/csv/cenario_4-1.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:12240, 0:3]
    #Y = dataset[:12240, 3]
    X = numpy.append(X, dataset[:12240, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:12240, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 4-1")
    
    #Accuracy: 53.27
    dataset = genfromtxt(r'../../simulacao/csv/cenario_4-2.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:4590, 0:3]
    #Y = dataset[:4590, 3]
    X = numpy.append(X, dataset[:4590, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:4590, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 4-2")
    
    # Accuracy: 54.38
    dataset = genfromtxt(r'../../simulacao/csv/cenario_4-3.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:5355, 0:3]
    #Y = dataset[:5355, 3]
    X = numpy.append(X, dataset[:5355, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:5355, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 4-3")
    
    # Accuracy: 57.23
    dataset = genfromtxt(r'../../simulacao/csv/cenario_4-4.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:3825, 0:3]
    #Y = dataset[:3825, 3]
    X = numpy.append(X, dataset[:3825, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:3825, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 4-4")
    
    # Accuracy: 63.56
    dataset = genfromtxt(r'../../simulacao/csv/cenario_4-5.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:3825, 0:3]
    #Y = dataset[:3825, 3]
    X = numpy.append(X, dataset[:3825, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:3825, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 4-5")
    
    # Accuracy: 70.90
    dataset = genfromtxt(r'../../simulacao/csv/cenario_4-6.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:6630, 0:3]
    #Y = dataset[:6630, 3]
    X = numpy.append(X, dataset[:6630, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:6630, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 4-6")
    
    # Accuracy: 56.34
    dataset = genfromtxt(r'../../simulacao/csv/cenario_5-1.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:24225, 0:3]
    #Y = dataset[:24225, 3]
    X = numpy.append(X, dataset[:24225, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:24225, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 5-1")
   
   # Accuracy: 75.90
    dataset = genfromtxt(r'../../simulacao/csv/cenario_5-2.csv', encoding='latin-1', delimiter=',', skip_header=2,
                         usecols=(13, 14, 15, 16))
    #X = dataset[:12240, 0:3]
    #Y = dataset[:12240, 3]
    X = numpy.append(X, dataset[:12240, 0:3], axis=0)
    Y = numpy.append(Y, dataset[:12240, 3], axis=0)
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if numpy.isnan(X[i][j]):
                print(i, j, "cenario 5-2")

    model = Sequential()
    model.add(Dense(25, input_dim=3, activation='softsign'))
    model.add(Dense(25, activation='softsign'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    history = model.fit(X, Y, epochs=10, batch_size=100)

    _, accuracy = model.evaluate(X, Y)
    print('Accuracy: %.2f' % (accuracy * 100))

    predictions = model.predict_classes(X)
    
    tempof = time.time()
    print("tempo de execucao (s):", tempof-tempoi)
   
    model.save('modelo.H5')
    
    # all scenarios: Accuracy: 65.74

    #''
    # graficos de acuracia e validacao
    #plt.plot(history.history['accuracy'])
    #plt.ylabel('Acurácia')
    #plt.xlabel('Época')
    #plt.legend(['Treinamento'])
    #plt.show()

    #plt.plot(history.history['loss'])
    #plt.ylabel('Perda')
    #plt.xlabel('Época')
    #plt.legend(['Treinamento'])
    #plt.show()
    #''
