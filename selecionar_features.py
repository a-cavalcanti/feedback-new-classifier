from carregar_dados import carregar_dados
def matriz_correlacao(data): #RETORNA A MATRIZ DE CORRELAÇÃO
    cor = data.corr()
    return cor

def seleciona_features(data, n): #Retorna os n mais relevantes features de acordo com a correlação deles com a classe
    cor = matriz_correlacao(data) #Matriz de correlação
    classe = data.keys()[-1]
    cor_target = abs(cor[classe]) #Correlação com a classe
    relevant_features = cor_target.sort_values(ascending=False) #Ordena
#    print(relevant_features)
    return relevant_features.keys()[1:n] #Pega as n maiores correlações com a classe


data,_,_ = carregar_dados("pt - pt.csv")

print(seleciona_features(data, 11))


#data,_,_ = carregar_dados("cognitiveEnglish.csv")

#print(seleciona_features(data, 20))