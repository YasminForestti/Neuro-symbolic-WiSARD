import numpy as np
from collections import defaultdict
import itertools

class WiSARD:
    def __init__(self, topologia_do_input, tuple_size=4, threshold = 0, regra = None, alfa = None):
        self.threshold = threshold
        self.input_size = np.reshape(topologia_do_input, -1).size
        self.tuple_size = tuple_size
        self.num_tuples = self.input_size // tuple_size
        self.rams = {}
        if regra is not None:
            self.mapping =  self.__insert_rule_general(regra, topologia_do_input, alfa)
        else: 
            self.mapping = [np.random.permutation(self.input_size)[:self.tuple_size] for _ in range(self.num_tuples)]


    def print_estados(self):
        print(f"Estado da RAM : {self.rams} \nMapping : {self.mapping}")

    
    def train(self, X, y):
        for xi, label in zip(X, y):
            if label not in self.rams:
                self.rams[label] = [defaultdict(int) for _ in range(len(self.mapping))]
            for i, idxs in enumerate(self.mapping):
                addr = tuple(xi[idxs])  # Pode ter tamanho variável
                self.rams[label][i][addr] += 1


    def __insert_rule_general(self, label_regras_dict, topologia_do_input, alfa):
        flat_vars = [var for ram_vars in topologia_do_input for var in ram_vars]
        var_to_idx = {var: i for i, var in enumerate(flat_vars)}

        # 1. Coletar variáveis usadas nas regras
        variaveis_usadas_em_regras = set()
        for regras in label_regras_dict.values():
            for vars_usadas, _ in regras:
                variaveis_usadas_em_regras.update(vars_usadas)

        # 2. Identificar variáveis livres
        variaveis_livres = [var for var in flat_vars if var not in variaveis_usadas_em_regras]
        variaveis_livres_restantes = variaveis_livres.copy()

        mapping = []

        # 3. Criar RAMs baseadas em regras
        regras_exemplo = next(iter(label_regras_dict.values()))
        for vars_usadas, _ in regras_exemplo:
            idxs = [var_to_idx[var] for var in vars_usadas]
            mapping.append(np.array(idxs))

        # 4. Criar RAMs adicionais com variáveis livres (mesmo que incompletas)
        while len(variaveis_livres_restantes) > 0:
            num_vars = min(self.tuple_size, len(variaveis_livres_restantes))
            variaveis_sorteadas = np.random.choice(variaveis_livres_restantes, num_vars, replace=False)
            for var in variaveis_sorteadas:
                variaveis_livres_restantes.remove(var)
            idxs = [var_to_idx[var] for var in variaveis_sorteadas]
            mapping.append(np.array(idxs))

        # 5. Inicializar RAMs para todas as classes com base no tamanho do mapping
        for label in label_regras_dict:
            if label not in self.rams:
                self.rams[label] = [defaultdict(float) for _ in range(len(mapping))]

        # 6. Preencher RAMs conforme as regras
        for label, regras in label_regras_dict.items():
            for ram_idx, (vars_usadas, regra) in enumerate(regras):
                for bits in itertools.product([0, 1], repeat=len(vars_usadas)):
                    contexto = dict(zip(vars_usadas, bits))
                    if regra(**contexto):
                        addr = tuple(bits)
                        self.rams[label][ram_idx][addr] += alfa

        return mapping

        
    def predict(self, X, bleaching_start=1):
        preds = []
        for xi in X:
            bleaching = bleaching_start
            while True:
                votes = {}
                for label, rams in self.rams.items():
                    score = 0
                    for i, idxs in enumerate(self.mapping):
                        addr = tuple(xi[idxs])
                        if rams[i][addr] >= bleaching:
                            score += 1
                    votes[label] = score
                # print(f"Bleaching={bleaching}, votes={votes}")
                max_score = max(votes.values())
                # Seleciona classes com score máximo
                best_labels = [label for label, score in votes.items() if score == max_score]
                if len(best_labels) == 1 or max_score == 0:
                    # Se só uma classe tem score máximo ou ninguém respondeu, para
                    if max_score <=  bleaching:
                        preds.append("não identificado")
                    else:
                        preds.append(best_labels[0])
                    break
                else:
                    # Empate, aumenta o bleaching
                    bleaching += 1
        return np.array(preds)