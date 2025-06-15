import numpy as np

# Matrizes para classe T
matrizes_T = [
    [
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0]
    ],
    [
        [1, 1, 1],
        [0, 1, 0],
        [1, 1, 0]
    ]
]

# Matrizes para classe H
matrizes_H = [
    [
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1]
    ],
    [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], 
    [
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 1]
    ]
]

# MATRIZ AMBIGUA
matriz_ambigua = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 0]
]

# Teste: sem regra, pode empatar entre T e H; com regra, a classe correta é escolhida
X_train = [np.array(m).flatten() for m in matrizes_T + matrizes_H]
y_train = ["T"] * len(matrizes_T) + ["H"] * len(matrizes_H)

X_test = [np.array(matriz_ambigua).flatten()]

print(f"X_train: {X_train}\n")

# Topologia e regras
topologia_do_input = [
    ['a', 'b', 'c'],
    ['d', 'e', 'f'],
    ['g', 'h', 'i']
]


regras_T = {
    "T": [
        (['a', 'b', 'c'], lambda a, b, c: a and b and c),
        (['b', 'e', 'h'], lambda b, e, h: b and e and h)
    ]
}

# Exemplo de uso:
from WiSARD import WiSARD

# wisard = WiSARD(topologia_do_input, tuple_size=3, regra = regras_T, alfa=10, threshold=0)

wisard = WiSARD(topologia_do_input, tuple_size=3, threshold=0)

wisard.print_estados()
print("\n")

wisard.train(X_train, y_train)

wisard.print_estados()
print("\n")


# Teste
pred = wisard.predict(X_test, bleaching_start=1)
print("Predição para matriz ambígua:", pred)