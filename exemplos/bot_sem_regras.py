import numpy as np
import pandas as pd
from WiSARD import WiSARD
from sklearn.metrics import accuracy_score
import kagglehub
import numpy as np
from sklearn.model_selection import train_test_split
from WiSARD import WiSARD
import os


path = kagglehub.dataset_download("juice0lover/users-vs-bots-classification")
path = f"{path}\\bots_vs_users.csv"
dataset_bot = pd.read_csv(path)
dataset_bot = dataset_bot.dropna()

dataset_bot_features = dataset_bot[['avg_comments', 'avg_likes', 'avg_keywords', 'avg_views', 'avg_text_length',  'is_profile_closed', 'has_hometown',  'all_posts_visible', 'has_career', 'has_personal_data', 'has_interests','target']].reset_index()
dataset_bot_features.loc[:, 'has_personal_data'] = dataset_bot_features['has_personal_data'].replace('Unknown', 0.0)
dataset_bot_features.loc[:, 'has_interests'] = dataset_bot_features['has_interests'].replace('Unknown', 2.0)
dataset_bot_features.loc[:, 'has_career'] = dataset_bot_features['has_career'].replace('Unknown', 2.0)
dataset_bot_features.loc[:, 'all_posts_visible'] = dataset_bot_features['all_posts_visible'].replace('Unknown', 2.0)
dataset_bot_features.loc[:, 'has_hometown'] = dataset_bot_features['has_hometown'].replace('Unknown', 2.0)
dataset_bot_features.loc[:, 'is_profile_closed'] = dataset_bot_features['is_profile_closed'].replace('Unknown', 1.0)
dataset_bot_features = dataset_bot_features.astype(float).drop(columns = 'index')


X = dataset_bot_features.drop('target', axis=1)
y = dataset_bot_features['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reset_index()[['target']]
y_test = y_test.reset_index()[['target']]

def thermometer_encoding(df, n_bits=10):
    binary_matrix = []
    
    for col in df.columns:
        col_vals = df[col].values
        
        # Normalizando os valores para o intervalo [0, 1]
        min_val = np.min(col_vals)
        max_val = np.max(col_vals)
        range_val = max_val - min_val
        
        # Caso o range seja zero (coluna constante), tratar separadamente
        if range_val == 0:
            col_encoded = np.zeros((len(col_vals), n_bits))  # Se todos os valores são iguais, codificar como 0
        else:
            # Aplicando a codificação termômetro com base na normalização
            col_encoded = np.array([
                [1 if i < int(((val - min_val) / range_val) * n_bits) else 0 for i in range(n_bits)]
                for val in col_vals
            ])
        
        # Adicionando a matriz binária codificada para a coluna
        binary_matrix.append(col_encoded)
    
    # Combinando todas as colunas binarizadas horizontalmente
    return np.hstack(binary_matrix)

X_train_bin = pd.DataFrame( thermometer_encoding(X_train, n_bits=10))
X_test_bin = pd.DataFrame( thermometer_encoding(X_test, n_bits=10))

# Prepare os dados binarizados (ajuste conforme seu notebook)
X_train_bin_fl = [np.array(row) for row in X_train_bin.values]
X_test_bin_fl = [np.array(row) for row in X_test_bin.values]
y_train_fl = y_train['target'].values
y_test_fl = y_test['target'].values

acuracias = []

for i in range(100):
    input_size = X_train_bin.shape[1]
    topologia_do_input = ['x' + str(i) for i in range(input_size)]
    wisard = WiSARD(topologia_do_input, tuple_size=6, threshold=0)

    wisard.train(X_train_bin_fl, y_train_fl)
    pred = wisard.predict(X_test_bin_fl, bleaching_start=1)
    
    y_pred_float = np.where(np.array(pred) == "não identificado", -1.0, pred).astype(float)
    y_test_float = y_test_fl.astype(float)
    acuracia = accuracy_score(y_test_float, y_pred_float)
    acuracias.append(acuracia)

media = np.mean(acuracias)
print(f"Média das acurácias em 100 execuções: {media:.4f}")

# Salva a média em um arquivo CSV em modo append

csv_path = 'media_acuracia_wisard.csv'
df = pd.DataFrame({'Modelo': ['WiSARD sem Regra'], 'Acurácia': [media] , 'Alpha': '-'})

if os.path.exists(csv_path):
    df.to_csv(csv_path, mode='a', header=False, index=False)
else:
    df.to_csv(csv_path, index=False)

print("Média salva em 'media_acuracia_wisard.csv'")