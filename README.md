# 🎮 Machine Learning na pratica

## 👩‍🏫 Objetivo da Atividade
Introduzir Machine Learning de maneira simples.

Hoje vocês irão:
- Treinar modelos de ML simples
- Testar e comparar resultados
- Trabalhar em equipe
- Apresentar seus modelos para a turma

---

# 🧩 Tema do Desafio
As equipes devem criar um modelo que **preveja a nota de um aluno** baseado em:

- Horas de estudo  
- Quantidade de exercícios resolvidos  
- Presença (%)  

Cada equipe recebe os mesmos dados e deve tentar criar **o melhor modelo da sala**.

---

# 📊 Base de Dados

As equipes devem começar com o seguinte dataset:

```python
import pandas as pd

dados = {
    "horas_estudo": [1,2,3,4,5,6,7,8,9,10],
    "exercicios":   [1,1,2,3,3,4,4,5,5,6],
    "presenca":     [60,65,70,75,80,85,88,90,93,95],
    "nota":         [50,55,60,63,67,72,78,82,88,91]
}

df = pd.DataFrame(dados)
df
```

---

# 🏆 Regras da Competição

### Cada equipe deve:

1. Criar um modelo de Machine Learning usando Python.  
   Sugestões:
   - `LinearRegression`
   - `DecisionTreeRegressor`
   - `KNeighborsRegressor`

2. Separar os dados com `train_test_split`.

3. Treinar o modelo e calcular o **score** (acurácia/regressão R²).

4. Mostrar o resultado final para o professor.

---

# ➕ Pontos Extras

| Ação | Pontos |
|------|--------|
| Criar gráfico visual | +1 |
| Prever nota de um aluno “novo” | +1 |
| Melhor visual / organização do código | +1 |
| Explicação para a sala sobre como o modelo funciona | +2 |
| Melhor score da competição | 🏆 Bonus |

---

# 🧠 Código Base

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Separação entre entradas (X) e saída (y)
X = df[["horas_estudo", "exercicios", "presenca"]]
y = df["nota"]

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Escolha do modelo
# modelo = LinearRegression()
# modelo = KNeighborsRegressor(n_neighbors=3)
modelo = DecisionTreeRegressor()

# Treinar o modelo
modelo.fit(X_train, y_train)

# Score final
score = modelo.score(X_test, y_test)
print("Pontuação do modelo:", score)

# Previsão de um novo aluno
novo = [[7, 4, 90]]
print("Previsão de nota:", modelo.predict(novo)[0])
```

---

# 📈 Sugestão de Gráfico (pontos extras)

```python
import matplotlib.pyplot as plt

plt.scatter(df["horas_estudo"], df["nota"])
plt.xlabel("Horas de Estudo")
plt.ylabel("Nota")
plt.title("Relação entre Estudo e Nota")
plt.show()
```

---

# ✔️ Observação
Esta atividade foi criada para ser resolvida mesmo sem conhecimento profundo de matemática ou estatística ou morando no refugio 🤠👌🏻.

---
