"""
Álgebra Linear Computacional - Prova 1 - Questão 5
Aluna: Kelly Vinente dos Santos
Data: 08/10/2026

Solução do sistema Ax = b por decomposição LU (A = LU), sem pivoteamento:
  - L: triangular inferior com 1 na diagonal (guarda os multiplicadores);
  - U: triangular superior (resultado da eliminação de Gauss).
Depois resolve Ly = b (substituição progressiva) e Ux = y (substituição regressiva).

Restrição: numpy só pode ser usado para criar arrays (np.array, np.eye, np.zeros).
Todas as operações são feitas com laços explícitos.
"""

"""
Função implementada no código: resolve_lu que resolve o sistema Ax = b
Subfunções:
1) decomposição_lu que fatora a matriz A em duas matrizes, L triangular inferior 
e U triangular superior, sem pivoteamento.
2) substituicao_progressiva que resolve o sistema Ly = b.
3) substituicao_regressiva que resolve o sistema Ux = y.
4) _vetor_b que trata o vetor b em diferentes formatos.
"""


import numpy as np

def decomposicao_lu(A, tol=1e-12):
    """Fatora A = LU por eliminação de Gauss, sem pivoteamento.

    Lança Exception se A não for quadrada ou se algum pivô for nulo
    (|pivô| < tol), incluindo o último pivô U[n-1][n-1].
    """
    U = np.array(A, dtype=float)              # cópia de A em float (A original permanece o mesmo)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise Exception(f"A deve ser uma matriz quadrada; recebido formato {U.shape}.")
    n = U.shape[0]
    L = np.eye(n)                             # L começa como identidade, diagonal = 1

    for k in range(n):                        # k = coluna do pivô U[k][k]
        if abs(U[k][k]) < tol:
            raise Exception(
                f"Pivô nulo encontrado em U[{k}][{k}]. A decomposição LU sem pivoteamento "
                "não pode prosseguir: utilize uma função alternativa, como a decomposição "
                "LU com pivoteamento parcial (PA = LU)."
            )
        for i in range(k + 1, n):             # linhas abaixo do pivô
            m = U[i][k] / U[k][k]             # multiplicador da eliminação
            L[i][k] = m                       # o multiplicador é guardado em L
            U[i][k] = 0.0                     # elemento eliminado
            for j in range(k + 1, n):         # Linha_i <- Linha_i - m * Linha_k
                U[i][j] = U[i][j] - m * U[k][j]
    return L, U


def substituicao_progressiva(L, b, tol=1e-12):
    """Resolve Ly = b, com L triangular inferior (de cima para baixo)."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        if abs(L[i][i]) < tol:
            raise Exception(f"Elemento nulo na diagonal de L: L[{i}][{i}].")
        soma = 0.0
        for j in range(i):                    # termos já conhecidos: y[0], ..., y[i-1]
            soma += L[i][j] * y[j]
        y[i] = (b[i] - soma) / L[i][i]        # aqui L[i][i] = 1
    return y


def substituicao_regressiva(U, y, tol=1e-12):
    """Resolve Ux = y, com U triangular superior (de baixo para cima)."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < tol:
            raise Exception(f"Elemento nulo na diagonal de U: U[{i}][{i}].")
        soma = 0.0
        for j in range(i + 1, n):             # termos já conhecidos: x[i+1], ..., x[n-1]
            soma += U[i][j] * x[j]
        x[i] = (y[i] - soma) / U[i][i]
    return x


def _vetor_b(b, n):
    """Aceita b como vetor 1D (n,) ou coluna (n, 1).

    Retorna b como vetor 1D de floats e um indicador de que ele veio como coluna.
    """
    b = np.array(b, dtype=float)
    if b.ndim == 1 and b.shape[0] == n:
        return b, False
    if b.ndim == 2 and b.shape[0] == n and b.shape[1] == 1:
        return np.array([b[i][0] for i in range(n)], dtype=float), True
    raise Exception(f"b deve ter {n} elementos (vetor 1D ou coluna {n}x1); recebido formato {b.shape}.")


def resolve_lu(A, b, tol=1e-12):
    """Resolve Ax = b por decomposição LU sem pivoteamento.

    Retorna, nesta ordem: L, U e x. O vetor x sai no mesmo formato de b
    (1D se b for 1D; coluna n x 1 se b for coluna).
    """
    L, U = decomposicao_lu(A, tol)                    # 1) A = LU
    n = L.shape[0]
    b_vet, b_era_coluna = _vetor_b(b, n)
    y = substituicao_progressiva(L, b_vet, tol)       # 2) Ly = b
    x = substituicao_regressiva(U, y, tol)            # 3) Ux = y
    if b_era_coluna:
        x = np.array([[x[i]] for i in range(n)])
    return L, U, x