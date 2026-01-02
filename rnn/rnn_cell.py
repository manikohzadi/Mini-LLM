import math # اضافه کردن کتابخانه math برای نوشتن activation functions مثل sigmoid و tanh
import random # اضافه کردن این کتابخانه برای ساخت وزن ها و بایاس ها ماتریس ها
from typing import List, Tuple # اضافه کردن این کتابخانه برای type annotations توابع

def sigmoid(x: float) -> float: # تابع فعال سازی sigmoid
    """sigmoid activation function for Neural Networks and LSTM RNN"""
    return 1 / (1 + math.exp(-x)) # توضیحات در عکس function_formulas.png

def tanh(x: float) -> float: # activation function تانژانت هایپربولیک
    """hyperbolic tangent activation function for Neural Networks and LSTM RNN"""
    return math.tanh(x) # توضیحات در عکس function_formulas.png

def zeros(shape: Tuple[int, ...]) -> List[List[float]]:
    """این تابع یک آرایه پر شده با صفر بر می گرداند با شکل دلخواه"""
    if len(shape) == 1: # بررسی کردن اینکه shape تک بعدی باشد
        return [0.0 for _ in range(shape[0])] # ساخت لیستی از صفر ها به طول اولین عضو تاپل shape با استفاده از List Comprehension
    elif len(shape) == 2: # بررسی کردن اینکه shape دو بعدی باشد
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])] # لیستی از لیست صفر ها به طول (shape[0], shape[1])
    else: # اگر کاربر shapeی وارد کرده بود که طولش بیشتر از 2 بود مثلا 3 بعدی یا 4 بعدی وارد کرده بود
        raise ValueError("Unsupported shape") # ارور می دهیم چون این تابع فقط آرایه و ماتریس می سازد نه تنسور

def random_matrix(rows: int, cols: int, scale: float = 0.1) -> List[List[float]]:
    return [[(random.random() * 2 - 1) * scale for _ in range(cols)] for _ in range(rows)]

def matvec_mul(mat: List[List[float]], vec: List[float]) -> List[float]:
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

def vec_add(v1: List[float], v2: List[float]) -> List[float]:
    return [v1[i] + v2[i] for i in range(len(v1))]

def layer_norm(vec: List[float], eps: float = 1e-5) -> List[float]:
    mean = sum(vec) / len(vec)
    variance = sum((x - mean) ** 2 for x in vec) / len(vec)
    return [(x - mean) / math.sqrt(variance + eps) for x in vec]

class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        scale = math.sqrt(1 / input_size)
        # weights
        self.W_i = random_matrix(hidden_size, input_size, scale)
        self.W_f = random_matrix(hidden_size, input_size, scale)
        self.W_o = random_matrix(hidden_size, input_size, scale)
        self.W_g = random_matrix(hidden_size, input_size, scale)

        self.U_i = random_matrix(hidden_size, hidden_size, scale)
        self.U_f = random_matrix(hidden_size, hidden_size, scale)
        self.U_o = random_matrix(hidden_size, hidden_size, scale)
        self.U_g = random_matrix(hidden_size, hidden_size, scale)

        # peephole connections
        self.V_i = [random.random() * scale for _ in range(hidden_size)]
        self.V_f = [random.random() * scale for _ in range(hidden_size)]
        self.V_o = [random.random() * scale for _ in range(hidden_size)]

        # biases
        self.b_i = [0.0 for _ in range(hidden_size)]
        self.b_f = [1.0 for _ in range(hidden_size)]  # forget bias = 1
        self.b_o = [0.0 for _ in range(hidden_size)]
        self.b_g = [0.0 for _ in range(hidden_size)]

        # states
        self.h = [0.0 for _ in range(hidden_size)]
        self.c = [0.0 for _ in range(hidden_size)]

    def forward(self, x: List[float]) -> List[float]:
        # input gate
        i = [sigmoid(a + b + c * d + e) for a, b, c, d, e in zip(
            matvec_mul(self.W_i, x),
            matvec_mul(self.U_i, self.h),
            self.V_i,
            self.c,
            self.b_i
        )]
        # forget gate
        f = [sigmoid(a + b + c * d + e) for a, b, c, d, e in zip(
            matvec_mul(self.W_f, x),
            matvec_mul(self.U_f, self.h),
            self.V_f,
            self.c,
            self.b_f
        )]
        # cell candidate
        g = [tanh(a + b + e) for a, b, e in zip(
            matvec_mul(self.W_g, x),
            matvec_mul(self.U_g, self.h),
            self.b_g
        )]
        # new cell state
        self.c = [f_t * c_prev + i_t * g_t for f_t, c_prev, i_t, g_t in zip(f, self.c, i, g)]
        # output gate
        o = [sigmoid(a + b + c_t * v + e) for a, b, c_t, v, e in zip(
            matvec_mul(self.W_o, x),
            matvec_mul(self.U_o, self.h),
            self.c,
            self.V_o,
            self.b_o
        )]
        # new hidden state
        self.h = [o_t * tanh(c_t) for o_t, c_t in zip(o, self.c)]
        return self.h
