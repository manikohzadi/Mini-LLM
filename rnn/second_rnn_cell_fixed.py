import copy # برای استفاده کردن از توابعی چون deep copy
import math # اضافه کردن کتابخانه math برای نوشتن activation functions مثل sigmoid و tanh
import random # اضافه کردن این کتابخانه برای ساخت وزن ها و بایاس ها ماتریس ها
from typing import List, Tuple # اضافه کردن این کتابخانه برای type annotations توابع


Matrix = List[List[float]] # ساختن این data type برای پیروی از قانون DRY و خوانایی بیشتر

def random_gaussian_matrix(rows: int, cols: int) -> Matrix: # سازنده ماتریس گاوسی که درایه های آن از توزیع گاوسی نمونه برداری شده اند
    """
    سازنده ماتریس گاوسی که درایه های آن از توزیع گاوسی نمونه برداری شده اند که میانگین آن 0 و انحراف از معیار آن 1 است
    
    :param rows: تعداد سطر های ماتریس
    :type rows: int
    :param cols: تعداد ستون های ماتریس
    :type cols: int
    :return: ماتریس گاوسی با ابعاد rows × cols
    :rtype: Matrix
    """
    return [[random.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def identity(n: int) -> Matrix: # سازنده ماتریس identity که اعداد روی قطر اصلی ماتریس 1 و بقیه 0 هستند.
    """
    سازنده ماتریس identity که فقط اعداد روی قطر اصلی آن 1 هستند و بقیه 0
    
    :param n: طول سطر و ستون ماتریس
    :type n: int
    :return: یک ماتریس identity به ابعاد n × n
    :rtype: Matrix
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(A: Matrix) -> Matrix: # ترانهاده که جای سطر و ستون با هم عوض میشه یعنی اعدادی که باهم در یک سطر بودن میان تو یه ستون زیر هم دیگه
    """
    ترانهاده که جای سطر و ستون را عوض می کند
    
    :param A: ماتریس ورودی
    :type A: Matrix
    :return: ماتریس ورودی با این تفاوت که جای سطر و ستون عوض شده
    :rtype: Matrix
    """
    return [list(row) for row in zip(*A)] # این تکنیک باعث می شود که ابتدا عضو اول بعد دوم و بعد ... به ترتیب تبدیل بشن به یک سطر جدید و نیاز به کار پیچیده ای نباشه


def matmul(A: Matrix, B: Matrix) -> Matrix: # این تابع دو ماتریس را در هم ضرب می کند
    """
    ضرب ماتریس در ماتریس (برای اینکه خطا نگیرید باید n = k باشد)
    
    :param A: ماتریس اول m × n
    :type A: Matrix
    :param B: ماتریس دوم k × i
    :type B: Matrix
    :return: ماتریس اول × ماتریس دوم
    :rtype: Matrix
    """
    rows, cols, inner = len(A), len(B[0]), len(B) # تعداد سطر ماتریس اول, تعداد ستون ماتریس دوم, تعداد سطر ماتریس دوم
    return [
        [sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]

    # در قطعه کد بالا می آییم:
    #       درایه های سطر اول A را به ترتیب در تمام درایه های ستون دوم B .ضرب می کنیم و درایه های حاصل را به ترتیب جمع می کنیم و آن را در ستون اول سطر اول ماتریس حاصل می نویسیم
    #       الان کار بالا را برای ستون های بعدی ماتریس B انجام می دهیم و دوباره آن را در ستون های بعدی سطر اول ماتریس حاصل می نویسیم.
    #       وقتی کار بالا را تمام کردیم می رویم سراغ سطر های بعدی ماتریس A و کار های قبلی را برایش تکرار می کنیم و این بار حاصل ها را در سطر های بعدی ماتریس حاصل می نویسیم.


def householder_qr(A: Matrix) -> Matrix:
    """
    Docstring for householder_qr
    
    :param A: یک ماتریس n × m
    :type A: Matrix
    :return: یک ماتریس orthogonal Q با روش householder
    :rtype: Matrix
    """
    n = len(A) # تعداد سطر های ماتریس اول
    m = len(A[0]) # تعداد ستون های ماتریس دوم
    R = copy.deepcopy(A) # گرفتن یک deep copy از A. می توانستیم بنویسیم [row[:] for row in A] که دقیقا این کار را انجام می دهد ولی به اندازه این یکی سریع و امن نیست
    Q = identity(n) # یک ماتریس همانی n × n

    # حلقه ی اصلی روی ستون ها
    for k in range(min(n, m)): # برای هر ستون k از 0 تا min(n, m) - 1 می خواهیم بازتاب householder بسازیم تا ستون k را از ردیف k به پایین صفر کنیم.
        x = [R[i][k] for i in range(k, n)] # x برداری از ستون k از ردیف k به پایین می‌گیرد
        # این همون برداریه که می خواهیم با اون بازتاب رو صاف کنیم یعنی اینکه فقط اولین عضو غیر صفر باقی بماند و بقیه صفر شوند
        norm_x = math.sqrt(sum(v * v for v in x))
        if norm_x == 0:
            continue

        sign = -1.0 if x[0] < 0 else 1.0
        u1 = x[0] + sign * norm_x
        v = [u1] + x[1:]
        norm_v = math.sqrt(sum(vv * vv for vv in v))
        v = [vv / norm_v for vv in v]

        for j in range(k, m):
            dot = sum(v[i] * R[k + i][j] for i in range(len(v)))
            for i in range(len(v)):
                R[k + i][j] -= 2 * v[i] * dot

        for j in range(n):
            dot = sum(v[i] * Q[j][k + i] for i in range(len(v)))
            for i in range(len(v)):
                Q[j][k + i] -= 2 * v[i] * dot

    return Q


def orthogonal_matrix(rows: int, cols: int) -> Matrix:
    """
    Returns a matrix Q of shape (rows, cols) where columns are orthonormal:
        Q^T Q = I
    Suitable for LSTM / RNN U matrices.
    """
    A = random_gaussian_matrix(rows, cols)
    Q_full = householder_qr(A)
    return [row[:cols] for row in Q_full]


def check_orthogonality(Q: Matrix) -> Matrix:
    Qt = transpose(Q)
    return matmul(Qt, Q)


def sigmoid(x: float) -> float: # تابع فعال سازی sigmoid که ورودی را بین 0 و 1 نگه می دارد
    """sigmoid activation function for Neural Networks and LSTM RNN"""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def tanh(x: float) -> float: # activation function تانژانت هایپربولیک که ورودی را بین -1 و 1 نگه می دارد
    """hyperbolic tangent activation function for Neural Networks and LSTM RNN"""
    return math.tanh(x)

def zeros(shape: Tuple[int, ...]):
    if len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError("Unsupported shape")

def full(shape: Tuple[int, ...], fill_value: float):
    if len(shape) == 1:
        return [fill_value for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[fill_value for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError("Unsupported shape")

def add_bias(matrix: Matrix, bias: List[float]) -> Matrix:
    return [[val + bias[j] for j, val in enumerate(row)] for row in matrix]

def layer_norm_matrix(mat: Matrix, eps: float = 1e-5):
    output = []
    for vec in mat:
        mean = sum(vec) / len(vec)
        var = sum((x - mean) ** 2 for x in vec) / len(vec)
        inv_std = 1.0 / math.sqrt(var + eps)
        output.append([(x - mean) * inv_std for x in vec])
    return output


class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        scale = math.sqrt(1.0 / input_size)
        
        self.W = random_gaussian_matrix(4 * hidden_size, input_size)
        self.U = orthogonal_matrix(4 * hidden_size, hidden_size)
        self.b = zeros((4 * hidden_size,))
        self.b[self.hidden_size:2*self.hidden_size] = full((hidden_size,), 1.0)

    def forward(self, x: Matrix, h_prev: Matrix, c_prev: Matrix):
        Wx = matmul(x, transpose(self.W))
        Uh = matmul(h_prev, transpose(self.U))

        z = add_bias([[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(Wx, Uh)], self.b)
        z = layer_norm_matrix(z)

        H = self.hidden_size
        batch_size = len(z)

        h_out = []
        c_out = []

        for b_idx in range(batch_size):
            row = z[b_idx]
            z_i = row[0:H]
            z_f = row[H:2*H]
            z_g = row[2*H:3*H]
            z_o = row[3*H:4*H]

            i = [sigmoid(z_i[j]) for j in range(H)]
            f = [sigmoid(z_f[j]) for j in range(H)]
            g = [tanh(z_g[j]) for j in range(H)]

            c = [f[j] * c_prev[b_idx][j] + i[j] * g[j] for j in range(H)]
            o = [sigmoid(z_o[j]) for j in range(H)]
            h = [o[j] * tanh(c[j]) for j in range(H)]

            h_out.append(h)
            c_out.append(c)

        return h_out, c_out