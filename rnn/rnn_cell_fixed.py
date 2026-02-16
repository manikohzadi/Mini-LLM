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
    Docstring for matmul
    
    :param A: ماتریس اول
    :type A: Matrix
    :param B: ماتریس دوم
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

    for k in range(min(n, m)): # برای هر ستون k از 0 تا min(n, m) - 1 می خواهیم بازتاب householder بسازیم تا ستون k را از ردیف k به پایین صفر کنیم.
        x = [R[i][k] for i in range(k, n)]
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


def sigmoid(x: float) -> float: # تابع فعال سازی sigmoid که ورودی را بین 0 و 1 نگه می دارد
    # در LSTM برای دروازه ها استفاده می شود تا مشخص شود چه مقدار اطلاعات عبور کند یعنی 0 به معنای اصلا عبور نکند است و 1 به معنای قطعا عبور کند است       
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

def zeros(shape: Tuple[int, ...]) -> List[List[float]]:
    """این تابع یک آرایه پر شده با صفر بر می گرداند با شکل دلخواه"""
    if len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[0.0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError("Unsupported shape")
    
def random_vector(length: int, scale: float = 0.1) -> List[float]:
    """این تابع یک بردار تصادفی با طول و مقیاس دلخواه می سازد که داده هایی که تولید می کند بین -scale و scale است"""
    return [(random.random() * 2 - 1) * scale for _ in range(length)]

def random_matrix(rows: int, cols: int, scale: float = 0.1) -> List[List[float]]:
    """این تابع یک ماتریس تصادفی با سطر و ستون مقیاس دلخواه می سازد که داده هایی که تولید می کند بین -scale و scale است"""
    return [[(random.random() * 2 - 1) * scale for _ in range(cols)] for _ in range(rows)]

def full(shape: Tuple[int, ...], fill_value: int) -> List[List[float]]:
    """این تابع یک آرایه پر شده با مقدار دلخواه بر می گرداند با شکل دلخواه"""
    fill_value = float(fill_value)

    if len(shape) == 1:
        return [fill_value for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[fill_value for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[fill_value for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError("Unsupported shape")

def matvec_mul(mat: List[List[float]], vec: List[float]) -> List[float]:
    if len(mat[0]) != len(vec):
        raise ValueError("باید تعداد ستون های بردار و ماتریس برابر باشد.")
    result = []
    for row in mat:
        s = 0.0
        for a, b in zip(row, vec):
            s += a * b
        result.append(s)
    return result

def vecs_add(*vecs):
    if not vecs:
        raise ValueError("برای انجام این عملیات حداقل یک بردار لازم است.")
    n = len(vecs[0])
    for v in vecs:
        if len(v) != n:
            raise ValueError("باید تمام بردار ها طول یکسانی داشته باشند")
    return [sum(values) for values in zip(*vecs)]

def layer_norm(vec: List[float], eps: float = 1e-5, gamma: List[float] = None, beta: List[float] = None) -> List[float]:
    mean = sum(vec) / len(vec)
    var = sum((x - mean) ** 2 for x in vec) / len(vec)
    inv_std = 1.0 / math.sqrt(var + eps)
    out = [(x - mean) * inv_std for x in vec]
    if gamma is not None and beta is not None:
        out = [g * o + b for o, g, b in zip(out, gamma, beta)]
    return out

class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        scale = math.sqrt(1.0 / input_size)
        
        self.W = random_matrix(4 * hidden_size, input_size, scale)
        self.U = orthogonal_matrix(4 * hidden_size, hidden_size)
        self.b = zeros((4 * hidden_size))
        self.b[self.hidden_size:2*self.hidden_size] = full((hidden_size), 1)

        self.gamma = full((4 * hidden_size,), 1.0)
        self.beta = zeros((4 * hidden_size,))

        self.V_i = random_vector(hidden_size, scale)
        self.V_f = random_vector(hidden_size, scale)
        self.V_o = random_vector(hidden_size, scale)

    def forward(self, x: List[float], h_prev: List[float], c_prev: List[float]) -> Tuple[List[float], List[float]]:
        Wx = matvec_mul(self.W, x)
        Uh = matvec_mul(self.U, h_prev)

        z = vecs_add(Wx, Uh, self.b)
        z = layer_norm(z, gamma=self.gamma, beta=self.beta)

        H = self.hidden_size

        z_i = z[0:H]
        z_f = z[H:2*H]
        z_g = z[2*H:3*H]
        z_o = z[3*H:4*H]

        i = [sigmoid(z_i[j] + self.V_i[j] * c_prev[j]) for j in range(H)]
        f = [sigmoid(z_f[j] + self.V_f[j] * c_prev[j]) for j in range(H)]
        g = [tanh(z_g[j]) for j in range(H)]

        c = [f[j] * c_prev[j] + i[j] * g[j] for j in range(H)]

        o = [sigmoid(z_o[j] + self.V_o[j] * c[j]) for j in range(H)]
        h = [o[j] * tanh(c[j]) for j in range(H)]

        return h, c
