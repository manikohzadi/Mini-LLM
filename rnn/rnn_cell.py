import math # اضافه کردن کتابخانه math برای نوشتن activation functions مثل sigmoid و tanh
import random # اضافه کردن این کتابخانه برای ساخت وزن ها و بایاس ها ماتریس ها
from typing import List, Tuple # اضافه کردن این کتابخانه برای type annotations توابع


Matrix = List[List[float]]

def random_gaussian_matrix(rows: int, cols: int) -> Matrix:
    return [[random.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]

def orthogonal_matrix(rows: int, cols: int) -> Matrix:
    """
    Returns a matrix Q of shape (rows, cols) where columns are orthonormal:
        Q^T Q = I
    Suitable for LSTM / RNN U matrices.
    """

    A = random_gaussian_matrix(rows, cols)
    Q = [[0.0] * cols for _ in range(rows)]

    for k in range(cols):
        # بردار ستون k از A
        v = [A[i][k] for i in range(rows)]

        # حذف مؤلفه‌های قبلی (Modified Gram-Schmidt بسیار پایدار)
        for j in range(k):
            dot = sum(Q[i][j] * v[i] for i in range(rows))
            for i in range(rows):
                v[i] -= dot * Q[i][j]

        # نرمال‌سازی
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            # fallback نادر
            v = [random.gauss(0, 1) for _ in range(rows)]
            norm = math.sqrt(sum(x * x for x in v))

        for i in range(rows):
            Q[i][k] = v[i] / norm

    return Q

def sigmoid(x: float) -> float: # تابع فعال سازی sigmoid که ورودی را بین 0 و 1 نگه می دارد
    # در LSTM برای دروازه ها استفاده می شود تا مشخص شود چه مقدار اطلاعات عبور کند یعنی 0 به معنای اصلا عبور نکند است و 1 به معنای قطعا عبور کند است       
    """sigmoid activation function for Neural Networks and LSTM RNN"""
    # return 1 / (1 + math.exp(-x))  توضیحات در عکس function_formulas.png (این فرمول ریاضی محضشه)
    # نسخه دارای پایداری عددی:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def tanh(x: float) -> float: # activation function تانژانت هایپربولیک که ورودی را بین -1 و 1 نگه می دارد
    """hyperbolic tangent activation function for Neural Networks and LSTM RNN"""
    return math.tanh(x) # توضیحات در عکس function_formulas.png

def zeros(shape: Tuple[int, ...]) -> List[List[float]]:
    """این تابع یک آرایه پر شده با صفر بر می گرداند با شکل دلخواه"""
    if len(shape) == 1: # بررسی کردن اینکه shape تک بعدی باشد
        return [0.0 for _ in range(shape[0])] # ساخت لیستی از صفر ها به طول اولین عضو تاپل shape با استفاده از List Comprehension
    elif len(shape) == 2: # بررسی کردن اینکه shape دو بعدی باشد
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])] # لیستی از لیست صفر ها به ابعاد (shape[0], shape[1])
    elif len(shape) == 3:
        return [[[0.0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])] # لیستی از لیستی از لیست صفر ها به ابعاد (shape[0], shape[1], shape[2])
    else: # اگر کاربر shapeی وارد کرده بود که طولش بیشتر از 3 بود مثلا 4 بعدی یا 5 بعدی وارد کرده بود
        raise ValueError("Unsupported shape") # ارور می دهیم چون این تابع فقط بردار و ماتریس و تنسور می سازد
    
def random_vector(length: int, scale: float = 0.1) -> List[float]:
    """این تابع یک بردار تصادفی با طول و مقیاس دلخواه می سازد که داده هایی که تولید می کند بین -scale و scale است"""
    return [(random.random() * 2 - 1) * scale for _ in range(length)] # هر سلول = می تونه یک عدد منفی یا مثبت باشه. نکته : هر چقدر عدد تصادفی تولید شده بزرگ تر باشه جواب مثبت در می آید و هر چقدر عدد تصادفی تولید شده کوچکتر باشه جواب منفی در می آید و اگر عدد تصادفی تولید شده 0.5 باشه اون وقت میشه 0


def random_matrix(rows: int, cols: int, scale: float = 0.1) -> List[List[float]]: # این تابع برای ساخت وزن های اولیه است
    """این تابع یک ماتریس تصادفی با سطر و ستون مقیاس دلخواه می سازد که داده هایی که تولید می کند بین -scale و scale است"""
    return [[(random.random() * 2 - 1) * scale # هر سلول = می تونه یک عدد منفی یا مثبت باشه. نکته : هر چقدر عدد تصادفی تولید شده بزرگ تر باشه جواب مثبت در می آید و هر چقدر عدد تصادفی تولید شده کوچکتر باشه جواب منفی در می آید و اگر عدد تصادفی تولید شده 0.5 باشه اون وقت میشه 0
                for _ in range(cols)] # حلقه زدن روی تعداد ستون ها
                for _ in range(rows) # حلقه زدن روی تعداد سطر ها
        ]
    # اگر scale را مقداری خیلی بزرگ یا خیلی کوچک  بذاریم اونوقت داده هامون خیلی بزرگ یا خیلی کوچک میشن

def full(shape: Tuple[int, ...], fill_value: int) -> List[List[float]]:
    """این تابع یک آرایه پر شده با مقدار دلخواه بر می گرداند با شکل دلخواه"""
    fill_value = float(fill_value)

    if len(shape) == 1: # بررسی کردن اینکه shape تک بعدی باشد
        return [fill_value for _ in range(shape[0])] # ساخت لیستی از fill_value ها به طول اولین عضو تاپل shape با استفاده از List Comprehension
    elif len(shape) == 2: # بررسی کردن اینکه shape دو بعدی باشد
        return [[fill_value for _ in range(shape[1])] for _ in range(shape[0])] # لیستی از لیست fill_value ها به ابعاد (shape[0], shape[1])
    elif len(shape) == 3:
        return [[[fill_value for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])] # لیستی از لیستی از لیست fill_value ها به ابعاد (shape[0], shape[1], shape[2])
    else: # اگر کاربر shapeی وارد کرده بود که طولش بیشتر از 3 بود مثلا 4 بعدی یا 5 بعدی وارد کرده بود
        raise ValueError("Unsupported shape") # ارور می دهیم چون این تابع فقط بردار و ماتریس و تنسور می سازد

def matvec_mul(mat: List[List[float]], vec: List[float]) -> List[float]: # این تابع یک ماتریس و بردار را طبق قانون ریاضی اش ضرب می کند
    if len(mat[0]) != len(vec): # بررسی کردن اینکه طول ها با هم برابر نباشند
        raise ValueError("باید تعداد ستون های بردار و ماتریس برابر باشد.") # خطا می دهیم چون در دو حالت بزرگی و کوچکی یکی از طول ها کار اشتباه انجام می شود
    """این تابع یک ماتریس و یک بردار  با هر طول دلخواهی از شما می گیره و حاصل ضرب آنها را به شما می ده و باید تعداد ستون های بردار و ماتریس با هم برابر باشه"""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))] # این میگه عضو با سطر x و در اول ستون 1 در ماتریس ضرب میشه با عضو با اندیس 1 بردار ما و به همین ترتیب هر سطر که تموم شد میریم سطر بعدی و دوباره کل ستون های آن و بعد ...

def vecs_add(*vecs):
    if not vecs: # بررسی اینکه تاپل بردار خالی باشد
        raise ValueError("برای انجام این عملیات حداقل یک بردار لازم است.")

    it = iter(vecs) # ساخت یک Iterator از بردار ها
    n = len(next(it)) # گرفتن طول آیتم بعدی این Iterator که اولین عضو Iterator را بر می گردونه

    for v in it: # حلقه زدن روی Iterator از عضو دوم به صورت اتوماتیک
        if len(v) != n: # اگر طول یکی از بردار ها با اولی برابر نبود
            raise ValueError("باید تمام بردار ها طول یکسانی داشته باشند")

    return [sum(xs) for xs in zip(*vecs)] # جمع تک به تک اعضای متناظر در هر بردار

def layer_norm(vec: List[float], eps: float = 1e-5) -> List[float]: # این تابع یک بردار را نرمال می کند و کمک می کند شبکه پایدارتر بشه و سریعتر آموزش ببینه
    mean = sum(vec) / len(vec) # گرفتن میانگین
    variance = sum((x - mean) ** 2 for x in vec) / len(vec) # گرفتن واریانس طبق فرمول ریاضی اش
    return [(x - mean) / math.sqrt(variance + eps) for x in vec] # هر عضو داخل  بردار برابر عضو منهای میانگین تقسیم بر رادیکال واریانس + اپسیلون که یک عدد خیلی کوچک است و کاربردش اینه که اگر واریانس صفر شد مثل وقتی که همه داده ها یکسان باشند از تقسیم بر صفر جلوگیری کنیم

class LSTMCell: # این کلاس نشان دهنده یک سلول LSTM است
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size # تعداد ویژگی(feature) های ورودی
        self.hidden_size = hidden_size # تعداد نورون های مخفی در سلول یعنی ابعاد حالت مخغی

        # مقدار دهی اولیه وزن ها و پارامتر ها
        scale = math.sqrt(2 / (input_size + hidden_size)) # این متغیر مقدار نرمال برای وزن ها است که بر اساس اندازه ورودی محاسبه میشود و برای بهبود آموزش است که کمک می کند وزن ها در بازه مناسب قرار گیرند و این فرمول به Xavier معروف است
        
        self.W = random_matrix(4 * hidden_size, input_size, scale)
        self.U = orthogonal_matrix(4 * hidden_size, hidden_size)
        self.b = zeros((4 * hidden_size))
        self.b[self.hidden_size:2*self.hidden_size] = full((hidden_size), 1)

        # peephole connections یا دروازه های چشم درشت یا ارتباط های پیه پول
        # توضیحات : این ها وزن هایی هستند که به peephole connections معروف اند  که مستقیم به حالت سلولی وصل می شوند تا در تصمیم گیری دروازه ها کمک کنند
        self.V_i = random_vector(hidden_size, scale) # ورودی
        self.V_f = random_vector(hidden_size, scale) # فراموشی
        self.V_o = random_vector(hidden_size, scale) # خروجی

        # states یا حالت های اولیه
        # self.h = zeros((hidden_size)) # حالت مخفی اولیه که از صفر شروع می کنیم
        # self.c = zeros((hidden_size)) # حالت سلولی اولیه که از صفر شروع می کنیم دوباره
        # ما نمی خواهیم که سلول state را در خودش قفل کنه چون باعث میشه :
        #       1. نتونیم راحت sequence پردازش کنیم
        #       2. BPTT بعدا سخت بشه 
        #       3. کنترل حافظه مثل reset و detach نداریم

        # سلول باید state را بگیرد و state بدند

    def forward(self, x: List[float], h_prev: List[float], c_prev: List[float]) -> Tuple[List[float], List[float]]: # این متد برای پردازش یک توکن ورودی است
        """
        این متد برای پردازش یک توکن ورودی است که به جای اینکه از 8 تا matvec_mul استفاده کند از دو تا برای performance بهتر استفاده می کند
        
        :param self: شی فعلی از این کلاس
        :param x: لیستی از عدد های به عنوان ویژگی های ورودی
        :type x: List[float]
        :return: لیستی از عدد های نشان دهنده حالت مخفی بعدی
        :rtype: List[float]
        """
        Wx = matvec_mul(self.W, x) # ضرب ماتریس وزن ورودی در ورودی
        Uh = matvec_mul(self.U, h_prev) # ضرب ماتریس وزن حالت مخفی قبلی در حالت مخفی قبلی

        z = vecs_add(Wx, Uh, self.b) # جمع دو ماتریس بالا و بایاس ها

        H = self.hidden_size # طول حالت مخفی

        z_i = layer_norm(z[0:H])
        z_f = layer_norm(z[H:2*H])
        z_o = layer_norm(z[2*H:3*H])
        z_g = layer_norm(z[3*H:4*H])

        # input gate
        # این دروازه می گه که چقدر اطلاعات جدید وارد شود
        # i = [sigmoid(a + b + c * d + e) # تابع فعال سازی برای تنظیم مقدار دروازه بین 0 و 1
        #     for a, b, c, d, e in zip(
        #         matvec_mul(self.W_i, x), # ضرب ماتریس وزن ورودی در ورودی
        #         matvec_mul(self.U_i, self.h), # وزن های مربوط به حالت مخفی قبلی  و تاثیر آن بر دروازه ورودی را محاسبه می کند.
        #         self.V_i, # این مقادیر نرون های یادآور هستند یعنی حالت سلول قبلی را مستقیما وارد دروازه می کنند تا در تصمیم گیری اثرگذار باشد. به زبان ساده می گویند چقدر از حافظه بلند مدت(c) در تصمیم گیری دخیل باشد
        #         self.c, # حالت سلول قبلی که اطلاعات بلندمدت را نگه می دارد
        #         self.b_i # بایاس یا دگرگونی پایه که کمک می کند که مدل بهتر و سریعتر آموزش ببینه
        # )]
        # حالت بالا low performance دارد

        # input gate
        # این دروازه می گه که چقدر اطلاعات جدید وارد شود
        i = [sigmoid(z_i[j] + self.V_i[j] * c_prev[j]) for j in range(H)]

        # forget gate
        # توضیحات مثل دروازه ورودی فقط برای دروازه فراموشی و مشابه دروازه ورودی است ولی برای فراموش کردن معمولا
        f = [sigmoid(z_f[j] + self.V_f[j] * c_prev[j]) for j in range(H)]

        # cell candidate
        # این قسمت پیشنهاد می دهد چه مقدار اطلاعات جدید به وضعیت سلولی افزوده شود.
        g = [tanh(z_g[j]) for j in range(H)]

        # new cell state
        # حالت جدید سلول ترکیبی از حالت قبلی و اطلاعات جدید است
        # فاکتور فراموشی قسمت قبلی را نگه می دارد, و دروازه ورودی و پیشنهاد مقدار جدید را وارد می کنند.
        c = layer_norm([f[j] * c_prev[j] + i[j] * g[j] for j in range(H)])

        # output gate
        # این دروازه مشخص می کند چه مقدار اطلاعات وضعیت سلولی به حالت مخفی در خروجی داده شود.
        # در اینجا تاثیر وضعیت سلولی در خروجی در نظر گرفته شده است.
        o = [sigmoid(z_o[j] + self.V_o[j] * c[j]) for j in range(H)]

        # new hidden state
        h = [o[j] * tanh(layer_norm(c)[j]) for j in range(H)] # آپدیت کردن حالت مخفی فعلی با ضرب هر مقدار دروازه ورودی در تانژانت هایپربولیک هر عضو حالت سلولی
        return h, c # برگرداندن حالت مخفی و سلولی فعلی


# توضیحات تکمیلی :
#       می توانیم بگیم که self.c همان حافظه بلند مدت است
#       می توانیم بگیم که self.h همان حافظه کوتاه مدت است
#       فرق g و i به زبان ساده این است g می گوید چه چیزی وارد شود و i می گوید چقدرش وارد شود
#       o تصمیم می گیرد چقدر از حافظه خروجی داده شود و دیده شود ولی اکر نباشد همیشه همه ی حافظه همیشه لو می رود و این باعث over-sharing و کاهش قدرت مدل می شه
#       over-sharing یعنی اینکه شبکه بیش از حد لازم اطلاعات داخلی اش را به خروجی می ریزد و تمام c ها می ره داخل h و این به دلیل o ضعیف است
#       over-sharing وقتی اتفاق بیفته مدل هر چی یاد گرفته میگه و نمی دونه کی باید سکوت کنه و نتیجه اش میشه حرف های بی ربط و لو رفتن اطلاعات قدیمی و ناتوانی در تمرکز روی موضوع اصلی
#       ز حافظه بلند مدته که خصوصی است ولی h چیزی است که بیرون داده می شود