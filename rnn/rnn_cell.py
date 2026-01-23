import math # اضافه کردن کتابخانه math برای نوشتن activation functions مثل sigmoid و tanh
import random # اضافه کردن این کتابخانه برای ساخت وزن ها و بایاس ها ماتریس ها
from typing import List, Tuple # اضافه کردن این کتابخانه برای type annotations توابع

def sigmoid(x: float) -> float: # تابع فعال سازی sigmoid که ورودی را بین 0 و 1 نگه می دارد
    # در LSTM برای دروازه ها استفاده می شود تا مشخص شود چه مقدار اطلاعات عبور کند یعنی 0 به معنای اصلا عبور نکند است و 1 به معنای قطعا عبور کند است       
    """sigmoid activation function for Neural Networks and LSTM RNN"""
    return 1 / (1 + math.exp(-x)) # توضیحات در عکس function_formulas.png

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

def vec_add(v1: List[float], v2: List[float]) -> List[float]:
    """این تابع عضو های متناظر دو بردار را با هم جمع می کند و بردار حاصل را به شما می دهد"""
    if len(v1) != len(v2): # بررسی کردن اینکه طول ها با هم برابر نباشند
        raise ValueError("باید تعداد ستون های بردار و ماتریس برابر باشد.") # خطا می دهیم چون در دو حالت بزرگی و کوچکی یکی از طول ها کار اشتباه انجام می شود
    return [v1[i] + v2[i] for i in range(len(v1))] # جمع اعضای متناظر دو بردار

def layer_norm(vec: List[float], eps: float = 1e-5) -> List[float]: # این تابع یک بردار را نرمال می کند و کمک می کند شبکه پایدارتر بشه و سریعتر آموزش ببینه
    mean = sum(vec) / len(vec) # گرفتن میانگین
    variance = sum((x - mean) ** 2 for x in vec) / len(vec) # گرفتن واریانس طبق فرمول ریاضی اش
    return [(x - mean) / math.sqrt(variance + eps) for x in vec] # هر عضو داخل  بردار برابر عضو منهای میانگین تقسیم بر رادیکال واریانس + اپسیلون که یک عدد خیلی کوچک است و کاربردش اینه که اگر واریانس صفر شد مثل وقتی که همه داده ها یکسان باشند از تقسیم بر صفر جلوگیری کنیم

class LSTMCell: # این کلاس نشان دهنده یک سلول LSTM است
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size # تعداد ویژگی(feature) های ورودی
        self.hidden_size = hidden_size # تعداد نورون های مخفی در سلول یعنی ابعاد حالت مخغی

        # مقدار دهی اولیه وزن ها و پارامتر ها
        scale = math.sqrt(1 / input_size) # این متغیر مقدار نرمال برای وزن ها است که بر اساس اندازه ورودی محاسبه میشود و برای بهبود آموزش است که کمک می کند وزن ها در بازه مناسب قرار گیرند
        # Input Weights
        self.W_i = random_matrix(hidden_size, input_size, scale) # وزن برای دروازه ورودی به صورت تصادفی و با مقیاس نرمال و به شکل hidden_size * input_size
        self.W_f = random_matrix(hidden_size, input_size, scale) # وزن برای دروازه فراموشی به صورت تصادفی و با مقیاس نرمال و به شکل hidden_size * input_size
        self.W_o = random_matrix(hidden_size, input_size, scale) # وزن برای دروازه خروجی به صورت تصادفی و با مقیاس نرمال و به شکل hidden_size * input_size
        self.W_g = random_matrix(hidden_size, input_size, scale) # وزن برای مقدار جدید حافظه یا cell canditate به صورت تصادفی و با مقیاس نرمال و به شکل hidden_size * input_size
        # Recurrent Weights
        # چهار ماتریس دیگر برای ارتباطات حالت مخفی قبلی مثل حالت قبل
        self.U_i = random_matrix(hidden_size, hidden_size, scale) # ورودی
        self.U_f = random_matrix(hidden_size, hidden_size, scale) # فراموشی
        self.U_o = random_matrix(hidden_size, hidden_size, scale) # خروجی
        self.U_g = random_matrix(hidden_size, hidden_size, scale) # مقدار جدید حافظه یا cell canditate

        # peephole connections یا دروازه های چشم درشت یا ارتباط های پیه پول
        # توضیحات : این ها وزن هایی هستند که به peephole connections معروف اند  که مستقیم به حالت سلولی وصل می شوند تا در تصمیم گیری دروازه ها کمک کنند
        self.V_i = random_vector(hidden_size, scale) # ورودی
        self.V_f = random_vector(hidden_size, scale) # فراموشی
        self.V_o = random_vector(hidden_size, scale) # خروجی

        # biases
        self.b_i = zeros((hidden_size)) # ورودی
        self.b_f = full((hidden_size), 1.0) # فراموشی : forget bias = 1
        # نکته مهم در مورد بایاس های فراموشی :
        #   مقدار اولیه اش 1 است که به شبکه کمک می کند تا از ذخیره سازی اطلاعات جدید راحت تر استفاده کند
        self.b_o = zeros((hidden_size)) # خروجی
        self.b_g = zeros((hidden_size)) # cell canditate

        # states یا حالت های اولیه
        self.h = zeros((hidden_size)) # حالت مخفی اولیه که از صفر شروع می کنیم
        self.c = zeros((hidden_size)) # حالت سلولی اولیه که از صفر شروع می کنیم دوباره

    def forward(self, x: List[float]) -> List[float]: # این متد برای پردازش یک توکن ورودی است
        """
        این متد برای پردازش یک توکن ورودی است
        
        :param self: شی فعلی از این کلاس
        :param x: لیستی از عدد های به عنوان ویژگی های ورودی
        :type x: List[float]
        :return: لیستی از عدد های نشان دهنده حالت مخفی بعدی
        :rtype: List[float]
        """
        # input gate
        # این دروازه می گه که چقدر اطلاعات جدید وارد شود
        i = [sigmoid(a + b + c * d + e) # تابع فعال سازی برای تنظیم مقدار دروازه بین 0 و 1
            for a, b, c, d, e in zip(
                matvec_mul(self.W_i, x), # ضرب ماتریس وزن ورودی در ورودی
                matvec_mul(self.U_i, self.h), # وزن های مربوط به حالت مخفی قبلی  و تاثیر آن بر دروازه ورودی را محاسبه می کند.
                self.V_i, # این مقادیر نرون های یادآور هستند یعنی حالت سلول قبلی را مستقیما وارد دروازه می کنند تا در تصمیم گیری اثرگذار باشد. به زبان ساده می گویند چقدر از حافظه بلند مدت(c) در تصمیم گیری دخیل باشد
                self.c, # حالت سلول قبلی که اطلاعات بلندمدت را نگه می دارد
                self.b_i # بایاس یا دگرگونی پایه که کمک می کند که مدل بهتر و سریعتر آموزش ببینه
        )]
        # forget gate
        # توضیحات مثل دروازه ورودی فقط برای دروازه فراموشی و مشابه دروازه ورودی است ولی برای فراموش کردن معمولا
        f = [sigmoid(a + b + c * d + e) for a, b, c, d, e in zip(
            matvec_mul(self.W_f, x),
            matvec_mul(self.U_f, self.h),
            self.V_f,
            self.c,
            self.b_f
        )]
        # cell candidate
        # این قسمت پیشنهاد می دهد چه مقدار اطلاعات جدید به وضعیت سلولی افزوده شود.
        g = [tanh(a + b + e) # tanh تابع فعال سازی است که مقدارش بین -1 و 1 است.
                for a, b, e in zip(
                matvec_mul(self.W_g, x), # ضرب وزن ها وروی مقدار جدید حافظه در ورودی
                matvec_mul(self.U_g, self.h), # ضرب وزن recurrent در hidden state قبلی برای محاسبه contribution به gate یا canditate
                self.b_g # بایاس های cell canditate
        )]
        # new cell state
        # حالت جدید سلول ترکیبی از حالت قبلی و اطلاعات جدید است
        # فاکتور فراموشی قسمت قبلی را نگه می دارد, و دروازه ورودی و پیشنهاد مقدار جدید را وارد می کنند.
        self.c = [f_t * c_prev + i_t * g_t for f_t, c_prev, i_t, g_t in zip(f, self.c, i, g)]
        # output gate
        # این دروازه مشخص می کند چه مقدار اطلاعات وضعیت سلولی به حالت مخفی در خروجی داده شود.
        # در اینجا تاثیر وضعیت سلولی در خروجی در نظر گرفته شده است.
        o = [sigmoid(a + b + c_t * v + e) for a, b, c_t, v, e in zip(
            matvec_mul(self.W_o, x),
            matvec_mul(self.U_o, self.h),
            self.c,
            self.V_o,
            self.b_o
        )]
        # new hidden state
        self.h = [o_t * tanh(c_t) for o_t, c_t in zip(o, self.c)] # آپدیت کردن حالت مخفی فعلی با ضرب هر مقدار دروازه ورودی در تانژانت هایپربولیک هر عضو حالت سلولی
        return self.h # برگرداندن حالت مخفی فعلی 


# توضیحات تکمیلی :
#       می توانیم بگیم که self.c همان حافظه بلند مدت است
#       می توانیم بگیم که self.h همان حافظه کوتاه مدت است
#       فرق g و i به زبان ساده این است g می گوید چه چیزی وارد شود و i می گوید چقدرش وارد شود
#       o تصمیم می گیرد چقدر از حافظه خروجی داده شود و دیده شود ولی اکر نباشد همیشه همه ی حافظه همیشه لو می رود و این باعث over-sharing و کاهش قدرت مدل می شه
#       over-sharing یعنی اینکه شبکه بیش از حد لازم اطلاعات داخلی اش را به خروجی می ریزد و تمام c ها می ره داخل h و این به دلیل o ضعیف است
#       over-sharing وقتی اتفاق بیفته مدل هر چی یاد گرفته میگه و نمی دونه کی باید سکوت کنه و نتیجه اش میشه حرف های بی ربط و لو رفتن اطلاعات قدیمی و ناتوانی در تمرکز روی موضوع اصلی
#       ز حافظه بلند مدته که خصوصی است ولی h چیزی است که بیرون داده می شود