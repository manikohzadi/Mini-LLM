from rnn.rnn_cell import LSTMCell, zeros # اضافه کردن سلول LSTM و تابعی که بردار صفر می سازد

class LSTMLayer: # این کلاس یک لایه در شبکه عصبی بازگشتی LSTM است
    def __init__(self, input_size, hidden_size):
        self.cell = LSTMCell(input_size, hidden_size) # ساخت سلول LSTM با ابعاد ورودی و حالت مخفی
        self.hidden_size = hidden_size # ذخیره کردن ابعاد حالت مخفی در یک متغیر

    def init_state(self):
        """این تابع حالت های اولیه را تعریف می کند"""
        h = zeros(self.hidden_size) # حالت مخفی با برداری از صفر ها به طول حالت مخفی
        c = zeros(self.hidden_size) # حالت سلولی با برداری از صفر ها به طول حالت مخفی
        return h, c # برگرداندن حالت ها

    def forward(self, sequence):
        """پردازش دنباله ای از توکن های ورودی"""
        h, c = self.init_state() # تعریف حالت های مخفی و سلولی
        outputs = [] # خروجی ها

        for x in sequence: # حلقه زدن روی هر توکن ورودی
            h, c = self.cell.forward(x, h, c) # دادن هر توکن ورودی به علاوه state ها به سلول LSTM و گرفتن state های جدید
            outputs.append(h) # اضافه کردن حالت مخفی به حروجی

        return outputs, (h, c) # برگرداندن خروجی و state های جدید
    

# 🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴
# detach / reset