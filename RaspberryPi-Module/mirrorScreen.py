# requirements
# requests, feedparser, traceback, Pillow
from tkinter import *
import locale
import threading
import time
import requests
import traceback
import feedparser
from bs4 import BeautifulSoup
from PIL import Image, ImageTk
from contextlib import contextmanager
import firebase_admin
from firebase_admin import credentials, firestore
import random
from PyQt5.QtCore import QTimer

import faceAuth.authentication

LOCALE_LOCK = threading.Lock()

ui_locale = ''  # e.g. 'fr_FR' fro French, '' as default
time_format = 24  # 12 or 24
date_format = "%b %d, %Y"  # check python doc for strftime() for options
news_country_code = 'us'
xlarge_text_size = 94
large_text_size = 48
medium_text_size = 18
small_text_size = 12
medium_small_text_size = 16

uniqueUserId = 'VQOCOsQlPKN6llexkFMua4ZtSgz1'

cred = credentials.Certificate("C:/Users/ZENBOOK/Raspberry-Pi-Module/faceAuth/serviceAccountKey.json")
app = firebase_admin.initialize_app(cred)
db = firestore.client()

document = db.collection("Person").document(uniqueUserId)

data = document.get().to_dict()
cityName = data["country"]

# Widgets that are coming from database
# They are saved from Smart Mirror Application
time_widget = data["time_widget"]
weather_widget= data["weather_widget"]
motivational_widget = data["motivational_widget"]
date_widget = data["date_widget"]
daily_widget= data["daily_widget"]
exchange_widget = data["exchange_widget"]
reminder_widget= data["reminder_widget"]
welcome_widget = data["welcome_widget"]
food_widget= data["food_widget"]


@contextmanager
def setlocale(name):  # thread proof function to work with locale
    with LOCALE_LOCK:
        saved = locale.setlocale(locale.LC_ALL)
        try:
            yield locale.setlocale(locale.LC_ALL, name)
        finally:
            locale.setlocale(locale.LC_ALL, saved)


# maps open weather icons to
# icon reading is not impacted by the 'lang' parameter


class Clock(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')
        # initialize time label
        self.time1 = ''
        self.timeLbl = Label(self, font=('Segoe UI Light', large_text_size, "bold"), fg="white", bg="black")
        self.timeLbl.pack(side=TOP, anchor=W)
        # initialize day of week
        self.day_of_week1 = ''
        self.dayOWLbl = Label(self, text=self.day_of_week1, font=('Segoe UI Light', medium_small_text_size), fg="white",
                              bg="black")
        self.dayOWLbl.pack(side=TOP, anchor=W)
        # initialize date label
        self.date1 = ''
        self.dateLbl = Label(self, text=self.date1, font=('Segoe UI Light', medium_small_text_size), fg="white",
                             bg="black")
        self.dateLbl.pack(side=TOP, anchor=W)
        self.tick()

    def tick(self):
        with setlocale(ui_locale):
            if time_format == 12:
                time2 = time.strftime('%I:%M %p')  # hour in 12h format
            else:
                time2 = time.strftime('%H:%M')  # hour in 24h format

            day_of_week2 = time.strftime('%A')
            date2 = time.strftime(date_format)
            # if time string has changed, update it
            if time2 != self.time1:
                self.time1 = time2
                self.timeLbl.config(text=time2)
            if day_of_week2 != self.day_of_week1:
                self.day_of_week1 = day_of_week2
                self.dayOWLbl.config(text=day_of_week2)
            if date2 != self.date1:
                self.date1 = date2
                self.dateLbl.config(text=date2)
            # calls itself every 200 milliseconds
            # to update the time display as needed
            # could use >200 ms, but display gets jerky
            self.timeLbl.after(200, self.tick)




class Weather(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        city = cityName+'weather'  # this will be from the db
        res = requests.get(
            f'https://www.google.com/search?q={city}&oq={city}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8',
            headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        info = soup.select('#wob_dc')[0].getText().strip()
        weather = soup.select('#wob_tm')[0].getText().strip()
        # weather
        self.weather = weather + "°C"
        self.weatherLbl = Label(self, text=self.weather, font=('Segoe UI Light', medium_text_size), fg="white",
                                bg='black')
        self.weatherLbl.pack(side=TOP, anchor=E)
        # weather information
        self.weatherInfo = info
        self.weatherInfoLbl = Label(self, text=self.weatherInfo, font=('Segoe UI Light', medium_text_size), fg="white",
                                    bg='black')
        self.weatherInfoLbl.pack(side=TOP, anchor=E)


class News(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.config(bg='black')
        self.title = 'News'  # 'News' is more internationally generic
        self.newsLbl = Label(self, text=self.title, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.newsLbl.pack(side=TOP, anchor=W)
        self.headlinesContainer = Frame(self, bg="black")
        self.headlinesContainer.pack(side=TOP)
        self.get_headlines()

    def get_headlines(self):
        try:
            # remove all children
            for widget in self.headlinesContainer.winfo_children():
                widget.destroy()
            if news_country_code == None:
                headlines_url = "https://news.google.com/news?ned=us&output=rss"
            else:
                headlines_url = "https://news.google.com/news?ned=%s&output=rss" % news_country_code

            feed = feedparser.parse(headlines_url)

            for post in feed.entries[0:5]:
                headline = NewsHeadline(self.headlinesContainer, post.title)
                headline.pack(side=TOP, anchor=W)
        except Exception as e:
            traceback.print_exc()
            print("Error: %s. Cannot get news." % e)

        self.after(600000, self.get_headlines)


class NewsHeadline(Frame):
    def __init__(self, parent, event_name=""):
        Frame.__init__(self, parent, bg='black')

        image = Image.open("Newspaper1.png")
        image = image.resize((20, 20), Image.ANTIALIAS)
        image = image.convert('RGB')
        photo = ImageTk.PhotoImage(image)

        self.iconLbl = Label(self, bg='black', image=photo)
        self.iconLbl.image = photo
        self.iconLbl.pack(side=LEFT, anchor=N)

        self.eventName = event_name
        self.eventNameLbl = Label(self, text=self.eventName, font=('Segoe UI Light', small_text_size), fg="white",
                                  bg="black")
        self.eventNameLbl.pack(side=LEFT, anchor=N)


class WelcomeMessage(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')
        self.title = 'Welcome Message'
        self.welcome_frm = Frame(self, bg="black")
        self.welcome_frm.pack(side=TOP, anchor=W)

        self.nameLbl = Label(self.welcome_frm, font=('Segoe UI Light', medium_small_text_size), fg="white", bg="black")
        self.nameLbl.pack(side=TOP, anchor=N)

        self.get_welcome_message()

    def get_welcome_message(self):
        user_name = ''

        name_value = data["userName"]
        user_name = ("Hello " + name_value + "!")

        self.nameLbl.config(text=user_name)


class Stock(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')

        self.gold = ''
        self.dollar = ''
        self.euro = ''
        self.gbp = ''

        self.stock_frm = Frame(self, bg="black")
        self.stock_frm.pack(side=TOP, anchor=E)

        self.dollarLbl = Label(self.stock_frm, font=('Segoe UI Light', medium_small_text_size), fg="white", bg="black")
        self.dollarLbl.pack(side=TOP, anchor=E)

        self.euroLbl = Label(self.stock_frm, font=('Segoe UI Light', medium_small_text_size), fg="white", bg="black")
        self.euroLbl.pack(side=TOP, anchor=E)

        self.gbpLbl = Label(self.stock_frm, font=('Segoe UI Light', medium_small_text_size), fg="white", bg="black")
        self.gbpLbl.pack(side=TOP, anchor=E)

        self.goldLbl = Label(self.stock_frm, font=('Segoe UI Light', medium_small_text_size), fg="white", bg="black")
        self.goldLbl.pack(side=TOP, anchor=E)

        self.get_stock_value()

    def get_stock_value(self):
        try:
            dollar_print = ''
            euro_print = ''
            gold_print = ''
            gbp_print = ''
            r = requests.get("https://www.doviz.com")
            soup = BeautifulSoup(r.content, "html.parser")
            stockdata = soup.find_all('span', attrs={'class': 'value'})

            goldval = stockdata[0].text
            dolarval = stockdata[1].text
            euroval = stockdata[2].text
            gbpval = stockdata[3].text

            dollar_print = ("1$: " + dolarval + " TL")
            euro_print = ("1€: " + euroval + " TL")
            gbp_print = ("1£: " + gbpval + " TL")
            gold_print = ("Gold: " + goldval + " TL")

            if self.gold != None:
                self.gold = gold_print
                self.goldLbl.config(text=gold_print)

            if self.dollar != None:
                self.dollar = dollar_print
                self.dollarLbl.config(text=dollar_print)

            if self.euro != None:
                self.euro = euro_print
                self.euroLbl.config(text=euro_print)

            if self.gbp != None:
                self.gbp = gbp_print
                self.gbpLbl.config(text=gbp_print)

        except IOError:
            print('no internet')


class foodReq(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')
        self.foodLbl = Label(self, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.foodLbl.pack(side=TOP, anchor=E)
        self.fheadlinesContainer = Frame(self, bg="black")
        self.fheadlinesContainer.pack(side=TOP)
        self.get_foodheadlines()

    def get_foodheadlines(self):
        filename = 'C:/Users/ZENBOOK/Raspberry-Pi-Module/db/food1.csv'
        for widget in self.fheadlinesContainer.winfo_children():
            widget.destroy()
        with open(filename) as f:
            reader = f.readlines()
            chosen_row = random.choice(reader)
            self.chosen_row = chosen_row
            foodrec = FoodHeadline(self.fheadlinesContainer, chosen_row)
            foodrec.pack(side=TOP, anchor=E)

class Motiva(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')

        self.quotetext = ''

        self.quote_frm = Frame(self, bg="black")
        self.quote_frm.pack(side=LEFT, anchor=W)

        self.quoteLbl = Label(self.quote_frm, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.quoteLbl.pack(side=LEFT, anchor=W)

        self.get_MotivationalHeadlines()

    def get_MotivationalHeadlines(self):

        filename = 'C:/Users/ZENBOOK/Raspberry-Pi-Module/db/quotes.csv'
        for widget in self.quoteLbl.winfo_children():
            widget.destroy()
        with open(filename) as f:
            reader = f.readlines()
            chosen_row = random.choice(reader)
            i = 0
            while chosen_row[i]!= ",":
                i=i+1

            if len(chosen_row[i+1:]) <= 100:
                quote_print= chosen_row[i+1:]+"\n"+"-"+chosen_row[:i]

                if self.quotetext != None:
                    self.quotetext = quote_print
                    self.quoteLbl.config(text=quote_print)
            else:
                self.get_MotivationalHeadlines()


class FoodHeadline(Frame):
    def __init__(self, parent, chosen_row=""):
        Frame.__init__(self, parent, bg='black')

        image = Image.open("foodicon.png")
        image = image.resize((20, 20), Image.ANTIALIAS)
        image = image.convert('RGB')
        photo = ImageTk.PhotoImage(image)

        self.ficonLbl = Label(self, bg='black', image=photo)
        self.ficonLbl.image = photo
        self.ficonLbl.pack(side=LEFT, anchor=NW)

        self.chosen_row = chosen_row
        self.feventNameLbl = Label(self, text=self.chosen_row, font=('Segoe UI Light', small_text_size), fg="white",
                                   bg="black")
        self.feventNameLbl.pack(side=LEFT, anchor=NW)


class Reminder(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')

        reminder1 = data["reminder1"]
        reminder2 = data["reminder2"]
        reminder3 = data["reminder3"]

        self.r1 = ("-" + reminder1)

        self.r1Lbl = Label(self, text=self.r1, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.r1Lbl.pack(side=TOP, anchor=E)
        self.r2 = ("-" + reminder2)

        self.r2Lbl = Label(self, text=self.r2, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.r2Lbl.pack(side=TOP, anchor=E)
        self.r3 = ("-" + reminder3)

        self.r3Lbl = Label(self, text=self.r3, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.r3Lbl.pack(side=TOP, anchor=E)

class DefaultClock(Frame):
        def __init__(self, parent, *args, **kwargs):
            Frame.__init__(self, parent, bg='black')
            # initialize time label
            self.time1 = ''
            self.timeLbl = Label(self, font=('Segoe UI Light', large_text_size, "bold"), fg="white", bg="black")
            self.timeLbl.pack(side=TOP, anchor=W)
            # initialize day of week
            self.day_of_week1 = ''
            self.dayOWLbl = Label(self, text=self.day_of_week1, font=('Segoe UI Light', medium_small_text_size),
                                  fg="white",
                                  bg="black")
            self.dayOWLbl.pack(side=TOP, anchor=W)
            # initialize date label
            self.date1 = ''
            self.dateLbl = Label(self, text=self.date1, font=('Segoe UI Light', medium_small_text_size), fg="white",
                                 bg="black")
            self.dateLbl.pack(side=TOP, anchor=W)
            self.tick()

        def tick(self):
            with setlocale(ui_locale):
                if time_format == 12:
                    time2 = time.strftime('%I:%M %p')  # hour in 12h format
                else:
                    time2 = time.strftime('%H:%M')  # hour in 24h format

                day_of_week2 = time.strftime('%A')
                date2 = time.strftime(date_format)
                # if time string has changed, update it
                if time2 != self.time1:
                    self.time1 = time2
                    self.timeLbl.config(text=time2)
                if day_of_week2 != self.day_of_week1:
                    self.day_of_week1 = day_of_week2
                    self.dayOWLbl.config(text=day_of_week2)
                if date2 != self.date1:
                    self.date1 = date2
                    self.dateLbl.config(text=date2)
                # calls itself every 200 milliseconds
                # to update the time display as needed
                # could use >200 ms, but display gets jerky
                self.timeLbl.after(200, self.tick)


class FullscreenWindow:
    def __init__(self):
        self.tk = Tk()
        self.tk.configure(background='black')
        self.topFrame = Frame(self.tk, background='black')
        self.bottomFrame = Frame(self.tk, background='black')
        self.middleFrame = Frame(self.tk, background='black')
        self.middle2Frame = Frame(self.tk, background='black')
        self.topFrame.pack(side=TOP, fill=BOTH, expand=YES)
        self.bottomFrame.pack(side=BOTTOM, fill=BOTH, expand=YES)
        self.middleFrame.pack(side=LEFT, fill=BOTH, expand=YES)
        self.middle2Frame.pack(side=RIGHT, fill=BOTH, expand=YES)

        # Necessary for maximizing the screen...
        self.tk.attributes('-fullscreen',
                           True)
        self.frame = Frame(self.tk)
        self.frame.pack()
        self.state = False
        self.tk.bind("<F11>", self.toggle_fullscreen)
        self.tk.bind("<Escape>", self.end_fullscreen)

        if(time_widget== 'true'):
            # clock
            self.clock = Clock(self.topFrame)
            self.clock.pack(side=LEFT, anchor=NW, padx=50, pady=50)


        # weather
        if (weather_widget == 'true'):
            self.weather = Weather(self.topFrame)
            self.weather.pack(side=RIGHT, anchor=NE, padx=50, pady=60)


        # news
        if (daily_widget == 'true'):
            self.news = News(self.bottomFrame)
            self.news.pack(side=LEFT, anchor=SW, padx=50, pady=50)


        # reminder
        if (reminder_widget == 'true'):
            self.reminder = Reminder(self.middle2Frame)
            self.reminder.pack(side=RIGHT, padx=(0, 50), pady=50)


        # motivational
        if (motivational_widget == 'true'):
            self.quote = Motiva(self.middleFrame)
            self.quote.pack(anchor=SW, padx=50, pady=50)


        # stock
        if (exchange_widget == 'true'):
            self.stocks = Stock(self.bottomFrame)
            self.stocks.pack(side=RIGHT, anchor=SE, padx=50, pady=50)


        # foodReq
        if (food_widget == 'true'):
            self.food = foodReq(self.middleFrame)
            self.food.pack(side=LEFT, anchor=W, padx=50, pady=0)


        # welcome message
        if (welcome_widget == 'true'):
            self.welcome = WelcomeMessage(self.topFrame)
            self.welcome.pack(padx=0, pady=50)



    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.tk.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.tk.attributes("-fullscreen", False)
        return "break"


if __name__ == '__main__':
    w = FullscreenWindow()
    w.tk.mainloop()
    # print(faceAuth.authentication.name)