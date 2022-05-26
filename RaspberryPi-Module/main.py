import cv2
import numpy as np
import os
import face_recognition as fr
from firebase import Firebase
from google.cloud import storage
from tkinter import *
import locale
import threading
import time
import requests
import traceback
import feedparser
from bs4 import BeautifulSoup
import firebase_admin
from firebase_admin import credentials, firestore
import random
from PIL import Image, ImageTk
from contextlib import contextmanager
import math
from datetime import date, timedelta

LOCALE_LOCK = threading.Lock()



ui_locale = ''  # e.g. 'fr_FR' fro French, '' as default
time_format = 24  # 12 or 24
date_format = "%b %d, %Y"  # check python doc for strftime() for options
news_country_code = 'us'
xlarge_text_size = 94
large_text_size = 48
medium_text_size = 24
small_text_size = 12
medium_small_text_size = 16

cred = credentials.Certificate("../faceAuth/serviceAccountKey.json")  #path to your service account key
app = firebase_admin.initialize_app(cred)  #initialize firebase
db = firestore.client()  #initialize firestore
api_key = "c8759b62d883f490862a5363692aef41"  #api key for openweathermap
base_url = "http://api.openweathermap.org/data/2.5/weather?"  #api url


@contextmanager
def setlocale(name):  # thread proof function to work with locale
    with LOCALE_LOCK:
        saved = locale.setlocale(locale.LC_ALL)
        try:
            yield locale.setlocale(locale.LC_ALL, name)
        finally:
            locale.setlocale(locale.LC_ALL, saved)




class authenticationModule:    #class for authentication

    def train_dataset(self):  # function to Train dataset

        config = {
            "apiKey": "AIzaSyATJUWiR18Mfc_Yrd4CTAqZVwn-pVpXnno",
            "authDomain": "signin-example-b3f10.firebaseapp.com",
            "databaseURL": "https://signin-example-b3f10-default-rtdb.firebaseio.com",
            "storageBucket": "signin-example-b3f10.appspot.com",
            "serviceAccount": "../faceAuth/serviceAccountKey.json"
        }

        firebase = Firebase(config)  # initialize firebase
        store = firebase.storage() # initialize firebase storage

        os.environ[
            "GOOGLE_APPLICATION_CREDENTIALS"] = "../faceAuth/serviceAccountKey.json"

        storage_client = storage.Client()
        bucket = "signin-example-b3f10.appspot.com"

        bucket = storage_client.get_bucket(bucket)
        blobs = list(bucket.list_blobs())
        for blob in blobs:
            store.child(blob.name).download("../RaspberryPi-Module/Train/" + blob.name[5:] + ".jpeg")

    def faceAuthentication(self):  # function to authenticate face
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # initialize camera
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        face_detector = cv2.CascadeClassifier(   # initialize face detector
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Loading the required haar-cascade xml classifier file
        count = 0
        t_end= time.time() + 60  # set time limit for authentication

        while (time.time()<t_end):

            ret, img = cam.read()  # read camera frame
            # img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            faces = face_detector.detectMultiScale(gray, 1.3,
                                                   5)  # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.

            for (x, y, w, h) in faces:
                count += 1
                cv2.imwrite("Test/compared.jpg",
                            gray[y:y + h, x:x + w])  # Save the captured image into the datasets folder

            if count >= 30:  # Take 30 face sample and stop video
                break

        cam.release() # turn off camera
        cv2.destroyAllWindows()  # remove window
        if count ==0:
            return "Unknown Person"
        path = "../RaspberryPi-Module/Train/"

        known_names = []  # Initialize known face name
        known_name_encodings = []       # Initialize known face encodings

        images = os.listdir(path) # Get the list of all the available images
        for _ in images:     # Loop over the list of all the available images
            image = fr.load_image_file(path + _)
            image_path = path + _
            encoding = fr.face_encodings(image)[0]
            known_name_encodings.append(encoding)
            known_names.append(os.path.splitext(os.path.basename(image_path))[0])      # Append the name of the image to the list of known names

        test_image = "../RaspberryPi-Module/Test/compared.jpg"    # Load the Test image

        image = cv2.imread(test_image) # Read the Test image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image to RGB

        face_locations = fr.face_locations(image)    # Find all the faces in the Test image
        face_encodings = fr.face_encodings(image, face_locations)       # Find all the face encodings in the Test image

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):    # Loop over all the faces in the Test image
            matches = fr.compare_faces(known_name_encodings, face_encoding, tolerance=0.5)   # Determine if the face is a match for the known face or not
            userID = "Unknown Person"

            face_distances = fr.face_distance(known_name_encodings, face_encoding)    # Find the face with the smallest distance to the Test face
            best_match = np.argmin(face_distances)   # Get the index of the face with the smallest distance to the Test face

            if matches[best_match]:      # Check if the face with the smallest distance to the Test face is a match
                userID = known_names[best_match]  # If so, get the name of the corresponding person

        dir = '../RaspberryPi-Module/Train/'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

        return userID



class Database(object):     #class for database
    def __init__(self, userID):
        document = db.collection("Person").document(userID)    # Get the document of the user
        data = document.get().to_dict()      # Get the data of the user

        self.defaultScreen = data["defaultScreen"]
        self.time_widget = data["time_widget"]
        self.weather_widget = data["weather_widget"]
        self.motivational_widget = data["motivational_widget"]
        self.date_widget = data["date_widget"]
        self.daily_widget = data["daily_widget"]
        self.exchange_widget = data["exchange_widget"]
        self.reminder_widget = data["reminder_widget"]
        self.welcome_widget = data["welcome_widget"]
        self.food_widget = data["food_widget"]
        self.cityName = data["country"]
        self.reminder1 = data["reminder1"]
        self.reminder2 = data["reminder2"]
        self.reminder3 = data["reminder3"]
        self.name_value = data["userName"]
        self.birthdate = data["birth"]
        star = data["birth"]
        star = str(star)
        firstThree = star[5:10] #take birtday in mm-dd format
        print(firstThree)
        self.birthdate = firstThree
        # today = date.today()
        # d1 = today.strftime("%m-%d")
        # print("d1 =", d1)

    def get_birthday(self):
        return self.birthdate

    def get_cityName(self):
        return self.cityName

    def get_time_widget(self):
        return self.time_widget

    def get_name_value(self):
        return self.name_value

    def get_weather_widget(self):
        return self.weather_widget

    def get_motivational_widget(self):
        return self.motivational_widget

    def get_date_widget(self):
        return self.date_widget

    def get_reminder1(self):
        return self.reminder1

    def get_reminder2(self):
        return self.reminder2

    def get_reminder3(self):
        return self.reminder3

    def get_food_widget(self):
        return self.food_widget

    def get_welcome_widget(self):
        return self.welcome_widget

    def get_reminder_widget(self):
        return self.reminder_widget

    def get_exchange_widget(self):
        return self.exchange_widget

    def get_daily_widget(self):
        return self.daily_widget

class Clock(Frame):        #class for clock
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
            self.timeLbl.after(200, self.tick)


class ClockDefault(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')
        # initialize time label
        self.time1 = ''
        self.timeLbl = Label(self, font=('Segoe UI Light', xlarge_text_size, "bold"), fg="white", bg="black")
        self.timeLbl.pack(side=TOP, anchor=W)
        # initialize day of week
        self.day_of_week1 = ''
        self.dayOWLbl = Label(self, text=self.day_of_week1, font=('Segoe UI Light', large_text_size), fg="white",
                              bg="black")
        self.dayOWLbl.pack(side=TOP, anchor=W)
        # initialize date label
        self.date1 = ''
        self.dateLbl = Label(self, text=self.date1, font=('Segoe UI Light', large_text_size), fg="white",
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
            self.timeLbl.after(200, self.tick)




#reference: https://www.codemag.com/Article/1511071/Building-a-Weather-App-using-OpenWeatherMap-and-AFNetworking
class Weather(Frame):          #class for weather
    def __init__(self, parent,mirror, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')
        cityName = mirror.get_cityName()
        complete_url = base_url + "appid=" + api_key + "&q=" + cityName  # url for weather api
        response = requests.get(complete_url) # get request
        x = response.json() # json format
        if x["cod"] != "404":
            y = x["main"]
            current_temperature = y["temp"]
            current_temperature = math.ceil(current_temperature - 273.15)
            max_celsius = y["temp_max"]
            max_celsius = math.ceil(max_celsius - 273.15)
            min_celsius = y["temp_min"]
            min_celsius = math.floor(min_celsius - 273.15)
            z = x["weather"]
            weather_description = z[0]["description"]
        else:
            print(" City Not Found ")

            # store the value corresponding
        self.weather = str(current_temperature) + "°C"     #temperature
        self.weatherLbl = Label(self, text=self.weather, font=('Segoe UI Light', medium_text_size), fg="white",
                                bg='black')
        self.weatherLbl.pack(side=TOP, anchor=E)

        self.weather = str(min_celsius) + "°C" + " / "  + str(max_celsius) + "°C"       #min and max temperature
        self.weatherLbl = Label(self, text=self.weather, font=('Segoe UI Light', medium_small_text_size), fg="white",
                                bg='black')
        self.weatherLbl.pack(side=TOP, anchor=E)

        # weather information
        self.weatherInfo = str(weather_description).capitalize()      #weather description
        self.weatherInfoLbl = Label(self, text=self.weatherInfo, font=('Segoe UI Light', medium_text_size), fg="white",
                                    bg='black')
        self.weatherInfoLbl.pack(side=TOP, anchor=E)


class News(Frame):       #class for news
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
            if news_country_code == None:     #if country code is not selected
                headlines_url = "https://news.google.com/news?ned=us&output=rss"
            else:
                headlines_url = "https://news.google.com/news?ned=%s&output=rss" % news_country_code      #url for news api

            feed = feedparser.parse(headlines_url)

            for post in feed.entries[0:5]:      # only show the first 5 posts
                headline = NewsHeadline(self.headlinesContainer, post.title)
                headline.pack(side=TOP, anchor=W)
        except Exception as e:
            traceback.print_exc()
            print("Error: %s. Cannot get news." % e)

        self.after(600000, self.get_headlines)


class NewsHeadline(Frame):      #class for news headlines image
    def __init__(self, parent, event_name=""):
        Frame.__init__(self, parent, bg='black')

        image = Image.open("../Assets/Newspaper.png")
        newsize = (20, 20)
        image = image.resize(newsize)
        image = image.convert('RGB')
        photo = ImageTk.PhotoImage(image)

        self.iconLbl = Label(self, bg='black', image=photo)
        self.iconLbl.image = photo
        self.iconLbl.pack(side=LEFT, anchor=N)

        self.eventName = event_name
        self.eventNameLbl = Label(self, text=self.eventName, font=('Segoe UI Light', small_text_size), fg="white",
                                  bg="black")
        self.eventNameLbl.pack(side=LEFT, anchor=N)


class WelcomeMessage(Frame):      #class for welcome message
    def __init__(self, parent,mirror, *args, **kwargs):

        Frame.__init__(self, parent, bg='black')

        self.title = 'Welcome Message'
        self.welcome_frm = Frame(self, bg="black")
        self.welcome_frm.pack(side=TOP, anchor=W)

        self.nameLbl = Label(self.welcome_frm, font=('Segoe UI Light', medium_small_text_size), fg="white", bg="black")
        self.nameLbl.pack(side=TOP, anchor=N)

        self.get_welcome_message(mirror)

    def get_welcome_message(self, mirror):
        birthdate = mirror.get_birthday()
        today = date.today()
        yesterday = today - timedelta(days=1)

        d1 = yesterday.strftime("%m-%d")
        name_value = mirror.get_name_value()       #get name value
        if(d1 == birthdate):
            user_name = ("Happy Birthday " + name_value + "!")
        else:
            user_name = ("Hello " + name_value + "!")

        self.nameLbl.config(text=user_name)


class Stock(Frame):     #class for stock
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
            r = requests.get("https://www.doviz.com")     #get stock value from doviz.com
            soup = BeautifulSoup(r.content, "html.parser")    #parse the html
            stockdata = soup.find_all('span', attrs={'class': 'value'})      #find all span tags with class value

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


class foodReq(Frame):     #class for food reccomendation
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')
        self.foodLbl = Label(self, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.foodLbl.pack(side=TOP, anchor=E)
        self.fheadlinesContainer = Frame(self, bg="black")
        self.fheadlinesContainer.pack(side=TOP)
        self.get_foodheadlines()

    def get_foodheadlines(self):
        filename = '../Database/Food.csv'  #get food headlines from csv file
        for widget in self.fheadlinesContainer.winfo_children():
            widget.destroy()
        with open(filename) as f:    #read csv file
            reader = f.readlines()
            chosen_row = random.choice(reader)
            self.chosen_row = chosen_row
            foodrec = FoodHeadline(self.fheadlinesContainer, chosen_row)
            foodrec.pack(side=TOP, anchor=E)


class MotivationalText(Frame):    #class for motivational text
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')

        self.quotetext = ''
        self.authortext = ''

        self.quote_frm = Frame(self, bg="black")
        self.quote_frm.pack(side=TOP, anchor=W)

        self.quoteLbl = Label(self.quote_frm, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.quoteLbl.pack(side=TOP, anchor=W)

        self.authorLbl = Label(self.quote_frm, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.authorLbl.pack(side=TOP, anchor=W)

        self.get_MotivationalHeadlines()

    def get_MotivationalHeadlines(self):

        filename = '../Database/Motivational.csv'  #get motivational text from csv file
        for widget in self.quoteLbl.winfo_children():
            widget.destroy()
        with open(filename) as f:
            reader = f.readlines()
            chosen_row = random.choice(reader)
            i = 0
            while chosen_row[i] != ",":
                i = i + 1

            if len(chosen_row[i + 1:]) <= 100:    #if the text is less than 100 characters
                self.str_text = chosen_row[i + 1:] + "\n" + "-" + chosen_row[:i] #add author name to the end of the text
                self.feventNameLbl = Label(self, text=self.str_text, font=('Segoe UI Light', small_text_size),
                                           fg="white",
                                           bg="black")
                self.feventNameLbl.pack(side=LEFT, anchor=NW)
            else:
                self.get_MotivationalHeadlines()


class FoodHeadline(Frame):   #class for food reccomendation image
    def __init__(self, parent, chosen_row=""):
        Frame.__init__(self, parent, bg='black')

        image = Image.open("../Assets/Foodicon.png") #open food icon image
        newsize = (20, 20)
        image = image.resize(newsize)
        image = image.convert('RGB')
        photo = ImageTk.PhotoImage(image)

        self.ficonLbl = Label(self, bg='black', image=photo)
        self.ficonLbl.image = photo
        self.ficonLbl.pack(side=LEFT, anchor=NW)

        self.chosen_row = chosen_row
        self.feventNameLbl = Label(self, text=self.chosen_row, font=('Segoe UI Light', small_text_size), fg="white",
                                   bg="black")
        self.feventNameLbl.pack(side=LEFT, anchor=NW)


class Reminder(Frame):   #class for reminders
    def __init__(self, parent,mirror, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')

        reminder1 = mirror.get_reminder1()  #get reminders from database
        reminder2 = mirror.get_reminder2()
        reminder3 = mirror.get_reminder3()

        self.r1 = ("-" + reminder1)

        self.r1Lbl = Label(self, text=self.r1, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.r1Lbl.pack(side=TOP, anchor=E)
        self.r2 = ("-" + reminder2)

        self.r2Lbl = Label(self, text=self.r2, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.r2Lbl.pack(side=TOP, anchor=E)
        self.r3 = ("-" + reminder3)

        self.r3Lbl = Label(self, text=self.r3, font=('Segoe UI Light', small_text_size), fg="white", bg="black")
        self.r3Lbl.pack(side=TOP, anchor=E)


class DefaultScreen():  #class for default screen
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

        # #Necessary for maximizing the screen...
        # self.tk.attributes('-fullscreen',
        #                    True)
        # self.frame = Frame(self.tk)
        # self.frame.pack()
        # self.state = False
        # self.tk.bind("<F11>", self.toggle_fullscreen)
        # self.tk.bind("<Escape>", self.end_fullscreen)


        self.clock = ClockDefault(self.middleFrame)
        self.clock.pack(side=TOP,  padx=50, pady=50)

    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.tk.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.tk.attributes("-fullscreen", False)
        return "break"

class FullscreenWindow(object): #class for fullscreen window
    def __init__(self,mirror):
        self.tk=Tk()

        self.tk.configure(background='black')
        self.topFrame = Frame(self.tk, background='black')     #top frame
        self.bottomFrame = Frame(self.tk, background='black') #bottom frame
        self.middleFrame = Frame(self.tk, background='black') #middle frame
        self.middle2Frame = Frame(self.tk, background='black') #middle2 frame
        self.topFrame.pack(side=TOP, fill=BOTH, expand=YES)   #pack top frame
        self.bottomFrame.pack(side=BOTTOM, fill=BOTH, expand=YES) #pack bottom frame
        self.middleFrame.pack(side=LEFT, fill=BOTH, expand=YES)     #pack middle frame
        self.middle2Frame.pack(side=RIGHT, fill=BOTH, expand=YES)   #pack middle2 frame

        #Necessary for maximizing the screen...
        # self.tk.attributes('-fullscreen',
        #                    True)
        # self.frame = Frame(self.tk)
        # self.frame.pack()
        # self.state = False
        # self.tk.bind("<F11>", self.toggle_fullscreen)
        # self.tk.bind("<Escape>", self.end_fullscreen)

        if(mirror.defaultScreen == 'true'):
            self.clock = ClockDefault(self.middleFrame)
            self.clock.pack(side=TOP, padx=50, pady=50)


        if (mirror.time_widget == 'true'):
            # clock
            self.clock = Clock(self.topFrame)
            self.clock.pack(side=LEFT, anchor=NW, padx=50, pady=50)

        # weather
        if (mirror.weather_widget == 'true'):
            self.weather = Weather(self.topFrame,mirror)
            self.weather.pack(side=RIGHT, anchor=NE, padx=50, pady=60)

        # news
        if (mirror.daily_widget == 'true'):
            self.news = News(self.bottomFrame)
            self.news.pack(side=LEFT, anchor=SW, padx=50, pady=50)

        # reminder
        if (mirror.reminder_widget == 'true'):
            self.reminder = Reminder(self.middle2Frame,mirror)
            self.reminder.pack(side=RIGHT, padx=(0, 50), pady=50)

        # motivational
        if (mirror.motivational_widget == 'true'):
            self.quote = MotivationalText(self.middleFrame)
            self.quote.pack(anchor=SW, padx=50, pady=70)

        # stock
        if (mirror.exchange_widget == 'true'):
            self.stocks = Stock(self.bottomFrame)
            self.stocks.pack(side=RIGHT, anchor=SE, padx=50, pady=50)

        # foodReq
        if (mirror.food_widget == 'true'):
            self.food = foodReq(self.middleFrame)
            self.food.pack(side=LEFT, anchor=W, padx=50, pady=0)

        # welcome message
        if (mirror.welcome_widget == 'true'):
            self.welcome = WelcomeMessage(self.topFrame,mirror)
            self.welcome.pack(padx=0, pady=50)

    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.tk.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.tk.attributes("-fullscreen", False)
        return "break"


def faceRecog():
    authenticationModule().train_dataset()  # Train dataset
    userID = authenticationModule().faceAuthentication()  # face recognition
    return userID



if __name__ == '__main__':
    while TRUE:

        print("basla")
        w = DefaultScreen()  # default screen

        w.tk.update()  # update the screen
        st = time.time()

        userID = faceRecog()  # face recognition

        print(userID)
        if userID == "Unknown Person":  # if face recognition is not successful
            userID = 'G9yt3NPxWDQgSHXh9H6RedhnBGh1'  # default user

        database = Database(userID)  # database
        print(database.get_name_value())  # get name value
        w.tk.destroy()  # destroy the screen

        mirror = FullscreenWindow(database)  # fullscreen window
        mirror.tk.update()  # update the screen
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
        time.sleep(50)  # sleep for 50 seconds
        mirror.tk.destroy()  # destroy the screen