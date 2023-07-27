from flask import Flask, render_template,request,flash
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

with open('models/CountVector_Spam.pkl','rb') as cv:
    CV=pkl.load(cv)

with open('models/CountVector_Phishing.pkl','rb') as cv:
    CV1=pkl.load(cv)

# Load the models using tf.keras
model1 = tf.keras.models.load_model('models/ANN_Spam.h5')
model2 = tf.keras.models.load_model('models/ANN_Phishing.h5')

##preprocessing function
ps=PorterStemmer()

def preprocessing(text):
    ps = PorterStemmer()
    review1 = re.sub('[^a-zA-Z]', ' ', text)
    review1 = review1.lower()
    review1 = review1.split()
    review1 = [ps.stem(word) for word in review1 if not word in stopwords.words('english')]
    review1 = ' '.join(review1)
    return review1


# Text data to be classified
# text_data = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"


app = Flask(__name__)

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def spam():
    corpus1 = []
    s = ''
    rs = None
    c=''
    if request.method == 'POST':
        email = request.form.get('email')
        email_str = str(email)
        corpus_text = preprocessing(email_str)
        corpus1.append(corpus_text)

        # Transform the text data using the fitted CountVectorizer
        text_data_transformed = CV.transform(corpus1)

        # Make predictions using the pre-trained models
        result=model1.predict([text_data_transformed])

        if result < 0.5:
            rs=np.floor(result)
        else:
            rs=np.ceil(result)

        if rs:
            s="📢📢 You are Unsafe ! This Email is Spam Email 🚫🚫"
            c="🔭🧬Take care of your self🔭🧬"
            print('Spam Email')
        else:   
            s="📢📢 You are Safe ! This Email is Safe Email!"
            c="👍🤞All The Best👍🤞👍"

    return render_template('spam.html',s=s,rs=rs,c=c)#,prediction_spam=prediction_spam)


# Route for the phishinh page
@app.route('/phishing-email-detector',methods=['GET','POST'])
def phishing():
    corpus2 = []
    s = ''
    c=''
    rs = None
    result=None
    if request.method == 'POST':
            email = request.form.get('email')
            email_str = str(email)

            email = re.sub('[^a-zA-Z]', ' ', email_str)
            email=email.lower()
            # print(email)
            corpus2.append(email)
            # Transform the text data using the fitted CountVectorizer
            text_data_transformed = CV1.transform(corpus2)

            result=model2.predict([text_data_transformed])

            if result < 0.5:
                rs=np.floor(result)
            else:
                rs=np.ceil(result)

            if not rs:
                s="📢📢 You are Unsafe ! This Email is Phishing Email 🚫🚫"
                c="🔭🧬Take care of your self🔭🧬"

            else:   
                s="📢📢 You are Safe ! This Email is Safe Email!"
                c="👍🤞All The Best👍🤞👍"

    return render_template('phinishing.html',s=s,rs=rs,c=c)

if __name__ == '__main__':
    app.run(debug=True)
