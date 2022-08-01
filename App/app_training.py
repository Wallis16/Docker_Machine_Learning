
from flask import Flask, render_template, request
from Training import train_online

FOLDER = "static/images"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FOLDER

@app.route('/')
def home():
   return render_template("home.html")

@app.route('/training')
def show_index():
   full_filename = FOLDER + '/diabetes.jpg' 
   return render_template("training.html", user_image = full_filename)

@app.route('/training', methods=['POST'])
def my_form_post():
    URL = request.form['fname']
    train_online.training(URL)
    return render_template("generated.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')