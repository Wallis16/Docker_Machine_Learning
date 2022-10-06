
from flask import Flask, render_template, request
from Inference import inference

FOLDER = "static/images"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FOLDER

@app.route('/')
def home():
   return render_template("home_inference.html")

@app.route('/training')
def show_index():
   full_filename = FOLDER + '/diabetes.jpg' 
   return render_template("inference.html", user_image = full_filename)

@app.route('/training', methods=['POST'])
def my_form_post():
    URL = request.form['fname']
    inference.prediction(URL)
    return render_template("generated_inference.html")

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')