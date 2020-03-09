import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np



app = Flask(__name__)


print("Loading model")
global sess
sess = tf.Session()
set_session(sess)
global model
model = load_model('my_cifar10_model.h5')
global graph
graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')


'''
In this prediction page, We will:
1 Read the image based on the filename, and store it as my_image
2 Resize the image to 32x32x3, which is the format that my model reads it in, and store it as my_image_re
3 Using the model, predict the probabilities that the image falls into the various classes, and put that under the variableprobabilities
4 Find the label and the probability of the top three most probable classes, and put them under predictions
5 Load the template of my webpage predict.html and give them the predictions in Step 4.
The five steps are reflected below with the code:
'''
#<filename> : placeholder for variable which stores the filename.
@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1
    my_image = plt.imread(os.path.join('uploads', filename))
    #Step 2
    my_image_re = resize(my_image, (32,32,3))
    
    #Step 3
    with graph.as_default():
      set_session(sess)
      probabilities = model.predict(np.array( [my_image_re,] ))[0,:]
      print(probabilities)
#Step 4
      number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
      index = np.argsort(probabilities)
      predictions = {
        "class1":number_to_class[index[9]],
        "class2":number_to_class[index[8]],
        "class3":number_to_class[index[7]],
        "prob1":probabilities[index[9]],
        "prob2":probabilities[index[8]],
        "prob3":probabilities[index[7]],
      }
#Step 5
    return render_template('predict.html', predictions=predictions)

app.run()



