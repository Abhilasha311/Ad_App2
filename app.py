#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import flask
from flask import Flask, request , jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from collections.abc import Mapping, MutableMapping

from flask import Flask, request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
filename = 'Ad_classifier.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/home', methods=['POST'])
def home():
    data1 = request.form['age']
    data2 = request.form['income']
    arr = np.array([[data1, data2]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)

