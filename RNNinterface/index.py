import math
from flask import Flask, render_template, request, url_for, redirect, escape, session, flash, redirect, jsonify
from scripts.pred import pred
import os

app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('client_login'))
    else:
        return render_template("index.html")


### La route pour se logger et atteindre l'interface index
@app.route('/login',methods = ['GET', 'POST'])
def client_login():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    if request.method == "POST":
        if request.form['username'] == 'admin':
            if request.form['password'] == 'password':
                session['logged_in'] = True
                return redirect(url_for('index'))
            else:
                flash('Wrong password !')
        else :
            flash('Unknown username !')
    return render_template("login.html")

#### La route pour effectuer le calcul de la prediction à l'aide du script 'pred'. En plus du resultat, on retourne également des éléments
### utiles à la visualisation (attentions et sentebces)
@app.route('/pred', methods = ['POST'])
def predict():
    if session['logged_in']:
        text = request.get_json()
        if text != "":
            (prediction, sentences,attentions) = pred([text])
            res = {}
            res['result'] = str(round(float(prediction)*100, 3))+'%'
            #attentions, colors et  sentences sont des listes de liste au cas ou plusieurs textes
            #sont envoyés à la fonction pred. Ici, comme on envoie qu'un seul text, on récupère toujours
            #le premier élement de ces listes.
            res['attentions'] = attentions.tolist()[0]
            res['sentences'] = sentences[0]
            return jsonify(res)
        else:
            return jsonify("0%")


if __name__ == '__main__':
    app.secret_key = os.urandom(24)
    app.run(debug=True)
