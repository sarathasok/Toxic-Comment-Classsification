
from flask import Flask, render_template, url_for, request, jsonify      
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import csv

import numpy as np

app = Flask(__name__)

# Load the TF-IDF vocabulary specific to the category
with open(r"../Pickled_Files/GRP_22_toxic_vect.pkl", "rb") as f:
    toxic = pickle.load(f)

with open(r"../Pickled_Files/GRP_22_severe_toxic_vect.pkl", "rb") as f:
    severe = pickle.load(f)

with open(r"../Pickled_Files/GRP_22_obscene_vect.pkl", "rb") as f:
    obscene = pickle.load(f)

with open(r"../Pickled_Files/GRP_22_insult_vect.pkl", "rb") as f:
    insult = pickle.load(f)

with open(r"../Pickled_Files/GRP_22_threat_vect.pkl", "rb") as f:
    threat = pickle.load(f)

with open(r"../Pickled_Files/GRP_22_identity_hate_vect.pkl", "rb") as f:
    hate = pickle.load(f)

# Load the pickled RDF models
with open(r"../Pickled_Files/GRP_22_toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open(r"../Pickled_Files/GRP_22_severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open(r"../Pickled_Files/GRP_22_obscene_model.pkl", "rb") as f:
    obs_model  = pickle.load(f)

with open(r"../Pickled_Files/GRP_22_insult_model.pkl", "rb") as f:
    ins_model  = pickle.load(f)

with open(r"../Pickled_Files/GRP_22_threat_model.pkl", "rb") as f:
    thr_model  = pickle.load(f)

with open(r"../Pickled_Files/GRP_22_identity_hate_model.pkl", "rb") as f:
    ide_model  = pickle.load(f)

# Render the HTML file for the home page
@app.route("/")
def home():
    return render_template('mytox.html')

def write_csv(out_list):
	with open('GRP22_results.csv','a') as csvfile:
		filewriter = csv.writer(csvfile)
		print("going to write")
		list=['Comment','Toxic','SevereToxic','Obscene','threat','insult','IdentityHate']
		filewriter.writerow(out_list)
		print("writing succesful")

@app.route("/predict", methods=['POST'])
def predict():
    
    # Take a string input from user
    user_input = request.form['text']
    data = [user_input]

    vect = toxic.transform(data)
    pred_tox = tox_model.predict_proba(vect)[:,1]

    vect = severe.transform(data)
    pred_sev = sev_model.predict_proba(vect)[:,1]

    vect = obscene.transform(data)
    pred_obs = obs_model.predict_proba(vect)[:,1]

    vect = threat.transform(data)
    pred_thr = thr_model.predict_proba(vect)[:,1]

    vect = insult.transform(data)
    pred_ins = ins_model.predict_proba(vect)[:,1]

    vect = hate.transform(data)
    pred_ide = ide_model.predict_proba(vect)[:,1]

    out_tox = round(pred_tox[0], 2)
    out_sev = round(pred_sev[0], 2)
    out_obs = round(pred_obs[0], 2)
    out_ins = round(pred_ins[0], 2)
    out_thr = round(pred_thr[0], 2)
    out_ide = round(pred_ide[0], 2)
    out_list = [user_input ,pred_tox,pred_sev ,pred_obs ,pred_thr ,pred_ins ,pred_ide  ]	
    write_csv(out_list)
    print(out_tox)

    return render_template('mytox.html', 
                            pred_tox = 'Toxic: {}'.format(out_tox),
                            pred_sev = 'Severe Toxic: {}'.format(out_sev), 
                            pred_obs = 'Obscene: {}'.format(out_obs),
                            pred_ins = 'Insult: {}'.format(out_ins),
                            pred_thr = 'Threat: {}'.format(out_thr),
                            pred_ide = 'Identity Hate: {}'.format(out_ide)                        
                            )
     
# Server reloads itself if code changes so no need to keep restarting:
app.run(debug=True)

