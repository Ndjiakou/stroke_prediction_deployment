# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:59:31 2022

@author: HP
"""


from flask import Flask, request, render_template, flash, jsonify
import pickle

##to initializae our app and create a class for our app
app = Flask(__name__)
app.secret_key = "apkofriowjfkf"

##to create a route and /output represents the last part of our URL
@app.route("/")
def index():
##the function that will render our html template 
    return render_template('index.html')
##GET method is used to retrieve or request data and POST method is use for posting or sending data.    
@app.route("/prediction_result",methods=["POST","GET"])
# #INPUT - FROM - USER

#user-input
def prediction_result():


    if request.method == 'POST':
        
        #gender
        g = request.form['gender']
        if g == "Male":
            g = 1
        elif g == "Female":
            g = 0
        else :
            g=2
        
        #age
        a = request.form['age']
        a = int(a)
        ## we use the standardscaler formular to scale data
        ## z= x-u/s
        #a = ((a-43.226614481409)/(22.61264672311349))
        
        
        #hyper-tension
        hyt = request.form['hypertension']
        hyt = hyt.lower()
        if hyt == "Yes":
            hyt = 1
        else:
            hyt = 0
            
        
        #heart-disease
        ht = request.form['heart_disease']
        ht = ht.lower()
        if ht == "Yes":
            ht = 1
        else:
            ht = 0
            
        
        #marriage
        m = request.form['ever_married']
        m = m.lower()
        if m == "Yes":
            m = 1
        else:
            m = 0
            
        
        #worktype
        
        work_type = request.form['work_type']
        work_type = work_type.lower()
         
        if(work_type=='work_type_Govt_job'):
            work_type_Govt_job = 1
            work_type_Never_worked = 0
            work_type_children = 0
            work_type_Private = 0
            work_type_Self_employed = 0
        elif(work_type == 'work_type_Never_worked '):
            work_type_Never_worked  = 1
            work_type_Govt_job = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0
        elif(work_type =="work_type_Private"):
            work_type_Never_worked  = 0
            work_type_Govt_job = 0
            work_type_Private = 1
            work_type_Self_employed = 0
            work_type_children = 0
        elif(work_type =="work_type_Self_employed"):
            work_type_Never_worked  = 0
            work_type_Govt_job = 0
            work_type_Private = 0
            work_type_Self_employed = 1
            work_type_children = 0
        else:
           work_type_Never_worked  = 0
           work_type_Govt_job = 0
           work_type_Private = 0
           work_type_Self_employed = 0
           work_type_children = 1
        
      
        #residency-type
        r = request.form['Residence_type']
        r = r.lower()
        if r == "Urban":
            r = 1
        else:
            r = 0
            
        #glucose-levels
        gl = request.form['avg_glucose_level']
        gl = float(gl)
        #gl =  ((int(gl) - 106.1476771037182)/(45.28356015058198))
            
            
        #bmi
        b = request.form['bmi']
        b = float(b)
        #b = ((b-28.862035225048924)/(7.699562318787506))
        
            
        #smoking
        
        smoking = request.form['smoking']
        smoking = smoking.lower()
         
        if(smoking=='smoking_status_formerly smoked'):
            smoking_status_formerly_smoked = 1
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0
            smoking_status_Unknown = 0
        elif(smoking == 'smoking_status_smokes'):
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 1
            smoking_status_Unknown = 0
        elif(smoking=="smoking_status_never_smoked"):
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 1
            smoking_status_smokes = 0
            smoking_status_Unknown = 0
        else:
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0
            smoking_status_Unknown = 1   
                
        try:
            prediction = stroke_prediction_function(g,a,hyt,ht,m,work_type_Govt_job,work_type_Never_worked,work_type_Private,work_type_Self_employed,work_type_children,r,gl,b,smoking_status_Unknown,smoking_status_formerly_smoked,smoking_status_never_smoked,smoking_status_smokes)
            return render_template('prediction_result.html',prediction=prediction)

        except ValueError:
            return "Please Enter Valid Values"
        

#prediction-model
def stroke_prediction_function(g,a,hyt,ht,m,work_type_Govt_job,work_type_Never_worked,work_type_Private,work_type_Self_employed,work_type_children,r,gl,b,smoking_status_Unknown,smoking_status_formerly_smoked,smoking_status_never_smoked,smoking_status_smokes):
    
    #load model
    new_pipe = pickle.load(open('new_pipe.pkl','rb'))

    #predictions
    patient_diagnostic = new_pipe.predict([[g,a,hyt,ht,m,work_type_Govt_job,work_type_Never_worked,work_type_Private,work_type_Self_employed,work_type_children,r,gl,b,smoking_status_Unknown,smoking_status_formerly_smoked,smoking_status_never_smoked,smoking_status_smokes]])

    #output
    if patient_diagnostic[0] == 1:
        conclusion = 'According to your parameters your chances of having stroke are hight '
    else:
        conclusion = 'According to your parameters your chances of having strokes are low '

    return conclusion