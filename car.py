from flask import Flask, redirect,request,render_template
import pickle
import pandas as pd


app = Flask(__name__,static_folder='static')

@app.route("/")
def fun1():
    return render_template("Car_price.html")

@app.route("/predict",methods=["post"])
def fun2():
    #getting values from user_form
    nm=request.form['txtname']
    year=int(request.form['txtyear'])
    p_price=float(request.form['txtprice'])/100000
    kms=int(request.form['txtkms'])
    fuel_type=request.form['radfuel']
    seller_type=request.form['radseller']
    transmission=request.form['radtrans']
    owner=request.form['radowner']
    
    #create dataframe using these inputs
    values=[[nm,year,p_price,kms,fuel_type,seller_type,transmission,owner]]
    df_test=pd.DataFrame(data=values,columns=[ 'Car_Name','Year', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])
    

    #loading ml model
    mymodel=pickle.load(open('car_model_rf.pkl',"rb"))
    car_price =round(mymodel.predict(df_test)[0],2)
    final_price=car_price*100000
    
    msg="Hello Sir, For model {} selling price is: ₹ {} ".format(nm,final_price)
    return render_template("prediction.html",msg=msg)

if __name__=="__main__":
    app.run(debug=True)