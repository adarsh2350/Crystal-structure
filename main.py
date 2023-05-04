## Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns', None)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from keras.models import load_model
import streamlit as st
import altair as alt




st.title("Crystal-Structures")

# Loading Data
df = pd.read_csv('crystal_data.csv')

# nav
nav = st.sidebar.radio("Navigation",["HOME","Prediction"])
if nav == "HOME" :
    st.image('dataset-cover.png', use_column_width=True)
    st.header("Sample Data set")
    st.table(df.head())



if nav == "Prediction":
    # Inputs
    st.header("Inputs")
    on,tw,th,fo = st.columns(4)
    fi,si,se,on_ = st.columns(4)
    tw_ ,th_,fo_,fi_ =  st.columns(4)
    si_,se_ =  st.columns(2)
    A = on.text_input("Enter A:",'Ac')
    B = tw.text_input("Enter B :","Ac")
    C = th.number_input("Enter v(A) :",0)
    D = fo.number_input("Enter v(B) :",0)
    E = fi.number_input("Enter r(AXII) :",1.12)
    F = si.number_input("Enter r(AVI) :",1.12)
    G = se.number_input("Enter r(BVI) :",1.12)
    H = on_.number_input("Enter EN(A) :",1.1)
    I = tw_.number_input("Enter EN(B) :",1.1)
    J = th_.number_input("Enter l(A-O) :",0)
    K = fo_.number_input("Enter l(B-O) :",0)
    L = fi_.number_input("Enter ΔENR :",-3.2)
    M = si_.number_input("Enter tG :",0.70)
    N = se_.number_input("Enter μ :",0.8)
    
    if st.button("Predict"): 

        val = {'Compound':"Pred",'A':"A",'B':"B",'In literature':"false",'v(A)':C,'v(B)':D,'r(AXII)(Å)':E,'r(AVI)(Å)':F,'r(BVI)(Å)':G,'EN(A)':H,'EN(B)':I,'l(A-O)(Å)':J,'l(B-O)(Å)':K,'ΔENR':L,'tG':M,'τ':"-",'μ':N,'Lowest distortion':"-"} 
        df = df.append(val, ignore_index=True)
        
        
        
        df = df.drop(["τ"], axis=1)
    

        val_a = pd.get_dummies(df['v(A)'], prefix="v(A)=", prefix_sep="")
        val_b = pd.get_dummies(df['v(B)'], prefix="v(B)=", prefix_sep="")
        df = pd.concat([df, val_a, val_b], axis=1)

        df = df.drop(["v(A)"], axis=1)
        df = df.drop(["v(B)"], axis=1)
    

        df = df.drop(["In literature"], axis=1)
        
        a_ = pd.get_dummies(df['A'], prefix="A=", prefix_sep="")
        b_ = pd.get_dummies(df['B'], prefix="B=", prefix_sep="")
        df = pd.concat([df, a_, b_], axis=1)

        df = df.drop(["A"], axis=1)
        df = df.drop(["B"], axis=1)

        raw = df[df["Compound"] == "Pred"]

        df = df.drop(df[df["Lowest distortion"] == "-"].index)
        df = df.drop(["Compound"], axis=1)
        raw = raw.drop(["Compound"], axis=1)
        raw = raw.drop(["Lowest distortion"], axis=1)



        # Modelling
        features = df.drop(labels=["Lowest distortion"], axis=1)
        target = df["Lowest distortion"]

        from sklearn.model_selection import train_test_split

        SEED = 0
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=SEED)





        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        accuracy1 = round(rf.score(X_test, y_test) * 100, 2)
        print(f"Accuracy of the SVC: {accuracy1}%")




        # SVM
        model = SVC()

        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        accuracy2 = round(model.score(X_test, y_test) * 100, 2)
        print(f"Accuracy of the SVC: {accuracy2}%")




        # Decision Tree
        dtree = DecisionTreeClassifier()
        
        dtree.fit(X_train,y_train)
        y_pred = dtree.predict(X_test)

        accuracy3 = round(dtree.score(X_test, y_test) * 100, 2)
        print(f"Accuracy of the dtree: {accuracy3}%")


        ans1 = rf.predict(raw)
        ans2 = model.predict(raw)
        ans3 = dtree.predict(raw)
        
        st.success("Done")

        data = {
            'Model': ['Random Forest', 'SVM', 'Decision tree'],
            'Prediction': [ans1, ans2, ans3],
        }

        df = pd.DataFrame(data)
        st.table(df)

        st.subheader("Accuracy of Models")

        data = pd.DataFrame({
            'X': ["Random forest","SVM","Decision tree"],
            'Y': [accuracy1, accuracy2, accuracy3]
        })
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('X', title=''),
            y=alt.Y('Y', title='accuracy %')
        )

        range_slider = alt.binding_range(min=1, max=5, step=1)
        selection = alt.selection_single(bind=range_slider, fields=['X'], name='Select X value')
        filtered_chart = chart.add_selection(selection).transform_filter(selection)

        st.altair_chart(filtered_chart, use_container_width=True)
