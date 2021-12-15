import streamlit as st
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings

header=st.container()
dataset=st.container()
features=st.container()
model_training=st.container()
model_results=st.container()

#@st.cache
#def get_data(filename):
    #telecom_data = pd.read_csv(filename)

    #return telecom_data
with header:
    st.header('Welcome to my data science project!')
    st.text('In this proyect i look into the churn of a telco company')
with dataset:
    st.header('1.Churn telco dataset')
    st.text("I've found this data set on: 'https://raw.githubusercontent.com/kaleko/CourseraML/master/ex2/data/ex2data1.txt'")
    telecom_data =pd.read_csv('telecom_churn.txt')
    st.text("This is my DATA before the EDA")
    st.text(telecom_data.head(3))
    # Limpieza del DF
    # carga
    # cambiar nombre
    telecom_data.rename(columns={"Churn_num": "Churn"}, inplace=True)
    # se transforma a numerica; 1,0 en vez de yes, no
    telecom_data['Churn'] = telecom_data['Churn'].astype('int64')
    # dumificamos y creamos un df con las variables voice y ip (voice mail plan/international calls.
    vmp = pd.get_dummies(telecom_data['Voice mail plan'], drop_first=True, prefix="voice")
    ip = pd.get_dummies(telecom_data['International plan'], drop_first=True, prefix="ip")
    # eliminamos las columnas que había antes
    telecom_data.drop(['Voice mail plan', 'International plan'], axis=1, inplace=True)
    # unimos al df los dfs que hemos creado con las variables categóricas
    telecom_data = pd.concat([telecom_data, vmp, ip], axis=1)
    telecom_data.drop('State', axis=1, inplace=True)
    st.text("This is my DATA after some EDA")
    st.text(telecom_data.head(6))
with features:
    st.header('2.The features I modified')
    st.markdown("* **State:** I don't consider it an useful feature.")
    st.markdown("* **Voice mail plan & international plan:** I converted them into numeric.")
with model_training:
    st.header('3.Time to train the model')
    #st.text('Choosee the hyperparameters of the model and see how the performance changes')
    sel_col, disp_col = st.columns(2)\

    #features= st.multiselect(
    #'What features should we use',
    #['Account length','Area code','Number vmail messages','Total day minutes','Total day calls',
     #'Total day charge','Total eve minutes','Total eve calls','Total eve charge','Total night minutes'
     #'Total night calls','Total night charge','Total intl minutes','Total intl calls','Total intl charge'
     #'Customer service calls','voice_Yes','ip_Yes',], help=("You can choose all! "))
    #st.write('You selected:', features)'''

    # MATRIZ DE ENTRENAMIENTO
    import random

    random.seed(113)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(telecom_data.drop('Churn', axis=1),
                                                        telecom_data['Churn'], test_size=0.25,
                                                        random_state=101)

    from sklearn.linear_model import LogisticRegression
    import warnings

    warnings.filterwarnings('ignore')
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
with model_results:
    st.header("4. Let's see the predictions")
    Account_length_new=st.number_input('New Account_length')
    Area_code_new=st.number_input('New Area_code ')
    Number_vmail_messages_new=st.number_input('New Number_vmail')
    Total_day_minutes_new=st.number_input('New Total_day_minutes ')
    Total_day_calls_new=st.number_input('New Total_day_calls')
    Total_day_charge_new=st.number_input('New Total_day_charge')
    Total_eve_minutes_new=st.number_input('New Total_eve_minutes_new')
    Total_eve_calls_new=st.number_input('New Total_eve_calls')
    Total_eve_charge_new=st.number_input('New Total_eve_charge ')
    Total_night_minutes_new=st.number_input('New Total_night_minutes')
    Total_night_calls_new=st.number_input('New Total_night_calls ')
    Total_night_charge_new=st.number_input('New Total_night_charge')
    Total_intl_minutes_new=st.number_input('New Total_intl_minutes ')
    Total_intl_calls_new=st.number_input('New Total_intl_calls')
    Total_intl_charge_new=st.number_input('New Total_intl_charge')
    Customer_service_calls_new=st.number_input('New Customer_service_calls')
    voice_Yes_new=st.selectbox('New voice_Yes. 1=Yes/0=No', options=[0,1], index=0)
    ip_Yes_new=st.selectbox('New ip_Yes1=Yes/0=No', options=[0,1], index=0)

    X_new = pd.DataFrame(
        {'Account length': [Account_length_new],'Area code': [Area_code_new],'Number vmail messages': [Number_vmail_messages_new],'Total day minutes': [Total_day_minutes_new],'Total day calls': [Total_day_calls_new],
     'Total day charge': [Total_day_charge_new],'Total eve minutes': [Total_eve_minutes_new],'Total eve calls': [Total_eve_calls_new],'Total eve charge': [Total_eve_charge_new],'Total night minutes': [Total_night_minutes_new],
     'Total night calls': [Total_night_calls_new],'Total night charge': [Total_night_charge_new],'Total intl minutes': [Total_intl_minutes_new],'Total intl calls': [Total_intl_calls_new],'Total intl charge': [Total_intl_charge_new],
     'Customer service calls': [Customer_service_calls_new],'voice_Yes': [voice_Yes_new],'ip_Yes': [ip_Yes_new]})

    st.write(f"La predicion para el nuevo valor es:{logmodel.predict(X_new)}\n")
    roc_auc = roc_auc_score(y_test, logmodel.predict_proba(X_test)[:, 1])

    st.text(f"Probabilidad={roc_auc}")









