#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[23]:

import pickle
import pandas as pd
import streamlit as st 
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title('Model Deployment: Random forest Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    
    engine_size = st.sidebar.number_input('engine_size', min_value=0.5, max_value=10.0)
    fuel_consumption_hwy = st.sidebar.number_input('fuel_consumption_hwy')
    
    fuel_type_D = st.sidebar.selectbox('fuel_type_D', ('0','1'))
    fuel_type_E = st.sidebar.selectbox('fuel_type_E', ('0','1'))
    fuel_type_N = st.sidebar.selectbox('fuel_type_N', ('0','1'))
    fuel_type_X = st.sidebar.selectbox('fuel_type_X', ('0','1'))
    fuel_type_Z = st.sidebar.selectbox('fuel_type_Z', ('0','1'))
    
    transmission_A = st.sidebar.selectbox('transmission_A', ('0','1'))
    transmission_AM = st.sidebar.selectbox('transmission_AM', ('0','1'))
    transmission_AS = st.sidebar.selectbox('transmission_AS', ('0','1'))
    transmission_AV = st.sidebar.selectbox('transmission_AV', ('0','1'))
    transmission_M = st.sidebar.selectbox('transmission_M', ('0','1'))
    
    #vehicle_class_COMPACT = st.sidebar.selectbox('vehicle_class_COMPACT', ('0','1'))
    #vehicle_class_FULL-SIZE = st.sidebar.selectbox('vehicle_class_FULL-SIZE', ('0','1'))
    #vehicle_class_MID-SIZE = st.sidebar.selectbox('vehicle_class_MID-SIZE', ('0','1'))
    #vehicle_class_MINICOMPACT = st.sidebar.selectbox('vehicle_class_MINICOMPACT', ('0','1'))
    #vehicle_class_MINIVAN = st.sidebar.selectbox('vehicle_class_MINIVAN', ('0','1'))
    #vehicle_class_PICKUP TRUCK - SMALL = st.sidebar.selectbox('vehicle_class_PICKUP TRUCK - SMALL', ('0','1'))
    #vehicle_class_PICKUP TRUCK - STANDARD = st.sidebar.selectbox('vehicle_class_PICKUP TRUCK - STANDARD', ('0','1'))
    #vehicle_class_SPECIAL PURPOSE VEHICLE = st.sidebar.selectbox('vehicle_class_SPECIAL PURPOSE VEHICLE', ('0','1'))
    #vehicle_class_STATION WAGON - MID-SIZE = st.sidebar.selectbox('vehicle_class_STATION WAGON - MID-SIZE', ('0','1'))
    #vehicle_class_STATION WAGON - SMALL = st.sidebar.selectbox('vehicle_class_STATION WAGON - SMALL', ('0','1'))
    #vehicle_class_SUBCOMPACT = st.sidebar.selectbox('vehicle_class_SUBCOMPACT', ('0','1'))
    #vehicle_class_SUV - SMALL= st.sidebar.selectbox('vehicle_class_SUV - SMALL', ('0','1'))
    #vehicle_class_SUV - STANDARD = st.sidebar.selectbox('vehicle_class_SUV - STANDARD', ('0','1'))
    #vehicle_class_TWO-SEATER  = st.sidebar.selectbox('vehicle_class_TWO-SEATER', ('0','1'))
    #vehicle_class_VAN - CARGO = st.sidebar.selectbox('vehicle_class_VAN - CARGO', ('0','1'))
    #vehicle_class_VAN - PASSENGER = st.sidebar.selectbox('vehicle_class_VAN - PASSENGER', ('0','1'))
    
    make_class_General = st.sidebar.selectbox('make_class_General', ('0','1'))
    make_class_Luxury = st.sidebar.selectbox('make_class_Luxury', ('0','1'))
    make_class_Premium = st.sidebar.selectbox('make_class_Premium', ('0','1'))
    make_class_Sports = st.sidebar.selectbox('make_class_Sports', ('0','1'))

    input_data = pd.DataFrame({'engine_size': [engine_size],
                    'fuel_consumption_hwy': [fuel_consumption_hwy],
                    'fuel_type_D': [fuel_type_D],
                    'fuel_type_E': [fuel_type_E],
                    'fuel_type_N': [fuel_type_N],
                    'fuel_type_X': [fuel_type_X],
                    'fuel_type_Z': [fuel_type_Z],
                    'transmission_A': [transmission_A],
                    'transmission_AM': [transmission_AM],
                    'transmission_AS': [transmission_AS],
                    'transmission_AV': [transmission_AV],
                    'transmission_M': [transmission_M],

                    'make_class_General': [make_class_General],
                    'make_class_Luxury': [make_class_Luxury],
                    'make_class_Premium': [make_class_Premium],
                    'make_class_Sports': [make_class_Sports]})
    
    features = pd.DataFrame(input_data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk




if st.button('Predict'):
    prediction = model.predict(df)
    st.write('Predicted CO2 Emissions:', prediction)



# In[ ]:






# In[ ]:
'''

                    'vehicle_class_COMPACT': [vehicle_class_COMPACT],
                    'vehicle_class_FULL-SIZE': [vehicle_class_FULL-SIZE],
                    'vehicle_class_MID-SIZE': [vehicle_class_MID-SIZE],
                    'vehicle_class_MINICOMPACT': [vehicle_class_MINICOMPACT],
                    'vehicle_class_MINIVAN': [vehicle_class_MINIVAN],
                    'vehicle_class_PICKUP TRUCK - SMALL': [vehicle_class_PICKUP TRUCK - SMALL],
                    'vehicle_class_PICKUP TRUCK - STANDARD': [vehicle_class_PICKUP TRUCK - STANDARD],
                    'vehicle_class_SPECIAL PURPOSE VEHICLE': [vehicle_class_SPECIAL PURPOSE VEHICLE],
                    'vehicle_class_STATION WAGON - MID-SIZE': [vehicle_class_STATION WAGON - MID-SIZE],
                    'vehicle_class_STATION WAGON - SMALL': [vehicle_class_STATION WAGON - SMALL],
                    'vehicle_class_SUBCOMPACT': [vehicle_class_SUBCOMPACT],
                    'vehicle_class_SUV - SMALL': [vehicle_class_SUV - SMALL],
                    'vehicle_class_SUV - STANDARD': [vehicle_class_SUV - STANDARD],
                    'vehicle_class_TWO-SEATER': [vehicle_class_TWO-SEATER],
                    'vehicle_class_VAN - CARGO': [vehicle_class_VAN - CARGO],
                    'vehicle_class_VAN - PASSENGER': [vehicle_class_VAN - PASSENGER],

'''