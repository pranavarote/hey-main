
import streamlit as st
# import plotly.express as px
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="PriceDekho",
    page_icon="🚗",
)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{

background-image: url("https://images.wallpaperscraft.com/image/single/bmw_headlights_lights_137326_1920x1080.jpg");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>

"""

# add_selectbox = st.sidebar.radio(
#    "Please choose an option",
#    ("check car price ", "Option 2", "Option 3")
# )
info = '''
                    Price prediction for cars and bikes using machine learning can be done using various
                    algorithms such as linear regression, decision tree regression, random forest regression, and
                    neural networks. Here's a basic approach:
                    1. Data Collection: Collect data on past sales of cars and bikes, along with their features
                    such as brand, model, engine size, mileage, year of manufacture, and other relevant
                    information.
                    2. Data Preprocessing: Clean and preprocess the collected data, including handling missing
                    values, removing outliers, and normalizing the data.
                    3. Feature Selection: Identify the features that are most relevant to the price prediction, and
                    eliminate any irrelevant or redundant features.
                    4. Model Training: Use a machine learning algorithm to train a model on the preprocessed
                    data. This involves splitting the data into training and testing sets, and then using the
                    training data to fit the model.
                    5. Model Evaluation: Evaluate the performance of the trained model on the testing data,
                    using metrics such as mean squared error, mean absolute error, and R-squared.
                    6. Model Deployment: Once you have a model with acceptable performance, deploy it into
                    a web application or a mobile app for end-users to use.
                    Keep in mind that the accuracy of your price predictions will depend on the quality and
                    quantity of data you have, as well as the complexity and accuracy of your chosen algorithm.'''


st.markdown(page_bg_img, unsafe_allow_html=True)
# Define the main function

st.title(" PREDICT PRICE FOR YOUR VEHICLE")
st.text \
    ('This website gets the data and provides the predicted result using machine learning model which is pretrained on lots of data ')

# Define the main function
def main():


    # Create the tabbed interface
    tabs = ["INFORMATION", "Bike price ! 🏍️", "Car price ! 🚗" ,"Charts"]
    tab = st.selectbox("", tabs)
    if tab == "INFORMATION":
        infopage1 = f'''
            <p style = "color:#8BF5FA"> Price prediction for cars and bikes using machine learning can be done using various algorithms such as linear regression, decision tree regression, random forest regression, and neural networks. Here's a basic approach:</p>
            <p style = "color:#8BF5FA"> Data Collection: Collect data on past sales of cars and bikes, along with their features such as brand, model, engine size, mileage, year of manufacture, and other relevant information. </p>
            <p style = "color:#8BF5FA">Data Preprocessing: Clean and preprocess the collected data, including handling missing values, removing outliers, and normalizing the data. </p>
            <p style = "color:#8BF5FA"> Feature Selection: Identify the features that are most relevant to the price prediction, and eliminate any irrelevant or redundant features. </p>
            <p style = "color:#8BF5FA"> Model Training: Use a machine learning algorithm to train a model on the preprocessed data. This involves splitting the data into training and testing sets, and then using the training data to fit the model. </p>
            <p style = "color:#8BF5FA"> Model Evaluation: Evaluate the performance of the trained model on the testing data, using metrics such as mean squared error, mean absolute error, and R-squared. </p>
            <p style = "color:#8BF5FA"> Model Deployment: Once you have a model with acceptable performance, deploy it into a web application or a mobile app for end-users to use. </p>
            <p style = "color:#8BF5FA"> Keep in mind that the accuracy of your price predictions will depend on the quality and quantity of data you have, as well as the complexity and accuracy of your chosen algorithm.</p> </p>


'''
        st.markdown(infopage1, unsafe_allow_html=True)




    elif tab == "Bike price ! 🏍️":
        with st.form("tab2_form"):
            brandoption = pkl.load(open('bike_brand_option.pkl','rb'))
            
            bikebrand = ['TVS', 'Royal Enfield', 'Triumph', 'Yamaha', 'Honda', 'Hero','Bajaj', 'Suzuki', 'Benelli', 'KTM', 'Mahindra', 'Kawasaki',
                            'Ducati', 'Hyosung', 'Harley-Davidson', 'Jawa', 'BMW']
            
            old = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            owner_bike = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner Or More']

            # c1,c2 = st.columns(2)
            # with c1:
            original_title = '<h1 style=" color:#3366FF; font-size: 40px;"> Bike price ! 🏍️</h1>'
            st.markdown(original_title, unsafe_allow_html=True)

            Brand_bike = st.selectbox(label='Select Brand ', options=bikebrand)
            # st.write('You selected :',Brand_bike)
            submit_button = st.form_submit_button("Click") == True
            Model_bike = st.selectbox(label='Select Model ', options=brandoption[Brand_bike])
            # st.write('You selected :',Model_bike)

            #City = st.selectbox(label='Enter the City', options=city)
            # st.write('You selected City as :',City)

            Driven_bike = st.text_input("kms_driven", 0, placeholder="Enter total kms completed ")
            # st.write('You selected Kms Driven as :',Model_bike)

            Owners_bike = st.selectbox(label='Select owner', options=owner_bike)
            # st.write('You selected Owner type as:',Owners_bike)

            # Old = st.select_slider(label='Select how old model model ', options=(old))
            Old = st.select_slider(label='Select how old the bike is :', options=(old))
            # Old =st.text_input("Old", placeholder= "Enter Age according to the selected model")
            # st.write('You selected model age / how much old :',Old)

            Power = st.text_input("Power", 0, placeholder="Enter Power according to the selected model")
            # st.write('You entered power ',Power)

            d = {'brand': [Brand_bike], 'bike_name': [Model_bike],'kms_driven': [Driven_bike],  'owner': [Owners_bike],
                  'age': [Old], 'power': [Power]}
            data = pd.DataFrame(data=d)
            st.dataframe(data)

            model = pkl.load(open('bike_model_extr.pkl', 'rb'))
            output_bike = model.predict(data)
            output_bike = output_bike[0]

            submit_button = st.form_submit_button("Submit")
        if submit_button:
            st.write('Predicted price for', Model_bike, 'is : ', output_bike)


    elif tab == "Car price ! 🚗":

        with st.form("tab3_form"):

            df = pd.read_csv('best_updated_cardata.csv')
            df = df.iloc[:, 1:]

            brands = ['Maruti', 'Hyundai', 'Datsun', 'Honda', 'Tata', 'Chevrolet','Toyota', 'Jaguar', 'Mercedes-Benz', 'Audi', 'Skoda', 'Jeep','BMW', 'Mahindra', 
                      'Ford', 'Nissan', 'Renault', 'Fiat','Volkswagen', 'Volvo', 'Mitsubishi', 'Land', 'Daewoo', 'MG','Force', 'Isuzu', 'OpelCorsa', 'Ambassador', 
                      'Kia', 'Porsche','Lexus', 'Mini', 'Premier', 'Maserati', 'Bentley']

            #owner = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner Or More']
            # model = df[df['brand']==brand]['model'].unique().tolist()
            fuel = ['Petrol', 'Diesel']
            transtype = ['Manual', 'Automatic']
            owner = [0, 1, 2, 3, 4, 5]
            year = [2005, 2004, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                    2020, 2021, 2022, 2023]
            models = pkl.load(open('model_option_car.pkl', 'rb'))

            Brand = st.selectbox(label='Select Brand ', options=brands)
            # st.write('You selected :',Brand)
            submit_button = st.form_submit_button("Click") == True

            modelss = models[Brand]
            Model = st.selectbox(label='Select Model ', options=modelss)
            # st.write('You selected car model:',Model)

            transtype = st.selectbox(label='Select tramsmission type ', options=transtype)
            # st.write('You selected transmission as:',transtype)

            Fuel = st.selectbox(label='Select fuel', options=fuel)
            # st.write('You selected Owner type as:',Fuel)

            Owners = st.selectbox(label='Select owner', options=owner)
            # st.write('You selected Owner type as:',Owners)

            manufacture = st.select_slider(label='Select the year of manufacturing', options=(year))
            # st.write('You selected manufacture year as :',manufacture)

            Driven = st.text_input("kms_driven", 0)
            # st.write('You selected Kms Driven as :',Driven)

            d = {'brand': [Brand], 'model': [Model], 'fuel': [Fuel], 'transtype': [transtype], 'owner': [Owners],
                 'manufacture': [manufacture], "kms": [Driven]}
            st.dataframe(data=d)
            data = pd.DataFrame(data=d)

            load = pkl.load(open('Algo_for_car.pkl','rb'))

            output = load.predict(data)
            output = output[0]
            submit_button = st.form_submit_button("Submit")
        if submit_button:
            st.write('Predicted price for ', Model, 'is :', output)






    elif tab == "Charts":
        df = pd.read_csv('updated_bikefile.csv')
        df = df.iloc[:, 1:]

        brands = df['brand'].unique()
        st.header('Select brand to check the top selling bikes')

        brand_name = st.selectbox(label='', options=brands)

        def display_bargraph(brand_name):
            # Create figure and axis
            fig, ax = plt.subplots()
            
            # Create bar chart
            ax.bar(labels, values)

            # Set chart properties
            ax.set_xlabel('X-axis label')
            ax.set_ylabel('Y-axis label')
            ax.set_title('Bar Graph')
            # col_map = plt.get_cmap('tab10')
            ax.figsize = [10, 5]
            ax.set_ylabel('SALES')
            ax.set_xlabel(brand_name)
            ax.set_title('TOP SELLING MODELS ')
            # Display chart
            st.pyplot(fig)

        # Sample input
        labels = df.groupby('brand').get_group(brand_name)['bike_name'].str.split(' ').str.slice(1, 3).str.join(
            ' ').value_counts().head().index.tolist()
        values = df.groupby('brand').get_group(brand_name)['bike_name'].value_counts().head().values.tolist()

        # Call function
        display_bargraph(brand_name)





     


# Call the main function
main()
