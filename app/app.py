import streamlit as st
import os
import pickle
import pandas as pd

regressor_from_file = 0

def show_app_info_page():
    st.write("## Задача")
    st.write(
        "Решить задачу регрессии - предсказать стоимость дома по совокупности различных признаков")
    st.write("## Входные данные")
    st.write(
        "Для работы необходим препроцессированный файл с данными(автоматический препроцессинг в разработке). "
        "В исходном датасете находится множество различных параметров жилого дома: площадь, наличие и характеристики гаража и т .д")
    st.write("## Модель")
    st.write(
        "В приложении используется модель XGBoost, она была выбрана в результате "
        "ряда тестов качества различных моделей регрессий так как показала наилучший результат. ")
    st.write("Данная модель принимала участие в соревновании на ресурсе Kaggle: "
        "https://www.kaggle.com/c/house-prices-advanced-regression-techniques/")


def show_predictions_page():
    st.write(
        "Загрузите обработанные данные")
    folder_path='.'
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    test_data = pd.read_csv(selected_filename, index_col = 0)
    result = regressor_from_file.predict(test_data)
    chart = st.line_chart(result)

def select_page():
    page = st.sidebar.selectbox("Выберите страницу", ("О сервисе", "Прогнозирование"))

    if page == "О сервисе":
        show_app_info_page()
    else:
        show_predictions_page()

with open('model/model.pkl', 'rb') as pkl_file:
    regressor_from_file = pickle.load(pkl_file)

st.title("House Prices - Advanced Regression Techniques")
select_page()

