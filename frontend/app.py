import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Определение клинической стадии", layout="centered", page_icon="🏥")

st.markdown("""
<style>
    .stRadio [role="radiogroup"] > label {
        font-size: 2rem !important;
        line-height: 1.1 !important;
    }
    .stRadio [role="radiogroup"] > label > div {
        font-size: 2rem !important;
    }
    .stRadio [role="radiogroup"] > label span {
        font-size: 2rem !important;
    }
    
    .stMultiSelect [data-baseweb="select"] {
        font-size: 1.2rem !important;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        font-size: 1.1rem !important;
    }
    .stMultiSelect ul[role="listbox"] {
        font-size: 1.2rem !important;
    }
    
    .stNumberInput input {
        font-size: 1.1rem !important;
    }
    .stSlider [role="slider"] {
        font-size: 1.1rem !important;
    }
    
    .stRadio [role="radiogroup"] {
        gap: 8px !important;
    }
    .stForm {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("🏥 Определение клинической стадии заболевания")
st.markdown("""
Заполните параметры пациента: 
            
- Пропущенные поля автоматически обработаются моделью по клиническим правилам.  
- Для контрактуры и кардиомиопатия можно выбрать несколько вариантов.

""")
st.divider()

with st.form("patient_input", clear_on_submit=True):
    
    st.markdown("### 📋 Базовые параметры")
    st.markdown("<div style='margin: 25px 0'></div>", unsafe_allow_html=True)
    
    st.markdown("**Возраст (лет):**")
    age = st.number_input("Возраст (лет)", min_value=0, max_value=100, value=None, step=1, 
                          placeholder="Например: 7", label_visibility="collapsed", key="age_input")
    
    st.markdown("**Способность к ходьбе и бегу:**")
    walk_ability = st.radio("Способность к ходьбе и бегу", 
        options=[
            "Бегает с отрывом двух ног от земли, присутствует фаза полета",
            "Бегает с отрывом двух ног от земли, но без фазы полета",
            "Способность к бегу по типу \"Duchenne jog\" (напоминает спортивную ходьбу)",
            "Невозможность бега при сохранении ходьбы",
            "Невозможность самостоятельно пройти 10 м",
            "Не ходит"
        ], 
        index=None, label_visibility="collapsed", key="walk_ability_radio"
    )
    
    st.markdown("**Баллы мускулатуры (верхние конечности):**")
    muscle_upper = st.slider("Баллы мускулатуры (верхние конечности)", 0.0, 5.0, value=None, step=0.5, 
                             label_visibility="collapsed", key="muscle_upper_slider")
    st.markdown("<div style='margin: 10px 0'></div>", unsafe_allow_html=True)
    
    st.markdown("**Баллы мускулатуры (нижние конечности):**")
    muscle_lower = st.slider("Баллы мускулатуры (нижние конечности)", 0.0, 5.0, value=None, step=0.5, 
                             label_visibility="collapsed", key="muscle_lower_slider")
    
    st.divider()

    st.markdown("### 🦵 Двигательные тесты")
    st.markdown("<div style='margin: 25px 0'></div>", unsafe_allow_html=True)
    
    st.markdown("**Использование перекатов при вставании:**")
    gowers_roll = st.radio("Использование перекатов при вставании",
        options=["Подъем с пола без переката", "Подъем с пола с перекатом", "Подъем с пола с переворотом на живот"], 
        index=None, label_visibility="collapsed", key="gowers_roll_radio")
    
    st.markdown("**Использование опоры при вставании:**")
    gowers_support = st.radio("Использование опоры при вставании",
        options=[
            "Подъем с пола без опоры",
            "Подъем с пола с опорой на одну руку о пол или о бедро",
            "Подъем с пола с опорой на две руки о пол и бедра, \"взбираясь по себе\"",
            "Подъем с пола с использованием окружающей мебели при вставании",
            "Утрата навыка подъема на ноги"
        ], 
        index=None, label_visibility="collapsed", key="gowers_support_radio")
    
    st.markdown("**Тест с шестиминутной ходьбой:**")
    six_min_walk = st.radio("Тест с шестиминутной ходьбой",
        options=["> 500 м", "от 350 м до 500 м", "от 150 м до 350 м", "от 10 м до 150 м", "< 10 м"], 
        index=None, label_visibility="collapsed", key="six_min_walk_radio")
    
    st.markdown("**Приседания:**")
    squats = st.radio("Приседания",
        options=[
            "Приседает без применения компенсирующих приемов",
            "Приседает с небольшими трудностями (компенсируются приемами Говерса, опирается одной рукой)",
            "Приседает только с использованием приемов Говерса, с опорой на обе руки",
            "Приседает, падая на ягодицы, вставание только с использованием окружающей мебели",
            "Невозможность приседания (навык утерян)"
        ], 
        index=None, label_visibility="collapsed", key="squats_radio")
    
    st.markdown("**Подпрыгивания (вверх на 2-х ногах):**")
    jump_2leg = st.radio("Подпрыгивания (вверх на 2-х ногах)",
        options=[
            "Прыжок с отрывом двух ног от поверхности пола",
            "Потеря способности к прыжкам при сохранении вставания на носки",
            "Невозможность прыжка, переминается, пытаясь подняться на носки",
            "Невозможность изобразить попытку прыжка"
        ], 
        index=None, label_visibility="collapsed", key="jump_2leg_radio")
    
    st.markdown("**Подпрыгивания (вверх на 1-ой ноге):**")
    jump_1leg = st.radio("Подпрыгивания (вверх на 1-ой ноге)",
        options=[
            "Может подпрыгнуть на одной ноге",
            "Подъем на носок без отрыва от поверхности пола",
            "Не может даже подняться на носок, попытка прыжка без отрыва от поверхности пола"
        ], 
        index=None, label_visibility="collapsed", key="jump_1leg_radio")

    st.divider()

    st.markdown("### 🪜 Лестница и особенности позы")
    st.markdown("<div style='margin: 25px 0'></div>", unsafe_allow_html=True)
    
    st.markdown("**Подъем по лестнице:**")
    stair_up = st.radio("Подъем по лестнице",
        options=[
            "Подъем альтернативным шагом, не нуждаясь в опоре",
            "Подъем приставным шагом без опоры",
            "Подъем приставным шагом, опираясь одной рукой о бедро или о перила",
            "Подъем приставным шагом, держась двумя руками за перила или одной рукой за перила и второй рукой за бедро, подтягивая себя наверх",
            "Невозможность подъема по лестнице"
        ], 
        index=None, label_visibility="collapsed", key="stair_up_radio")
    
    st.markdown("**Функциональность верхних конечностей:**")
    upper_func = st.radio("Функциональность верхних конечностей",
        options=[
            "Пациент может поднять руки выше уровня плеч",
            "Пациент может поднять руки не выше уровня плеч",
            "Пациент не может поднять руки"
        ], 
        index=None, label_visibility="collapsed", key="upper_func_radio")
    
    st.markdown("**Спуск по лестнице:**")
    stair_down = st.radio("Спуск по лестнице",
        options=[
            "Спуск альтернативным шагом, не нуждаясь в опоре",
            "Спуск приставным шагом без опоры",
            "Спуск приставным шагом, опираясь одной рукой о бедро или о перила",
            "Спуск приставным шагом, опираясь двумя руками о перила или держась одной рукой за перила и второй рукой за бедро",
            "Невозможность спуска по лестнице"
        ], 
        index=None, label_visibility="collapsed", key="stair_down_radio")
    
    st.markdown("**Положение пациента:**")
    position = st.radio("Положение пациента",
        options=[
            "Пациент в позе \"сидя\" (коленные суставы под углом более 90 градусов)",
            "Сидячий пациент может быть поставлен на ноги с помощью опоры для стояния",
            "Пациент в позе \"лежа\" (коленные суставы под углом менее 90 градусов)",
            "Пациента невозможно поставить в вертикальное положение"
        ], 
        index=None, label_visibility="collapsed", key="position_radio")

    st.divider()

    st.markdown("### 🦴 Осанка и сколиоз")
    st.markdown("<div style='margin: 20px 0'></div>", unsafe_allow_html=True)
    
    st.markdown("**Гиперлордоз:**")
    hyperlordosis = st.radio("Гиперлордоз",
        options=["Небольшой гиперлордоз при вертикализации", "Умеренный гиперлордоз при вертикализации", "Выраженный гиперлордоз при вертикализации"], 
        index=None, label_visibility="collapsed", key="hyperlordosis_radio")
    
    st.markdown("**Стадия сколиоза:**")
    scoliosis_stage = st.radio("Стадия сколиоза", 
        options=["Формирующийся сколиоз", "Сколиотическая деформация"], 
        index=None, label_visibility="collapsed", key="scoliosis_stage_radio")
    
    st.markdown("**Спутники сколиоза:**")
    scoliosis_comp = st.radio("Спутники сколиоза", 
        options=["Деформация грудной клетки", "Деформация таза"], 
        index=None, label_visibility="collapsed", key="scoliosis_comp_radio")
 
    st.divider()
    
    st.markdown("**Тугоподвижность и контрактуры:**")
    contracture_options = [
        "Тугоподвижность голеностопных суставов", "Контрактуры голеностопных суставов",
        "Тугоподвижность тазобедренных суставов", "Контрактуры тазобедренных суставов",
        "Тугоподвижность коленных суставов", "Контрактуры коленных суставов",
        "Тугоподвижность лучезапястных суставов", "Контрактуры лучезапястных суставов",
        "Тугоподвижность локтевых суставов", "Контрактуры локтевых суставов",
        "Контрактуры нижнечелюстно-височных суставов"
    ]
    contractures = st.multiselect("Тугоподвижность и контрактуры", options=contracture_options, label_visibility="collapsed", 
                                  placeholder="Выберите признаки", key="contractures_multiselect")
    
    st.markdown("**Кардиомиопатия:**")
    cardio_options = [
        "Синусовая тахикардия", "Повышение АД", "Нормальное АД",
        "Нестабильное АД (сниженное/нормальное АД/повышенное)", "Дилатационная кардиомиопатия",
        "Снижение глобальной сократительной способности миокарда", "Снижение фракции выброса",
        "Отеки / Сердечная недостаточность"
    ]
    cardiomyopathy = st.multiselect("Кардиомиопатия", options=cardio_options, label_visibility="collapsed", 
                                    placeholder="Выберите признаки", key="cardio_multiselect")

    st.divider()

    submitted = st.form_submit_button("🔍 Рассчитать стадию", type="primary", use_container_width=True)

if submitted:
    payload = {
        "Возраст": age,
        "Способность пациента к ходьбе и бегу": walk_ability,
        "Баллы мускулатуры (верхние конечности)": muscle_upper,
        "Баллы мускулатуры (нижние конечности)": muscle_lower,
        "Применение миопатических приемов Говерса при вставании с пола (использование перекаты)": gowers_roll,
        "Применение миопатических приемов Говерса при вставании с пола (использование опоры)": gowers_support,
        "Тест с шестиминутной ходьбой": six_min_walk,
        "Приседания": squats,
        "Подпрыгивания (вверх на 2-х ногах)": jump_2leg,
        "Подпрыгивания (вверх на 1-ой ноге)": jump_1leg,
        "Ходьба по лестнице (подъем)": stair_up,
        "Ходьба по лестнице (спуск)": stair_down,
        "Особенности позы пациента, обусловленные контрактурами (функциональные нарушения верхней конечности)": upper_func,
        "Особенности позы пациента, обусловленные контрактурами (положение пациента)": position,
        "Гиперлордоз": hyperlordosis,
        "Сколиоз (стадии сколиоза по его выраженности)": scoliosis_stage,
        "Сколиоз (спутники сколиотической деформации)": scoliosis_comp,
        "Тугоподвижность и контрактуры (нарушения суставной подвижности)": contractures,
        "Кардиомиопатия": cardiomyopathy
    }

    with st.spinner("⏳ Модель анализирует данные..."):
        try:
            resp = requests.post(API_URL, json=payload, timeout=10)
            resp.raise_for_status()
            res = resp.json()

            # Основная стадия
            st.success(f"✅ Предсказанная стадия: **{res['stage']}** (уверенность: {res['confidence']:.1%})")
            st.progress(res['confidence'])

            # График вероятностей
            st.markdown("### 📊 Распределение вероятностей")
            proba_df = pd.DataFrame({
                "Стадия": [str(k) for k in res["probabilities"].keys()],
                "Вероятность": list(res["probabilities"].values())
            })
            fig_proba = px.bar(proba_df, x="Стадия", y="Вероятность", 
                               color="Вероятность", color_continuous_scale="Blues", text_auto=".2%")
            fig_proba.update_layout(showlegend=False, yaxis_range=[0, 1], margin=dict(l=20, r=20, t=30, b=30))
            st.plotly_chart(fig_proba, width='stretch')

            # SHAP-объяснение
            if res.get("shap_explanation"):
                st.markdown("### 🔍 Топ-3 признака, повлиявших на решение")
                for i, item in enumerate(res["shap_explanation"], 1):
                    st.markdown(f"- **{item['feature']}** (влияние: `{item['impact']:.3f}`)")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка связи с сервером: {e}")
            st.code("Проверьте, что FastAPI запущен: uvicorn api.main:app --reload")