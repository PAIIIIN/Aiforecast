import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
import sys
import locale

# Пытаемся установить российскую локаль, если доступно
try:
    locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
except locale.Error:
    # Если локаль недоступна — оставляем системную по умолчанию
    locale.setlocale(locale.LC_TIME, '')

# Set page configuration
st.set_page_config(
    page_title="Прогноз землетрясений в Алматинском регионе",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("Прогноз землетрясений в Алматинском регионе")
st.markdown("""
Это приложение предоставляет прогнозы землетрясений для Алматинского региона на основе анализа исторических данных и моделей машинного обучения.
Прогнозы включают ожидаемое количество землетрясений и их максимальную магнитуду по месяцам.
""")

# Load models and scaler
@st.cache_resource
def load_models():
    try:
        # Adjust paths for deployment
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(base_path, "models")
        
        count_model = joblib.load(os.path.join(models_path, 'earthquake_count_model.pkl'))
        mag_model = joblib.load(os.path.join(models_path, 'earthquake_magnitude_model.pkl'))
        scaler = joblib.load(os.path.join(models_path, 'feature_scaler.pkl'))
        
        # Load feature list
        with open(os.path.join(models_path, 'feature_list.txt'), 'r') as f:
            feature_cols = f.read().splitlines()
            
        return count_model, mag_model, scaler, feature_cols
    except Exception as e:
        st.error(f"Ошибка при загрузке моделей: {e}")
        return None, None, None, None

# Load historical data
@st.cache_data
def load_data():
    try:
        # Adjust paths for deployment
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, "data")
        models_path = os.path.join(base_path, "models")
        
        # Load processed data
        all_eq = pd.read_csv(os.path.join(data_path, 'all_almaty_region_earthquakes.csv'))
        all_eq['time'] = pd.to_datetime(all_eq['time'], format='ISO8601')
        
        # Load monthly stats
        monthly_stats = pd.read_csv(os.path.join(models_path, 'monthly_stats_prepared.csv'))
        monthly_stats['date'] = pd.to_datetime(monthly_stats['date'], format='ISO8601')
        
        # Load existing predictions if available
        try:
            future_predictions = pd.read_csv(os.path.join(models_path, 'future_predictions.csv'))
            future_predictions['date'] = pd.to_datetime(future_predictions['date'], format='ISO8601')
            
            # Add risk_level column if it doesn't exist
            if 'risk_level' not in future_predictions.columns:
                # Calculate risk level based on count and magnitude
                def calculate_risk_level(row):
                    if row['predicted_count'] == 0 or row['predicted_max_mag'] < 4.0:
                        return "Низкий"
                    elif row['predicted_count'] < 5 and row['predicted_max_mag'] < 5.0:
                        return "Средний"
                    elif row['predicted_count'] < 10 and row['predicted_max_mag'] < 6.0:
                        return "Высокий"
                    else:
                        return "Очень Высокий"
                
                future_predictions['risk_level'] = future_predictions.apply(calculate_risk_level, axis=1)
                
            # Add risk_color column if it doesn't exist
            if 'risk_color' not in future_predictions.columns:
                risk_color_map = {
                    "Низкий": "green",
                    "Moderate": "yellow",
                    "Высокий": "orange",
                    "Very Высокий": "red"
                }
                future_predictions['risk_color'] = future_predictions['risk_level'].map(risk_color_map)
        except:
            future_predictions = None
            
        return all_eq, monthly_stats, future_predictions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Function to make predictions
def predict_future_months(count_model, mag_model, scaler, feature_cols, monthly_stats, months_ahead=12):
    # Get the last date in our dataset
    last_date = monthly_stats['date'].max()
    
    # Create a dataframe for future months
    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months_ahead)]
    future_df = pd.DataFrame({'date': future_dates})
    future_df['year_month'] = future_df['date'].dt.strftime('%Y-%m')
    
    # Initialize with the last known values
    last_row = monthly_stats.iloc[-1].copy()
    
    # Make predictions for each future month
    predictions = []
    
    for i, date in enumerate(future_dates):
        # Create a new row for prediction
        new_row = pd.Series(index=feature_cols)
        
        # Add month features
        new_row['month_sin'] = np.sin(2 * np.pi * date.month/12)
        new_row['month_cos'] = np.cos(2 * np.pi * date.month/12)
        
        # For the first prediction, use the last known values
        if i == 0:
            for lag in [1, 3, 6]:
                new_row[f'count_lag_{lag}'] = monthly_stats.iloc[-lag]['monthly_count']
                new_row[f'avg_mag_lag_{lag}'] = monthly_stats.iloc[-lag]['monthly_avg_mag']
                new_row[f'max_mag_lag_{lag}'] = monthly_stats.iloc[-lag]['monthly_max_mag']
            
            # Use the last rolling statistics
            for window in [3, 6]:
                new_row[f'count_roll_{window}'] = last_row[f'count_roll_{window}']
                new_row[f'avg_mag_roll_{window}'] = last_row[f'avg_mag_roll_{window}']
                new_row[f'max_mag_roll_{window}'] = last_row[f'max_mag_roll_{window}']
        else:
            # Update lags based on previous predictions
            for lag in [1, 3, 6]:
                if i >= lag:
                    new_row[f'count_lag_{lag}'] = predictions[i-lag]['predicted_count']
                    new_row[f'avg_mag_lag_{lag}'] = predictions[i-lag]['predicted_avg_mag']
                    new_row[f'max_mag_lag_{lag}'] = predictions[i-lag]['predicted_max_mag']
                else:
                    new_row[f'count_lag_{lag}'] = monthly_stats.iloc[-(lag-i)]['monthly_count']
                    new_row[f'avg_mag_lag_{lag}'] = monthly_stats.iloc[-(lag-i)]['monthly_avg_mag']
                    new_row[f'max_mag_lag_{lag}'] = monthly_stats.iloc[-(lag-i)]['monthly_max_mag']
            
            # Update rolling statistics (simplified approach)
            for window in [3, 6]:
                if i < window:
                    # Mix of actual data and predictions
                    actual_counts = [monthly_stats.iloc[-(window-j)]['monthly_count'] for j in range(1, i+1)]
                    pred_counts = [predictions[j]['predicted_count'] for j in range(i)]
                    counts = actual_counts + pred_counts
                    new_row[f'count_roll_{window}'] = np.mean(counts)
                    
                    actual_avg_mags = [monthly_stats.iloc[-(window-j)]['monthly_avg_mag'] for j in range(1, i+1)]
                    pred_avg_mags = [predictions[j]['predicted_avg_mag'] for j in range(i)]
                    avg_mags = actual_avg_mags + pred_avg_mags
                    new_row[f'avg_mag_roll_{window}'] = np.mean(avg_mags)
                    
                    actual_max_mags = [monthly_stats.iloc[-(window-j)]['monthly_max_mag'] for j in range(1, i+1)]
                    pred_max_mags = [predictions[j]['predicted_max_mag'] for j in range(i)]
                    max_mags = actual_max_mags + pred_max_mags
                    new_row[f'max_mag_roll_{window}'] = np.mean(max_mags)
                else:
                    # All predictions
                    counts = [predictions[j]['predicted_count'] for j in range(i-window, i)]
                    new_row[f'count_roll_{window}'] = np.mean(counts)
                    
                    avg_mags = [predictions[j]['predicted_avg_mag'] for j in range(i-window, i)]
                    new_row[f'avg_mag_roll_{window}'] = np.mean(avg_mags)
                    
                    max_mags = [predictions[j]['predicted_max_mag'] for j in range(i-window, i)]
                    new_row[f'max_mag_roll_{window}'] = np.mean(max_mags)
        
        # Extract features for prediction
        features = new_row.values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Make predictions
        predicted_count = max(0, round(count_model.predict(features_scaled)[0]))  # Ensure non-negative
        predicted_max_mag = mag_model.predict(features_scaled)[0]
        
        # Estimate average magnitude (simplified)
        if predicted_count > 0:
            predicted_avg_mag = predicted_max_mag * 0.8  # Simplified estimation
        else:
            predicted_avg_mag = 0
        
        # Calculate risk level based on count and magnitude
        if predicted_count == 0 or predicted_max_mag < 4.0:
            risk_level = "Низкий"
            risk_color = "green"
        elif predicted_count < 5 and predicted_max_mag < 5.0:
            risk_level = "Moderate"
            risk_color = "yellow"
        elif predicted_count < 10 and predicted_max_mag < 6.0:
            risk_level = "Высокий"
            risk_color = "orange"
        else:
            risk_level = "Very Высокий"
            risk_color = "red"
        
        # Store predictions
        predictions.append({
            'date': date,
            'year_month': date.strftime('%Y-%m'),
            'predicted_count': predicted_count,
            'predicted_max_mag': predicted_max_mag,
            'predicted_avg_mag': predicted_avg_mag,
            'risk_level': risk_level,
            'risk_color': risk_color
        })
    
    # Convert to DataFrame
    future_predictions = pd.DataFrame(predictions)
    return future_predictions

# Main application
def main():
    # Load models and data
    count_model, mag_model, scaler, feature_cols = load_models()
    all_eq, monthly_stats, existing_predictions = load_data()
    
    if count_model is None or all_eq is None:
        st.error("Failed to load models or data. Please check the file paths.")
        return
    
# Боковая панель
    st.sidebar.title("Настройки прогноза")

# Опция использовать существующие прогнозы или сгенерировать новые
    use_existing = st.sidebar.checkbox("Использовать существующие прогнозы", value=True if existing_predictions is not None else False)

# Количество месяцев для прогноза
    months_ahead = st.sidebar.slider("Период прогноза (в месяцах)", min_value=3, max_value=24, value=12)

# Генерация прогнозов
    if use_existing and existing_predictions is not None:
        future_predictions = existing_predictions
        st.sidebar.info(f"Используются существующие прогнозы с {existing_predictions['date'].min().strftime('%Y-%m-%d')} по {existing_predictions['date'].max().strftime('%Y-%m-%d')}")
    else:
        with st.spinner("Создание прогнозов..."):
            future_predictions = predict_future_months(count_model, mag_model, scaler, feature_cols, monthly_stats, months_ahead)
        st.sidebar.success("Новые прогнозы успешно созданы!")
    
    # Display information about the data
    st.header("Информация о наборе данных")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего землетрясений", f"{len(all_eq):,}")
    with col2:
        st.metric("Период наблюдений", f"{all_eq['time'].min().strftime('%Y-%m-%d')} — {all_eq['time'].max().strftime('%Y-%m-%d')}")
    with col3:
        st.metric("Радиус региона", "500 км вокруг Алматы")
    
    # Display forecasts
    st.header("Прогноз землетрясений")
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["Таблица прогнозов", "Диаграммы прогнозов", "Исторические данные"])
    


    with tab1:
    # Форматируем таблицу для отображения
        display_df = future_predictions.copy()
        display_df['date'] = display_df['date'].dt.strftime('%B %Y')  # Месяц и год по-русски
        display_df['predicted_count'] = display_df['predicted_count'].astype(int)
        display_df['predicted_max_mag'] = display_df['predicted_max_mag'].round(1)
        
        # Create a styled dataframe
    st.dataframe(
    display_df[['date', 'predicted_count', 'predicted_max_mag', 'risk_level']].rename(
        columns={
            'date': 'Месяц',
            'predicted_count': 'Прогноз количества землетрясений',
            'predicted_max_mag': 'Прогноз максимальной магнитуды',
            'risk_level': 'Уровень риска'
        }
    ),
    use_container_width=True,
    hide_index=True
)
    
    with tab2:
    # Создаём интерактивные графики с помощью Plotly
        fig1 = px.line(
        monthly_stats, x='date', y='monthly_count',
        labels={'date': 'Дата', 'monthly_count': 'Количество землетрясений'},
        title='Историческая статистика землетрясений'
    )
        
        # Add predictions to the chart
        fig1.add_scatter(
        x=future_predictions['date'],
        y=future_predictions['predicted_count'],
        mode='lines+markers',
        name='Прогноз',
        line=dict(dash='dash', color='red')
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Magnitude chart
        fig2 = px.line(
            monthly_stats, x='date', y='monthly_max_mag',
            labels={'date': 'Date', 'monthly_max_mag': 'Maximum Magnitude'},
            title='Исторические максимальные магнитуды'
        )
        
        # Add predictions to the chart
        fig2.add_scatter(
            x=future_predictions['date'],
            y=future_predictions['predicted_max_mag'],
            mode='lines+markers',
            name='Прогноз',
            line=dict(dash='dash', color='red')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Risk level chart
        risk_colors = {
            'Низкий': 'green',
            'Средний': 'yellow',
            'Высокий': 'orange',
            'Очень Высокий': 'red'
        }
        
        fig3 = go.Figure()
        
        for risk in risk_colors:
            risk_data = future_predictions[future_predictions['risk_level'] == risk]
            if not risk_data.empty:
                fig3.add_trace(go.Bar(
                    x=risk_data['date'],
                    y=[1] * len(risk_data),
                    name=risk,
                    marker_color=risk_colors[risk],
                    hovertemplate='%{x}<br>Risk: ' + risk
                ))
        
        fig3.update_layout(
            title='Ежемесячные уровни риска землетрясений',
            xaxis_title='Month',
            yaxis_title='Risk Level',
            barmode='stack',
            showlegend=True
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        # Display historical earthquake data
        st.subheader("Historical Earthquake Data")
        
        # Map of historical earthquakes
        st.subheader("Spatial Distribution of Earthquakes")
        fig = px.scatter_mapbox(
            all_eq, 
            lat="latitude", 
            lon="longitude", 
            color="mag",
            size="mag",
            color_continuous_scale="Viridis",
            size_max=15,
            zoom=5,
            center={"lat": 43.10, "lon": 76.87},
            hover_name="place",
            hover_data=["time", "depth", "mag"],
            title="Earthquakes near Almaty (1970-2024)",
            mapbox_style="open-street-map"
        )
        
        # Add Almaty marker
        fig.add_trace(
            go.Scattermapbox(
                lat=[43.10],
                lon=[76.87],
                mode='markers',
                marker=dict(size=12, color='red'),
                name='Almaty'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Magnitude distribution
        fig = px.histogram(
            all_eq, 
            x="mag", 
            nbins=20,
            title="Magnitude Distribution",
            labels={"mag": "Magnitude", "count": "Number of Earthquakes"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Depth distribution
        fig = px.histogram(
            all_eq, 
            x="depth", 
            nbins=30,
            title="Depth Distribution",
            labels={"depth": "Depth (km)", "count": "Number of Earthquakes"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
# Информация о модели
    st.header("Информация о модели")
    st.markdown("""
### Методология прогнозирования

Модели прогнозирования землетрясений используют машинное обучение для предсказания как количества землетрясений, так и их максимальной магнитуды на ежемесячной основе. Модели обучены на исторических данных о землетрясениях в радиусе 500 км от Алматы.

**Признаки, используемые в моделях:**
- Отстающие значения количества и магнитуд землетрясений за предыдущие месяцы
- Скользящие средние сейсмической активности
- Сезонные компоненты (месяц года)

**Ограничения:**
- Модель основана на исторических закономерностях и может не учитывать неожиданные геологические изменения
- Прогнозы более надёжны для ближайшего будущего; неопределённость увеличивается с течением времени
- Модель предсказывает агрегированные значения по месяцам, а не отдельные события
- Пространственная информация не полностью используется в текущем временном подходе

**Рекомендации по готовности на основе уровней риска:**
- **Низкий риск**: Обычные меры готовности
- **Средний риск**: Пересмотрите планы действий и аварийные запасы
- **Высокий риск**: Повышенное внимание и поддержание связи
- **Очень высокий риск**: Рассмотрите меры предосторожности и повышенную настороженность
""")
    
    # Footer
    st.markdown("---")
    st.markdown("Проект прогнозирования землетрясений для Алматинского региона | Создан на основе исторических данных о землетрясениях с 1970 по 2024 год")

if __name__ == "__main__":
    main()
