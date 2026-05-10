"""
ml_pipeline.py
AeroNet Lite - Module 5
BS Data Science AI Semester Project - SP2026

Importable ML module. Trains demand forecasting (regression) and anomaly
detection (classification) models on first import, then exposes three
public symbols for the rest of the project:

    get_demand_forecast(hour, weather, traffic, area, ...) -> float
    detect_anomaly(battery_drop, speed, route_deviation,
                   altitude_change, speed_change)          -> str
    DEMAND_BY_AREA                                         -> dict

Usage in main.py or any other module
--------------------------------------
    from ml_pipeline import get_demand_forecast, detect_anomaly, DEMAND_BY_AREA

    # Grid initialisation (Module 1)
    for cell in grid.cells:
        cell['demand'] = DEMAND_BY_AREA.get(cell['zone'], 120.0)

    # Fleet selection hint (Module 2)
    peak = get_demand_forecast(hour=17, weather='Sunny',
                               traffic='High', area='Urban')

    # Anomaly check inside simulation loop (Module 4 / main.py step 18)
    result = detect_anomaly(battery_drop=35, speed=13,
                            route_deviation=1.0,
                            altitude_change=0.5, speed_change=1.0)
    if result != 'Normal':
        print(f'ALERT: {result}')
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score
)

warnings.filterwarnings('ignore')
np.random.seed(42)


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_REGRESSION_FEATURES = [
    'Agent_Age', 'Agent_Rating',
    'Distance_km',
    'Order_Hour', 'Day_of_Week', 'Month',
    'Weather_enc', 'Traffic_enc', 'Area_enc', 'Vehicle_enc'
]

_TELE_FEATURES = [
    'battery_drop', 'speed', 'route_deviation',
    'altitude_change', 'speed_change'
]

_DATA_PATH = 'data/raw/amazon_delivery.csv'   # update if CSV lives elsewhere

_N_PER_CLASS = 500   # synthetic telemetry samples per anomaly class


# ---------------------------------------------------------------------------
# Internal label encoders (populated during _train_demand_model)
# ---------------------------------------------------------------------------

_le_weather = LabelEncoder()
_le_traffic = LabelEncoder()
_le_area    = LabelEncoder()
_le_vehicle = LabelEncoder()
_le_label   = LabelEncoder()

_demand_model     = None   # best regression model
_demand_model_name = None
_anomaly_model    = None   # best classification model
_anomaly_model_name = None


# ---------------------------------------------------------------------------
# Part A - Demand Forecasting
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _load_and_clean(path):
    df = pd.read_csv(path)

    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
    df['Agent_Rating'] = df['Agent_Rating'].fillna(df['Agent_Rating'].median())
    df = df[df['Traffic'] != 'NaN']

    df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True, errors='coerce')
    df['Day_of_Week'] = df['Order_Date'].dt.dayofweek
    df['Month']       = df['Order_Date'].dt.month
    df['Order_Hour']  = pd.to_datetime(
        df['Order_Time'], format='%H:%M:%S', errors='coerce'
    ).dt.hour

    df['Distance_km'] = _haversine(
        df['Store_Latitude'], df['Store_Longitude'],
        df['Drop_Latitude'],  df['Drop_Longitude']
    )
    return df


def _train_demand_model():
    global _demand_model, _demand_model_name, DEMAND_BY_AREA

    try:
        df = _load_and_clean(_DATA_PATH)
    except FileNotFoundError:
        print(f'[ml_pipeline] WARNING: {_DATA_PATH} not found.')
        print('  Demand model not trained. get_demand_forecast() will return 120.0.')
        DEMAND_BY_AREA = {
            'Urban': 120.0, 'Metropolitian': 130.0,
            'Semi-Urban': 110.0, 'Other': 100.0
        }
        _demand_model = None
        return

    df['Weather_enc'] = _le_weather.fit_transform(df['Weather'])
    df['Traffic_enc'] = _le_traffic.fit_transform(df['Traffic'])
    df['Area_enc']    = _le_area.fit_transform(df['Area'])
    df['Vehicle_enc'] = _le_vehicle.fit_transform(df['Vehicle'])

    df_model = df[_REGRESSION_FEATURES + ['Delivery_Time']].dropna()
    X = df_model[_REGRESSION_FEATURES]
    y = df_model['Delivery_Time']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    mae_lr = mean_absolute_error(y_test, lr.predict(X_test))

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    mae_rf = mean_absolute_error(y_test, rf.predict(X_test))

    if mae_rf <= mae_lr:
        _demand_model      = rf
        _demand_model_name = 'Random Forest Regressor'
        best_mae  = mae_rf
        best_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
    else:
        _demand_model      = lr
        _demand_model_name = 'Linear Regression'
        best_mae  = mae_lr
        best_rmse = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))

    area_demand   = df.groupby('Area')['Delivery_Time'].mean().round(2)
    DEMAND_BY_AREA = area_demand.to_dict()

    print('[ml_pipeline] Demand model trained.')
    print(f'  Model : {_demand_model_name}')
    print(f'  MAE   : {best_mae:.2f} min  |  RMSE : {best_rmse:.2f} min')
    print(f'  DEMAND_BY_AREA : {DEMAND_BY_AREA}')

    # make DEMAND_BY_AREA visible at module level
    # self-reference removed; DEMAND_BY_AREA is already updated as a global


# ---------------------------------------------------------------------------
# Part B - Anomaly Detection
# ---------------------------------------------------------------------------

def _make_telemetry():
    n = _N_PER_CLASS

    normal = pd.DataFrame({
        'battery_drop'    : np.random.normal(5, 3, n).clip(0, 15),
        'speed'           : np.random.normal(15, 4, n).clip(3, 30),
        'route_deviation' : np.random.normal(2, 1.5, n).clip(0, 8),
        'altitude_change' : np.random.normal(0, 2, n).clip(-6, 6),
        'speed_change'    : np.random.normal(0, 2, n).clip(-7, 7),
        'label'           : 'Normal'
    })

    battery = pd.DataFrame({
        'battery_drop'    : np.random.normal(12, 5, n).clip(2, 25),
        'speed'           : np.random.normal(13, 4, n).clip(3, 28),
        'route_deviation' : np.random.normal(3, 2, n).clip(0, 10),
        'altitude_change' : np.random.normal(1, 2.5, n).clip(-6, 8),
        'speed_change'    : np.random.normal(1, 2.5, n).clip(-7, 9),
        'label'           : 'Battery_Anomaly'
    })

    route = pd.DataFrame({
        'battery_drop'    : np.random.normal(6, 3, n).clip(0, 16),
        'speed'           : np.random.normal(19, 5, n).clip(3, 33),
        'route_deviation' : np.random.normal(8, 4, n).clip(1, 20),
        'altitude_change' : np.random.normal(1, 2.5, n).clip(-6, 8),
        'speed_change'    : np.random.normal(2, 3, n).clip(-7, 10),
        'label'           : 'Route_Anomaly'
    })

    spike = pd.DataFrame({
        'battery_drop'    : np.random.normal(6, 3.5, n).clip(0, 16),
        'speed'           : np.random.normal(16, 5, n).clip(3, 32),
        'route_deviation' : np.random.normal(3, 2.5, n).clip(0, 11),
        'altitude_change' : np.random.normal(10, 6, n).clip(-2, 28),
        'speed_change'    : np.random.normal(12, 7, n).clip(-3, 35),
        'label'           : 'Sensor_Spike'
    })

    telemetry = pd.concat([normal, battery, route, spike], ignore_index=True)
    return telemetry.sample(frac=1, random_state=42).reset_index(drop=True)


def _train_anomaly_model():
    global _anomaly_model, _anomaly_model_name

    telemetry = _make_telemetry()
    telemetry['label_enc'] = _le_label.fit_transform(telemetry['label'])

    X = telemetry[_TELE_FEATURES]
    y = telemetry['label_enc']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        'Decision Tree' : DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'KNN'           : KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes'   : GaussianNB()
    }

    best_acc  = -1
    best_name = None
    best_clf  = None

    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        if acc > best_acc:
            best_acc  = acc
            best_name = name
            best_clf  = clf

    _anomaly_model      = best_clf
    _anomaly_model_name = best_name

    print('[ml_pipeline] Anomaly model trained.')
    print(f'  Model    : {_anomaly_model_name}')
    print(f'  Accuracy : {best_acc:.4f}')


# ---------------------------------------------------------------------------
# Module-level initialisation - runs once on import
# ---------------------------------------------------------------------------

# Initialise DEMAND_BY_AREA with safe defaults before training attempts.
# _train_demand_model() will overwrite this if the CSV is found.
DEMAND_BY_AREA = {
    'Urban': 120.0,
    'Metropolitian': 130.0,
    'Semi-Urban': 110.0,
    'Other': 100.0
}

print('[ml_pipeline] Initialising Module 5 ML Pipeline...')
_train_demand_model()
_train_anomaly_model()
print('[ml_pipeline] Ready.\n')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_demand_forecast(
    hour,
    weather,
    traffic,
    area,
    agent_age=30,
    agent_rating=4.5,
    distance_km=5.0,
    day_of_week=0,
    month=6,
    vehicle='motorcycle'
):
    """
    Predict delivery demand (delivery-time proxy) given contextual inputs.

    Parameters
    ----------
    hour         : int   - Hour of the day (0-23)
    weather      : str   - One of: Sunny, Cloudy, Stormy, Fog, Windy, Sandstorms
    traffic      : str   - One of: Low, Medium, High, Jam
    area         : str   - One of: Urban, Metropolitan, Semi-Urban, Other
    agent_age    : int   - Default 30
    agent_rating : float - Default 4.5
    distance_km  : float - Estimated delivery distance in km. Default 5.0
    day_of_week  : int   - 0=Monday, 6=Sunday. Default 0
    month        : int   - 1-12. Default 6
    vehicle      : str   - Vehicle type. Default 'motorcycle'

    Returns
    -------
    float
        Predicted delivery time in minutes (used as demand proxy).
        Returns 120.0 if the demand model was not trained (CSV missing).
    """
    if _demand_model is None:
        return 120.0

    try:
        w_enc = _le_weather.transform([weather])[0]
    except ValueError:
        w_enc = 0
    try:
        t_enc = _le_traffic.transform([traffic])[0]
    except ValueError:
        t_enc = 1
    try:
        a_enc = _le_area.transform([area])[0]
    except ValueError:
        a_enc = 0
    try:
        v_enc = _le_vehicle.transform([vehicle])[0]
    except ValueError:
        v_enc = 0

    features = np.array([[
        agent_age, agent_rating, distance_km,
        hour, day_of_week, month,
        w_enc, t_enc, a_enc, v_enc
    ]])
    return float(_demand_model.predict(features)[0])


def detect_anomaly(battery_drop, speed, route_deviation, altitude_change, speed_change):
    """
    Classify a drone telemetry reading as Normal or one of three anomaly types.

    Parameters
    ----------
    battery_drop     : float - Battery percentage dropped in this step
    speed            : float - Current drone speed (m/s or grid units per step)
    route_deviation  : float - Cells deviated from planned route
    altitude_change  : float - Change in altitude since last step
    speed_change     : float - Change in speed since last step

    Returns
    -------
    str
        One of: 'Normal', 'Battery_Anomaly', 'Route_Anomaly', 'Sensor_Spike'
    """
    features = np.array([[
        battery_drop, speed, route_deviation,
        altitude_change, speed_change
    ]])
    pred_enc = _anomaly_model.predict(features)[0]
    return _le_label.inverse_transform([pred_enc])[0]


def demand_forecast_summary():
    """Print a short summary of the demand forecasting model."""
    print('Demand Forecasting Model')
    print(f'  Model : {_demand_model_name}')
    print(f'  DEMAND_BY_AREA : {DEMAND_BY_AREA}')


def anomaly_model_summary():
    """Print a short summary of the anomaly detection model."""
    print('Anomaly Detection Model')
    print(f'  Model   : {_anomaly_model_name}')
    print(f'  Classes : {list(_le_label.classes_)}')


# ---------------------------------------------------------------------------
# Quick self-test when run directly: python src/ml_pipeline.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 55)
    print('ml_pipeline.py  -  self-test')
    print('=' * 55)

    demand_forecast_summary()
    print()
    anomaly_model_summary()
    print()

    print('get_demand_forecast() tests:')
    print(f'  Sunny, Low, Urban, 8am       -> {get_demand_forecast(8, "Sunny", "Low", "Urban"):.1f} min')
    print(f'  Stormy, Jam, Metro, 5pm      -> {get_demand_forecast(17, "Stormy", "Jam", "Metropolitian"):.1f} min')
    print(f'  Fog, Medium, Semi-Urban, 12pm -> {get_demand_forecast(12, "Fog", "Medium", "Semi-Urban"):.1f} min')
    print()

    print('detect_anomaly() tests:')
    print(f'  Normal flight        : {detect_anomaly(2.0,  15.0, 0.5,  0.1,  0.5)}')
    print(f'  Battery drop         : {detect_anomaly(30.0, 14.0, 1.0,  0.0,  0.5)}')
    print(f'  Far from route       : {detect_anomaly(3.0,  16.0, 15.0, 0.5,  1.0)}')
    print(f'  Altitude+speed spike : {detect_anomaly(2.0,  15.0, 0.5,  30.0, 40.0)}')
    print()

    print('DEMAND_BY_AREA:')
    for zone, val in DEMAND_BY_AREA.items():
        print(f'  {zone:20s} : {val:.2f}')

    print()
    print('Simulation steps 15-19 walkthrough:')

    forecast_15 = get_demand_forecast(15, 'Cloudy', 'Medium', 'Urban')
    print(f'Step 15: Demand forecast -> {forecast_15:.1f} min', end='  ')
    print('(dispatch extra drone)' if forecast_15 > 130 else '(fleet sufficient)')

    forecast_16 = get_demand_forecast(16, 'Sunny', 'High', 'Metropolitian')
    print(f'Step 16: Metro forecast  -> {forecast_16:.1f} min. Adjusting drone priority.')

    print('Step 17: Cross-val handled in anomaly_classifier.ipynb.')

    anomaly_18 = detect_anomaly(35, 13, 1.0, 0.5, 1.0)
    print(f'Step 18: Drone D3 anomaly -> {anomaly_18}')
    if anomaly_18 != 'Normal':
        print(f'         ALERT: {anomaly_18} on Drone D3. Initiating return-to-hub.')

    actions = {
        'Battery_Anomaly' : 'Drone D3 returning to nearest hub.',
        'Route_Anomaly'   : 'Recalculating path via A* from current position.',
        'Sensor_Spike'    : 'Grounding Drone D3 pending diagnostics.',
        'Normal'          : 'Drone D3 continues on planned route.'
    }
    print(f'Step 19: {actions.get(anomaly_18, "Unknown anomaly type.")}')

    print('=' * 55)