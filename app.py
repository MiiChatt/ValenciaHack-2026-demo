"""
Aplicación Streamlit: Optimización del transporte público según demanda horaria con LSTM.
"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    keras = tf.keras
    layers = keras.layers
    TF_DISPONIBLE = True
except ImportError:
    TF_DISPONIBLE = False

st.set_page_config(page_title="Optimización de Transporte", page_icon="🚍", layout="wide")

st.title("🚍 Optimización del Transporte Público según Demanda Horaria con LSTM")
st.markdown("""
Esta aplicación simula la demanda de pasajeros, entrena un modelo predictivo (LSTM o Baseline), 
detecta saturaciones y propone estrategias de refuerzo de flota en tiempo real.
""")

# --- BARRA LATERAL DE CONFIGURACIÓN ---
st.sidebar.header("⚙️ Parámetros del Sistema")
dias = st.sidebar.slider("Días de simulación", min_value=30, max_value=365, value=120, step=30)
umbral = st.sidebar.slider("Umbral de saturación", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
flota_reserva = st.sidebar.number_input("Vehículos de reserva disponibles", min_value=0, max_value=20, value=3)

LINEAS = {
    "L1": (90, 6),
    "L2": (90, 5),
    "L3": (70, 4),
}
VENTANA = 24
FRECUENCIA_MAXIMA = 15
RNG = np.random.default_rng(42)

# --- FUNCIONES DEL PIPELINE ---
def generar_datos_sinteticos(dias_sim):
    horas = pd.date_range("2025-01-01", periods=dias_sim * 24, freq="h")
    filas = []
    for linea, (cap, frec) in LINEAS.items():
        base = cap * frec * RNG.uniform(0.5, 0.8)
        for t in horas:
            h = t.hour
            pico_m = np.exp(-((h - 8) ** 2) / 3.0)
            pico_t = np.exp(-((h - 18) ** 2) / 4.0)
            valle = 0.25 + 0.25 * np.exp(-((h - 13) ** 2) / 20.0)
            factor = valle + 1.0 * pico_m + 0.9 * pico_t
            if t.dayofweek >= 5:
                factor *= 0.55
            pasajeros = base * factor * RNG.normal(1.0, 0.07)
            filas.append((t, linea, max(pasajeros, 0)))
    df = pd.DataFrame(filas, columns=["fecha_hora", "linea", "pasajeros"])
    df["hora"] = df.fecha_hora.dt.hour
    df["dia_semana"] = df.fecha_hora.dt.dayofweek
    df["festivo"] = (df.fecha_hora.dt.dayofyear % 40 == 0).astype(int)
    return df

def variables_ciclicas(df):
    df = df.copy()
    df["hora_sin"] = np.sin(2 * np.pi * df.hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df.hora / 24)
    df["dia_sin"] = np.sin(2 * np.pi * df.dia_semana / 7)
    df["dia_cos"] = np.cos(2 * np.pi * df.dia_semana / 7)
    return df

CARACTERISTICAS = ["pasajeros", "hora_sin", "hora_cos", "dia_sin", "dia_cos", "festivo"]

def crear_secuencias(matriz, ventana):
    X, y = [], []
    for i in range(len(matriz) - ventana):
        X.append(matriz[i:i + ventana])
        y.append(matriz[i + ventana, 0])
    return np.array(X), np.array(y)

def construir_modelo(n_caracteristicas):
    modelo = keras.Sequential([
        layers.Input(shape=(VENTANA, n_caracteristicas)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    modelo.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return modelo

class PredictorBaseline:
    def predict(self, X, verbose=0):
        return X[:, -24, 0].reshape(-1, 1)

def entrenar_y_predecir(df_linea):
    df_linea = variables_ciclicas(df_linea).reset_index(drop=True)
    escalador = MinMaxScaler()
    datos = escalador.fit_transform(df_linea[CARACTERISTICAS])
    X, y = crear_secuencias(datos, VENTANA)

    corte = int(len(X) * 0.8)
    X_ent, y_ent, X_test, y_test = X[:corte], y[:corte], X[corte:], y[corte:]

    if TF_DISPONIBLE:
        modelo = construir_modelo(X.shape[2])
        modelo.fit(X_ent, y_ent, epochs=10, batch_size=64, validation_split=0.1,
                   callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                   verbose=0)
    else:
        modelo = PredictorBaseline()

    pred_norm = modelo.predict(X_test, verbose=0).flatten()
    minimo, maximo = escalador.data_min_[0], escalador.data_max_[0]
    pred = pred_norm * (maximo - minimo) + minimo
    real = y_test * (maximo - minimo) + minimo
    mae = np.mean(np.abs(pred - real))

    fechas = df_linea.fecha_hora.iloc[VENTANA + corte:].reset_index(drop=True)
    return pd.DataFrame({"fecha_hora": fechas, "real": real, "prevista": pred}), mae

def detectar_saturacion(resultado, linea):
    cap_veh, frec = LINEAS[linea]
    capacidad_hora = cap_veh * frec
    r = resultado.copy()
    r["linea"] = linea
    r["capacidad_hora"] = capacidad_hora
    r["ocupacion"] = r.prevista / capacidad_hora
    r["saturada"] = r.ocupacion > umbral
    return r

def proponer_optimizacion(saturaciones):
    sat = saturaciones[saturaciones.saturada].copy()
    if sat.empty:
        return pd.DataFrame(columns=["fecha_hora", "linea", "ocupacion", "frec_actual",
                                   "frec_necesaria", "veh_extra", "accion"])

    sat["frec_actual"] = sat.linea.map(lambda l: LINEAS[l][1])
    cap_veh = sat.linea.map(lambda l: LINEAS[l][0])
    sat["frec_necesaria"] = np.ceil(sat.prevista / (cap_veh * umbral)).clip(upper=FRECUENCIA_MAXIMA)
    sat["veh_extra"] = (sat.frec_necesaria - sat.frec_actual).astype(int)

    acciones = []
    for _, grupo in sat.groupby("fecha_hora"):
        reserva = flota_reserva
        for idx, fila in grupo.sort_values("ocupacion", ascending=False).iterrows():
            if fila.veh_extra <= reserva:
                acciones.append((idx, "Aumentar frecuencia"))
                reserva -= fila.veh_extra
            else:
                acciones.append((idx, "Redistribuir vehículos / modificar recorrido"))
    sat["accion"] = pd.Series(dict(acciones))
    return sat[["fecha_hora", "linea", "ocupacion", "frec_actual",
                "frec_necesaria", "veh_extra", "accion"]]

# --- INTERFAZ PRINCIPAL ---
if st.button("🚀 Ejecutar Simulación y Optimización", type="primary"):
    with st.spinner("Procesando datos y ejecutando modelos..."):
        df = generar_datos_sinteticos(dias)
        todas = []
        maes = {}
        resultados_dict = {}

        for linea in LINEAS:
            resultado, mae = entrenar_y_predecir(df[df.linea == linea])
            maes[linea] = mae
            resultados_dict[linea] = resultado
            todas.append(detectar_saturacion(resultado, linea))

        saturaciones = pd.concat(todas, ignore_index=True)
        propuesta = proponer_optimizacion(saturaciones)

    st.success("¡Simulación completada con éxito!")

    # Métricas de error
    st.subheader("📊 Métricas de Error por Línea (MAE)")
    cols = st.columns(len(LINEAS))
    for i, (linea, mae_val) in enumerate(maes.items()):
        cols[i].metric(
            label=f"Línea {linea}", 
            value=f"{mae_val:.1f} pas/h", 
            delta="TensorFlow activo" if TF_DISPONIBLE else "Baseline (TF inactivo)"
        )

    if not TF_DISPONIBLE:
        st.warning("⚠️ TensorFlow no está instalado. Si instalas TensorFlow (`pip install tensorflow`), el error disminuirá notablemente al usar redes LSTM reales.")

    # Pestañas de visualización
    tab1, tab2, tab3 = st.tabs(["📈 Gráficos de Demanda", "🚨 Franjas Saturadas", "💡 Propuesta de Optimización"])

    with tab1:
        st.subheader("Comparativa Demanda Real vs Prevista")
        linea_sel = st.selectbox("Selecciona la línea:", list(LINEAS.keys()))
        df_plot = resultados_dict[linea_sel].set_index("fecha_hora")
        st.line_chart(df_plot[["real", "prevista"]])

    with tab2:
        st.subheader("Detalle de Franjas Saturadas")
        st.dataframe(saturaciones[saturaciones.saturada].reset_index(drop=True), use_container_width=True)

    with tab3:
        st.subheader("Resumen de Acciones de Optimización")
        if len(propuesta) > 0:
            resumen = (propuesta.assign(hora=propuesta.fecha_hora.dt.hour)
                       .groupby(["linea", "hora"])
                       .agg(ocupacion_media=("ocupacion", "mean"),
                            veh_extra_medio=("veh_extra", "mean"),
                            veces=("ocupacion", "size"))
                       .round(2).reset_index())
            st.dataframe(resumen, use_container_width=True)
            
            csv = propuesta.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Propuesta Completa en CSV",
                data=csv,
                file_name="propuesta_optimizacion.csv",
                mime="text/csv",
            )
        else:
            st.info("No se han detectado franjas saturadas con los parámetros actuales.")
else:
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en **Ejecutar Simulación y Optimización** para abrir tu aplicación.")