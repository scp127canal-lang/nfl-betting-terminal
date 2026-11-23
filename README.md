# 🏈 NFL +EV Betting Terminal

Terminal de analítica para apuestas de la NFL que busca **valor esperado positivo (+EV)**. Compara las cuotas reales de las casas (The-Odds-API) con probabilidades propias calculadas por **simulación Monte Carlo (10,000 iteraciones)** sobre el historial de cada jugador y equipo (nflverse).

- Interfaz **Streamlit** de pantalla única, en modo oscuro.
- API **FastAPI** con chat de **Gemini** (function calling).
- Actualización diaria de cuotas a las 8:00 AM los días con partido.

> Todas las cuotas se muestran en **formato decimal** (ej. `1.91`).

---

## Funcionalidades

**Partidos**
- Tarjetas por partido con logos, marcador proyectado por simulación y cuotas de moneyline, spread y total.
- Un badge `🟢 +EV` aparece cuando el modelo encuentra valor.

**Analítica de jugadores** (al tocar un partido)
- Catálogo de jugadores con props y vista enfocada en uno solo, con botón para volver al resto.
- Categorías ⚽ Touchdowns · 🏈 Pases · 🏃 Acarreos · 🎯 Recepciones · 🔀 Todos. Cubre todos los mercados de jugador de The-Odds-API: yardas, TDs, intentos, completos, intercepciones, recepciones, Anytime TD y Primer TD, con sus líneas alternas.
- Métricas de los últimos 10 partidos (L10) y gráfico de *hit-rate* contra la línea activa.
- Tabla de escaleras alternas con la probabilidad Monte Carlo, el EV y la *Fair Odds* de cada línea.
- 🎯 **Línea Más Probable**: el resultado que más se repite en las 10k simulaciones (moda), comparado con la línea de la casa.
- Edición manual de línea y cuota para evaluar cualquier oferta.

**Combinadas sugeridas por riesgo**

| Riesgo | Probabilidad | Cuota |
|---|---|---|
| 🟢 Bajo | > 50% | 1.80 – 3.20 |
| 🟡 Moderado | 30 – 50% | 3.20 – 7.50 |
| 🔴 Alto | < 30% | > 7.50 (3+ selecciones) |

- Solo usan **líneas principales fijas**, con la cuota de consenso entre casas. Nunca usan escaleras alternas.
- Modos 🔀 Mixtos · 👤 Solo Jugadores · 🏟️ Solo Equipos.
- Las selecciones del mismo partido (🔥 SGP) se simulan de forma conjunta, teniendo en cuenta la correlación entre ellas.

**Boleta de apuestas** (barra lateral)
- Probabilidad conjunta MC, EV y stake sugerido con **Kelly al 25%** (tope del 5% del bankroll).
- Campo **🎯 Cuota Real Ajustada** para meter la cuota que de verdad ofrece la casa en combinadas del mismo partido. El EV y el pago se recalculan con ella.

**Salvaguardas del modelo**
- **Realism Guard**: rechaza líneas más de un 15% por encima del máximo del jugador en sus últimos 10 partidos.
- Distribución **Gamma cero-inflada** para yardas y **Poisson / Binomial negativa** para conteos (TDs, recepciones, INT…).

---

## Estructura

| Archivo | Rol |
|---|---|
| `app.py` | Interfaz Streamlit (vistas de partidos y de jugador) |
| `ui_components.py` | Componentes visuales, CSS, boleta y gráficos |
| `data_pipeline.py` | Datos de nflverse: stats semanales, calendario, semana activa |
| `odds_pipeline.py` | The-Odds-API: catálogo de mercados, conversión a decimal, caché |
| `prediction_engine.py` | Monte Carlo, Realism Guard, EV, Fair Odds, línea más probable |
| `bet_builder.py` | Patas, correlaciones, simulación conjunta, niveles de riesgo, Kelly |
| `scheduler.py` | Descarga diaria de cuotas (APScheduler) |
| `main.py` | API FastAPI y chat con Gemini |
| `run.py` | Arranca la API y Streamlit juntos |
| `config.py` | Variables de entorno, paleta de colores, `american_to_decimal` |

---

## Instalación

Requiere **Python 3.11**.

```bash
git clone https://github.com/scp127canal-lang/nfl-betting-terminal.git
cd nfl-betting-terminal
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copia `.env.example` a `.env` y añade tus claves:

```ini
GEMINI_API_KEY=tu_api_key_aqui          # https://aistudio.google.com/apikey
THE_ODDS_API_KEY=tu_odds_api_key_aqui   # https://the-odds-api.com (también se acepta ODDS_API_KEY)
MORNING_REFRESH_HOUR=8                  # hora de la descarga diaria (hora local)
GEMINI_MODEL=gemini-3.8-flash
# ODDS_PLAYER_MARKETS=core              # opcional: menos mercados para gastar menos créditos
# APP_PASSWORD=...                      # opcional: pide contraseña al abrir la app
```

`.env` está en `.gitignore`: **nunca subas tus claves al repositorio.**

## Uso

```bash
python run.py
```

- Interfaz: http://localhost:8501
- API: http://localhost:8000 (documentación interactiva en `/docs`)

`run.py` elige otro puerto si el 8000 o el 8501 están ocupados. Para abrir solo la interfaz: `streamlit run app.py`.

### Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado, semana activa y próxima actualización de cuotas |
| GET | `/matchups` | Partidos de la semana con proyecciones |
| POST | `/predict/prop` | Evalúa un prop: línea, cuota y lado → probabilidad, EV, Fair Odds |
| POST | `/parlay/evaluate` | Probabilidad conjunta y EV de una combinada |
| GET | `/parlays?risk=BAJO` | Combinadas sugeridas por nivel de riesgo |
| POST | `/odds/refresh` | Fuerza la descarga de cuotas |
| POST | `/chat` | Asistente Gemini que consulta partidos, props y combinadas |

---

## Créditos de The-Odds-API

La API cobra **1 crédito por mercado y región**:

| Petición | Coste aprox. |
|---|---|
| Lista de partidos | 0 |
| Moneyline, spreads y totals (todos los partidos) | 3 |
| Props de **un** partido, catálogo completo | ~17 |
| Props de un partido con `ODDS_PLAYER_MARKETS=core` | ~6 |

- Los props se cachean **30 minutos** en memoria y en disco (`cache/`), así que reiniciar no vuelve a cobrar.
- La sección de combinadas **no descarga props sola**: usa los ya cargados, o el botón "⬇️ Cargar props", que muestra el coste antes.
- Un domingo completo (~13 partidos) con el catálogo completo puede costar ~220 créditos.

---

## Despliegue

La interfaz Streamlit funciona sin la API, así que se puede publicar en **Streamlit Community Cloud** con `app.py` como archivo principal y **Python 3.11**. Las claves van en *Secrets* (formato TOML), no en el repositorio:

```toml
GEMINI_API_KEY = "..."
ODDS_API_KEY = "..."
APP_PASSWORD = "una-contraseña-para-compartir"   # activa la pantalla de acceso
```

Lo que vive en FastAPI (la descarga diaria de las 8 AM, el chat `/chat` y los demás endpoints) necesita un servicio que mantenga procesos corriendo, como Render o Railway, usando `run.py`.

⚠️ Si la app es pública, cualquier visitante puede gastar tus créditos de The-Odds-API. Define `APP_PASSWORD` para que pida contraseña al entrar; sin esa variable la app abre directamente, como en local.

---

## Fuentes de datos

- **nflverse**: estadísticas semanales por jugador y calendario con líneas de Las Vegas. Usa `nfl_data_py` y, para temporadas que este ya no publica, los archivos parquet de `nflverse-data`.
- **The-Odds-API v4**: cuotas de casas de EE. UU. (DraftKings, FanDuel, BetMGM, etc.). Draftea no está incluida: la "línea fija" es el consenso de esas casas.
- **ESPN**: logos de los equipos.

## Aviso

Herramienta de análisis con fines educativos. Las probabilidades son estimaciones de un modelo estadístico, no garantías: el modelo puede sobreestimar el valor en líneas extremas o con pocos partidos de historial. Apuesta con responsabilidad y solo lo que puedas permitirte perder. Verifica que las apuestas deportivas sean legales en tu jurisdicción.
