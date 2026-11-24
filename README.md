# 🏈 NFL +EV Betting Terminal

Terminal de analítica para apuestas de la NFL que busca **valor esperado positivo (+EV)**. Compara las cuotas reales de las casas (The-Odds-API) con probabilidades propias calculadas por **simulación Monte Carlo (10,000 iteraciones)** sobre el historial de cada jugador y equipo (nflverse).

- Interfaz **Streamlit** de pantalla única, en modo oscuro.
- API **FastAPI** con chat de **Gemini** (function calling).
- Cuotas actualizadas ~1 h antes de cada jornada con GitHub Actions, y **seguimiento de resultados reales** del modelo.

> Todas las cuotas se muestran en **formato decimal** (ej. `1.91`).

---

## Funcionalidades

**Partidos**
- Tarjetas por partido con logos, marcador proyectado por simulación y cuotas de moneyline, spread y total.
- Un badge `🟢 +EV` aparece cuando la probabilidad ajustada da un EV de al menos 3%.

**Analítica de jugadores** (al tocar un partido)
- Catálogo de jugadores con props y vista enfocada en uno solo, con botón para volver al resto.
- Categorías ⚽ Touchdowns · 🏈 Pases · 🏃 Acarreos · 🎯 Recepciones · 🔀 Todos. Cubre todos los mercados de jugador de The-Odds-API: yardas, TDs, intentos, completos, intercepciones, recepciones, Anytime TD y Primer TD, con sus líneas alternas.
- Métricas de los últimos 10 partidos (L10) y gráfico de *hit-rate* contra la línea activa.
- Tabla de escaleras alternas con la probabilidad ajustada, el EV y la *Fair Odds* de cada línea.
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
- **Probabilidad calibrada con resultados reales** (`model_calibration.json`, ver *Backtest*). Por cada mercado, una regresión logística ajustada con cuotas históricas decide cuánto pesa el Monte Carlo frente al mercado sin margen y corrige su sesgo:
  - *Modelo + mercado*: el Monte Carlo sí aportó información (ej. recepciones).
  - *Mercado + corrección de sesgo*: el modelo no aportó, pero el mercado tiene un sesgo medible (ej. los Overs de yardas de carrera se cumplen menos de lo que pagan).
  - *Solo mercado*: el modelo empeoraba la predicción.
  - Mercados sin datos históricos usan la mezcla fija de respaldo: 50% modelo en props, 30% en equipos, con penalización si discrepa mucho del mercado.
- **+EV solo en mercados validados**: cuenta como `🟢 +EV` un EV entre el umbral y el tope (elegidos por el backtest), y solo en mercados que ganaron dinero fuera de muestra. En el resto, un EV alto se marca `🟡 sin ventaja comprobada`, y por encima del tope `⚠️ Revisar`.
- **Combinadas**: correlación entre patas calibrada con 33,000 parejas reales del mismo partido. Las patas salen solo de mercados validados.
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
| `scheduler.py` | Lectura del snapshot de cuotas; descarga local opcional a las 8 AM (`LOCAL_SCHEDULER=1`) |
| `refresh_snapshot.py` | Descarga de cuotas antes de cada jornada (GitHub Actions) |
| `tracking.py` | Registro, CLV y calificación de las selecciones del modelo (`tracking/picks.csv`) |
| `backtest.py` | Descarga de cuotas históricas con tope de créditos (caché en `backtest/raw/`) |
| `backtest_eval.py` | Backtest punto-en-el-tiempo, calibración y validación fuera de muestra |
| `model_calibration.json` | Parámetros calibrados que usa la app (lo genera `backtest.py fit`) |
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
# LOCAL_SCHEDULER=1                     # opcional: descarga local diaria (redundante con GitHub Actions)
MORNING_REFRESH_HOUR=8                  # hora de esa descarga local
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
| GET | `/health` | Estado, semana activa y snapshot de cuotas vigente |
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

## Actualización automática (GitHub Actions)

El workflow `.github/workflows/odds-refresh.yml` actualiza las cuotas **~1 hora antes de cada jornada**, es decir, de cada bloque de kickoffs del día (ej. domingo 13:00 · 16:05/16:25 · 20:20 ET):

1. Corre cada 15 min en horario de partidos y lee el calendario de nflverse (gratis).
2. Si una jornada empieza en ~1 h y aún no se actualizó, ejecuta `refresh_snapshot.py`. Este descarga las líneas de partido y los props de todos los partidos de hoy que aún no empezaron.
3. Guarda el resultado en `snapshots/` y hace commit. Streamlit Cloud se actualiza sola y los visitantes no gastan créditos.

**Configuración:** GitHub → *Settings → Secrets and variables → Actions → New repository secret* → `ODDS_API_KEY`.
**Manual:** pestaña *Actions → Actualizar cuotas antes de cada jornada → Run workflow* (casilla *force* para descargar ya).
**Local:** `python refresh_snapshot.py --dry-run` muestra las jornadas de hoy y lo que haría.

Coste aproximado con el catálogo completo: domingo ~320 créditos (3 jornadas), jueves y lunes ~20 cada uno, ~1,500–2,000 al mes.

### Seguimiento de resultados

Para saber si el +EV se convierte en ganancias reales:

1. En cada jornada, el mismo workflow guarda en `tracking/picks.csv` lo que el modelo recomienda, con la cuota de ese momento:
   - Las sencillas +EV (máx. 12, solo líneas principales, cuota ≤ 4.00).
   - La mejor combinada de cada nivel de riesgo.
2. `.github/workflows/results-grade.yml` corre cada día a las 05:00 y 11:00 ET. Califica las pendientes con el marcador final y las estadísticas de nflverse:
   - Ganada / perdida / anulada. Push o jugador que no jugó = anulada.
   - Stake plano de 1 unidad.
3. En la app: **📈 Rendimiento del modelo**, al final de la vista de partidos. Muestra acierto real vs probabilidad esperada, unidades y ROI, por sencillas y por nivel de riesgo.

Con menos de ~100–200 selecciones, el ROI es sobre todo varianza. La señal útil es que el acierto real se parezca a la probabilidad esperada (calibración).

**CLV (*Closing Line Value*)**: ~10 min antes de cada kickoff, el workflow descarga la línea de cierre, pero solo de los mercados de las selecciones registradas (~3 créditos más ~3 por partido). Con ella calcula `CLV % = p_cierre_sin_vig × cuota_tomada − 1`.
- Si la cuota no se mueve, el CLV es ≈ −margen de la casa.
- Un CLV medio que mejora y supera 0% en 50+ selecciones es la señal más rápida de ventaja real, mucho antes que el ROI.

Local: `python tracking.py grade` · `python tracking.py summary`.

## Backtest con cuotas históricas

`backtest.py` usa los endpoints históricos de The-Odds-API (planes de pago):
- Lista de partidos: 1 crédito.
- Líneas de partido: 30 créditos por jornada.
- Props: 10 créditos por mercado y partido.

Las respuestas se guardan en `backtest/raw/` y nunca se vuelven a pagar.

```bash
python backtest.py fetch --budget 8000   # descarga con tope de créditos
python backtest.py run                   # selecciones punto-en-el-tiempo + resultado real
python backtest.py fit                   # calibra, valida y escribe model_calibration.json
```

- **Punto en el tiempo**: cada partido se evalúa con estadísticas de semanas anteriores y la cuota de ~1 h antes del kickoff, igual que en vivo.
- **Validación fuera de muestra**: cada semana se predice con una calibración ajustada sin esa semana. Los mercados "validados" se eligen con la primera mitad de la temporada y se prueban en la segunda.
- Resultados en la app: **📈 Rendimiento del modelo → 🧪 Backtest** (`backtest/report.json`).

**Resultados (temporada 2026, semanas 1–11, 6,924 líneas, 7,952 créditos):**

| | Log-loss fuera de muestra (menor = mejor) |
|---|---|
| Monte Carlo puro | 0.6379 (sobreconfiado: cuando dice 75%, se cumple ~55%) |
| Mercado sin vig | 0.5864 |
| **Probabilidad calibrada (la que usa la app)** | **0.5855** |

- **Sencillas con EV ≥ 2%:** 1,118 apuestas, ROI fuera de muestra **+2.3%**. En la prueba ciega (semanas 6–11, mercados elegidos con 3–5) el ROI fue **+1.2%** (808 apuestas). Antes de calibrar era **−9.2%**.
- **Ventaja comprobada:** yardas de carrera (+4.1%, sobre todo Unders). Los Overs en general perdieron (−3.3%).
- **Combinadas:** la probabilidad conjunta está bien calibrada (Moderado: 31.3% esperado vs 32.0% real). El ROI de 52 combinadas todavía es ruido.
- **Monte Carlo entrenado** (`python model_tuning.py`, ~1 vez al mes): parámetros por mercado (`model_params.json`) entrenados con la temporada anterior (CRPS sobre el resultado real), validados con las líneas reales y adoptados solo si no empeoran la prueba final. Resultado: hasta 17 partidos de historial, distribución 10–20% más ancha y ajuste por defensa rival. Log-loss del Monte Carlo 0.6379 → 0.6230; sencillas EV ≥ 2% ROI +2.0% → **+3.3%** (1,116); prueba ciega +1.2% → **+3.0%** (482).
- Es una sola temporada: los márgenes de error son de varios puntos. El seguimiento en vivo y el CLV confirmarán o descartarán la ventaja.

> Como el bot hace commits, antes de subir cambios propios ejecuta `git pull`.

## Despliegue

La interfaz Streamlit funciona sin la API, así que se puede publicar en **Streamlit Community Cloud** con `app.py` como archivo principal y **Python 3.11**. Las claves van en *Secrets* (formato TOML), no en el repositorio:

```toml
GEMINI_API_KEY = "..."
ODDS_API_KEY = "..."
APP_PASSWORD = "una-contraseña-para-compartir"   # activa la pantalla de acceso
```

Lo que vive en FastAPI (el chat `/chat` y los demás endpoints) necesita un servicio que mantenga procesos corriendo, como Render o Railway, usando `run.py`.

⚠️ Si la app es pública, cualquier visitante puede gastar tus créditos de The-Odds-API. Define `APP_PASSWORD` para que pida contraseña al entrar; sin esa variable la app abre directamente, como en local.

---

## Fuentes de datos

- **nflverse**: estadísticas semanales por jugador y calendario con líneas de Las Vegas. Usa `nfl_data_py` y, para temporadas que este ya no publica, los archivos parquet de `nflverse-data`.
- **The-Odds-API v4**: cuotas de casas de EE. UU. (DraftKings, FanDuel, BetMGM, etc.). Draftea no está incluida: la "línea fija" es el consenso de esas casas.
- **ESPN**: logos de los equipos.

## Aviso

Herramienta de análisis con fines educativos. Las probabilidades son estimaciones de un modelo estadístico, no garantías: el modelo puede sobreestimar el valor en líneas extremas o con pocos partidos de historial. Apuesta con responsabilidad y solo lo que puedas permitirte perder. Verifica que las apuestas deportivas sean legales en tu jurisdicción.
