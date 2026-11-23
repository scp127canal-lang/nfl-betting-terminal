"""Configuración global del proyecto: estilos Dark Mode y utilidades de cuotas."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Variables de entorno ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY", "")
MORNING_REFRESH_HOUR = int(os.getenv("MORNING_REFRESH_HOUR", "8"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
NFL_TIMEZONE = "America/New_York"  # las fechas de partido de la NFL van en hora del Este

# --- Estilo Dark Mode (Streamlit) ---
COLOR_BG = "#0E1117"
COLOR_CARD = "#161B22"
COLOR_NEON_GREEN = "#00FF87"
COLOR_NEON_PINK = "#FF2E93"
COLOR_BORDER = "#21262D"


def american_to_decimal(american_odds) -> float:
    """Convierte una cuota americana a decimal pura (float, 2 decimales).

    Ejemplos: -110 -> 1.91 | +150 -> 2.5 | -175 -> 1.57

    REGLA: devuelve solo un número; nunca 'x', prefijos ni sufijos.
    """
    odds = float(str(american_odds).strip().replace("+", ""))
    if odds == 0 or -100 < odds < 100:
        raise ValueError(f"Cuota americana inválida: {american_odds}")
    if odds > 0:
        decimal = 1 + odds / 100
    else:
        decimal = 1 + 100 / abs(odds)
    return round(decimal, 2)
