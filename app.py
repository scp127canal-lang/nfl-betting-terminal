"""NFL +EV Terminal — interfaz Streamlit de pantalla única.

Vistas (st.session_state.current_view):
  - "games":         tarjetas de partidos + combinadas sugeridas por riesgo.
  - "player_detail": sección dedicada de jugadores del partido elegido.
La boleta de apuestas vive en la barra lateral en ambas vistas.
"""

from __future__ import annotations

import os

import streamlit as st

import ui_components as ui

st.set_page_config(page_title="NFL +EV Terminal", page_icon="🏈", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()
ui.require_password()   # solo si APP_PASSWORD está en Secrets / .env
# Boleta: ancho mínimo garantizado desde el primer render (solo cuando está abierta,
# para que el botón de colapsar siga funcionando).
st.markdown("""<style>
section[data-testid="stSidebar"][data-collapsed="false"],
section[data-testid="stSidebar"][aria-expanded="true"] {
  min-width: 350px !important; width: 360px !important; }
</style>""", unsafe_allow_html=True)

st.session_state.setdefault("current_view", "games")
st.session_state.setdefault("selected_game", None)
st.session_state.setdefault("selected_player_id", None)

with st.spinner("Cargando datos NFL..."):
    ctx = ui.context()
    weekly = ui.load_weekly(ctx["seasons"])
    sched = ui.load_schedules(ctx["seasons"])

st.session_state.setdefault("week", ctx["week"])
with st.sidebar:
    ui.render_bet_slip(weekly)


def games_view() -> None:
    ui.render_header(ctx, os.getenv("NFL_API_URL"))
    c1, c2, _ = st.columns([1, 1, 3], vertical_alignment="bottom")
    c1.selectbox("Semana", list(range(1, 23)), key="week")
    c2.toggle("Mostrar finalizados", key="show_final")
    week = st.session_state["week"]

    games = ui.week_games(sched, ctx["season"], week, st.session_state.get("show_final", False))
    if games.empty:
        st.info(f"No hay partidos pendientes en la semana {week}. Activa 'Mostrar finalizados' "
                "o elige otra semana.")
        return

    snap = ui.snapshot_key()
    with st.spinner("Cargando cuotas y simulando partidos..."):
        board, event_map = ui.game_market_board(games, snap)
        legs = ui.team_legs(board, sched, f"{ctx['season']}-{week}-{snap}-{len(games)}")
        proj = ui.projections(games, sched)
    sources = board.groupby("event_id")["source"].first().to_dict() if not board.empty else {}

    for start in range(0, len(games), 2):
        cols = st.columns(2)
        for col, game in zip(cols, games.iloc[start:start + 2].itertuples(index=False)):
            with col:
                ui.render_game_card(game, proj.get(game.game_id), ui.best_market_legs(legs, game.game_id),
                                    sources.get(game.game_id, "Sin cuotas"))

    st.divider()
    mode = ui.render_parlay_mode_selector()
    pending_ids = set(games.loc[games["home_score"].isna(), "game_id"])
    team_pool = [leg for leg in legs if leg.event_id in pending_ids]
    pending_map = {g: e for g, e in event_map.items() if g in pending_ids}
    fetch = st.session_state.pop("_fetch_slate_props", False)
    with st.spinner("Cargando props de jugadores de los partidos pendientes..."):
        props, missing = ui.slate_props_live(pending_map, fetch_missing=fetch)
    if missing and ui.ODDS_API_ENABLED and mode != "teams":
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.caption(f"{len(missing)} partido(s) pendiente(s) sin props cargados. Abre un partido para cargar "
                   f"los suyos, o descárgalos todos (~17 créditos por partido"
                   + (f" · {ui.quota_text()}" if ui.quota_text() else "") + ").")
        c2.button(f"⬇️ Cargar props de {len(missing)} partido(s)", key="fetch_slate",
                  on_click=lambda: st.session_state.update(_fetch_slate_props=True))
    player_pool = []
    if not props.empty:
        with st.spinner("Simulando props de jugadores..."):
            player_pool = ui.player_legs_cached(props, weekly, f"main-{week}-{snap}-{len(props)}")
    pool = team_pool + player_pool
    note = ui.pool_note(player_pool, team_pool, props)
    if mode != "teams" and not player_pool:
        st.info("No hay props de jugadores cargados para los partidos pendientes: ábrelos o usa el botón "
                "de descarga (los props suelen publicarse 1-3 días antes del partido). "
                "Mientras tanto, usa 🏟️ Solo Equipos.")
        return
    if len(pool) < 2:
        st.info("No hay suficientes selecciones pendientes para generar combinadas.")
        return
    with st.spinner("Generando combinadas +EV (Monte Carlo + cópula)..."):
        tiers = ui.slate_parlays(pool, weekly, f"{ctx['season']}-{week}-{snap}-{len(pool)}", mode)
    ui.render_parlay_module(tiers, {leg.label: leg for leg in pool},
                            st.session_state["bankroll"], note)


def player_detail_view() -> None:
    gid = st.session_state.selected_game
    match = sched[sched["game_id"] == gid]
    st.button("⬅️ Volver a Lista de Partidos", key="back", on_click=ui.back_to_games)
    if match.empty:
        st.error("Partido no encontrado.")
        return
    game = match.iloc[0]
    home, away = game["home_team"], game["away_team"]

    games = match.reset_index(drop=True)
    snap = ui.snapshot_key()
    board, event_map = ui.game_market_board(games, snap)
    proj = ui.projections(games, sched).get(gid)
    source = board["source"].iloc[0] if not board.empty else "Sin cuotas"

    with st.spinner("Cargando props del partido..."):
        props = ui.game_props(gid, event_map.get(gid))
        players = ui.game_players(props, weekly, int(game["season"]), home, away)
    with st.spinner("Calculando +EV de todos los props..."):
        scan = ui.scan_game_props(props, weekly, f"{gid}-{snap}-{len(props)}")
    best_ev = ui.best_ev_by_player(scan)

    ui.render_game_header(game, proj, source, len(players), not props.empty)
    if props.empty:
        st.warning("Sin props de The-Odds-API para este partido: se muestran los jugadores con más uso "
                   "reciente y líneas de referencia (introduce la cuota manualmente).")
    elif not scan.empty:
        n_plus = int((scan["ev_pct"] > 0).sum())
        st.caption(f"Escáner +EV: {n_plus} de {int(scan['ev_pct'].notna().sum())} líneas con valor "
                   f"(modelo Monte Carlo > probabilidad implícita).")

    players = players.assign(player_id=[ui.player_id(k, t) for k, t in
                                        zip(players["name_key"], players["team"])])
    pid = st.session_state.get("selected_player_id")
    focus = players[players["player_id"] == pid] if pid else players.iloc[0:0]

    # --- Focus View: solo la ficha del jugador elegido ---
    if not focus.empty:
        ui.render_player_back_button(len(players))
        ui.render_player_terminal(focus.iloc[0], props, weekly, gid)
        return
    if pid:  # el jugador ya no está en este partido (cambio de cuotas/plantilla)
        st.session_state.selected_player_id = None

    # --- Catálogo completo de jugadores del partido ---
    c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
    query = c1.text_input("🔎 Buscar jugador", key=f"q_{gid}", placeholder="Nombre del jugador...")
    pos = c2.pills("Posición", ["Todos", *ui.POSITIONS], key=f"pos_{gid}", default="Todos")
    pos = pos or "Todos"

    shown = players
    if query:
        shown = shown[shown["player"].str.contains(query, case=False, regex=False)]
    if pos != "Todos":
        shown = shown[shown["position"] == pos]
    if shown.empty:
        st.info("Ningún jugador coincide con el filtro.")
        return
    st.markdown('<div class="pulse">👇 Toca "📊 Analizar" para abrir la ficha completa del jugador</div>',
                unsafe_allow_html=True)
    ui.render_player_cards(shown, best_ev)


if st.session_state.current_view == "player_detail" and st.session_state.selected_game:
    player_detail_view()
else:
    games_view()
