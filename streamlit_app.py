import streamlit as st
import pandas as pd
import numpy as np
import random
import io

st.set_page_config(page_title="Asta Fantacalcio – Sim", page_icon="🧮", layout="wide")

# =========================
# Utility
# =========================
ROLE_ALIASES = {
    "POR":"P","PORTIERE":"P","GK":"P","P":"P",
    "DIF":"D","DIFENSORE":"D","DEF":"D","D":"D",
    "CEN":"C","CENTROCAMPISTA":"C","MID":"C","M":"C","C":"C",
    "ATT":"A","ATTACCANTE":"A","FWD":"A","A":"A"
}
ROLE_ORDER = ["P","D","C","A"]  # per visualizzare

ROLE_EV_MULT = {"A": 11.0, "C": 8.5, "D": 6.0, "P": 5.0}  # per derivare expected_value se manca

DEFAULT_QUOTA = {"P":3, "D":8, "C":8, "A":6}  # classico 25

def normalize_role(x):
    if pd.isna(x): return None
    v = str(x).strip().upper()
    if v in ROLE_ALIASES: return ROLE_ALIASES[v]
    if v and v[0] in ROLE_ORDER: return v[0]
    return None

def derive_expected_value(bp, role):
    if pd.isna(bp): return np.nan
    m = ROLE_EV_MULT.get(role, 8.0)
    val = float(bp) * m
    return min(int(round(val)), 400)

def first(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# =========================
# Loader flessibile CSV/Excel
# =========================
@st.cache_data(show_spinner=False)
def load_players(uploaded_file=None, use_demo=False):
    if use_demo:
        demo = pd.DataFrame([
            # name, team, position, base_price, expected_value (alcuni senza EV per test)
            ["Martinez L.","Inter","A",34,374],
            ["Vlahovic","Juventus","A",24,np.nan],
            ["Zaccagni","Lazio","A",26,286],
            ["Immobile","Lazio","A",19,np.nan],
            ["Lookman","Atalanta","A",29,319],
            ["Gudmundsson A.","Genoa","A",24,258],
            ["Dimarco","Inter","D",19,114],
            ["Bremer","Juventus","D",13,np.nan],
            ["Bastoni","Inter","D",16,96],
            ["Di Lorenzo","Napoli","D",14,np.nan],
            ["Calhanoglu","Inter","C",23,196],
            ["Barella","Inter","C",14,119],
            ["Koopmeiners","Atalanta","C",14,np.nan],
            ["Frattesi","Inter","C",14,119],
            ["Locatelli","Juventus","C",12,102],
            ["Maignan","Milan","P",16,80],
            ["Sommer","Inter","P",16,np.nan],
            ["Provedel","Lazio","P",12,60],
            ["Svilar","Roma","P",15,75],
        ], columns=["name","team","position","base_price","expected_value"])
        # Normalizza ruoli ed EV mancanti
        demo["position"] = demo["position"].map(normalize_role)
        demo["expected_value"] = demo.apply(
            lambda r: derive_expected_value(r["base_price"], r["position"]) if pd.isna(r["expected_value"]) else r["expected_value"], axis=1
        ).astype(int)
        return demo

    if uploaded_file is None:
        raise ValueError("Nessun file caricato")

    raw = uploaded_file.read()
    uploaded_file.seek(0)
    buf = io.BytesIO(raw)

    # Prova Excel, poi CSV ,; come separatore
    if uploaded_file.name.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(buf)
    else:
        try:
            df = pd.read_csv(buf)
        except Exception:
            buf.seek(0)
            df = pd.read_csv(buf, sep=";")

    df.columns = [str(c).strip().lower() for c in df.columns]

    name_col = first(df, ["name","nome","giocatore","player"]) or df.columns[0]
    team_col = first(df, ["team","squadra","club"])
    role_col = first(df, ["position","ruolo","pos","role"])
    price_col = first(df, ["base_price","prezzo","prezzo base","quotazione","quota","qu","price"])
    ev_col = first(df, ["expected_value","ev","fantavalore","fvm","expected value"])

    out = pd.DataFrame()
    out["name"] = df[name_col].astype(str).str.strip()
    out["team"] = df[team_col].astype(str).str.strip() if team_col else ""
    out["position"] = df[role_col].map(normalize_role) if role_col else None

    # base_price numerico
    if price_col:
        prices = df[price_col].astype(str).str.replace(",", ".", regex=False)
        out["base_price"] = pd.to_numeric(prices, errors="coerce")
    else:
        out["base_price"] = np.nan

    # expected_value
    if ev_col:
        ev = df[ev_col].astype(str).str.replace(",", ".", regex=False)
        out["expected_value"] = pd.to_numeric(ev, errors="coerce")
    else:
        out["expected_value"] = np.nan

    # Completa i mancanti
    out["position"] = out["position"].apply(lambda r: r if r in ROLE_ORDER else None)
    # Fallback ruolo grezzo dal prezzo per sicurezza
    def infer_role(row):
        if row["position"] in ROLE_ORDER:
            return row["position"]
        bp = row["base_price"]
        if pd.isna(bp): return "D"
        if bp >= 25: return "A"
        if bp >= 15: return "C"
        if bp >= 8:  return "D"
        return "P"
    out["position"] = out.apply(infer_role, axis=1)

    out["base_price"] = out["base_price"].fillna(1).clip(lower=1).astype(int)
    out["expected_value"] = out.apply(
        lambda r: derive_expected_value(r["base_price"], r["position"]) if pd.isna(r["expected_value"]) else r["expected_value"], axis=1
    ).astype(int)

    out = out.drop_duplicates(subset=["name","team"]).reset_index(drop=True)
    return out[["name","team","position","base_price","expected_value"]]

# =========================
# Bot & Asta
# =========================
def make_bots(n_bots, budget, roster_quota, styles):
    bots = []
    for i in range(n_bots):
        style = random.choice(styles)
        bots.append({
            "name": f"Bot {i+1} ({style})",
            "style": style,
            "budget": int(budget),
            "roster": {r: [] for r in ROLE_ORDER},
            "quota_left": roster_quota.copy()
        })
    return bots

def style_factor(style):
    # quanto aggressivo è il tetto massimo vs expected_value
    return {
        "Aggressive": 1.15,
        "Balanced":   1.00,
        "Saver":      0.85,
        "Sniper":     1.05
    }.get(style, 1.0)

def need_bonus(bot, role):
    # se ha pochi slot rimasti su quel ruolo, bidare un filo di più
    left = bot["quota_left"].get(role, 0)
    if left <= 1: return 1.10
    if left <= 2: return 1.05
    return 1.00

def reservation_price(player, bot, market_tightness=1.00):
    ev = player["expected_value"]
    role = player["position"]
    base = player["base_price"]
    cap_by_budget = bot["budget"]
    # tetto teorico
    max_wtp = ev * style_factor(bot["style"]) * need_bonus(bot, role) * market_tightness
    # non scendiamo mai sotto base_price * 0.8 per realism, e mai sopra budget
    rp = int(max(base, round(max_wtp * 0.12)))  # 0.12 ~ scala EV→€
    # piccolo rumor per non essere tutti identici
    rp = int(max(base, rp + random.randint(-2, 3)))
    return min(rp, cap_by_budget)

def can_bid_on(player, bot):
    role = player["position"]
    return bot["quota_left"].get(role,0) > 0 and bot["budget"] > 0

def run_auction(players, bots, price_increment=1, order="EV (alto→basso)", seed=None, market_tightness=1.0):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    if order.startswith("EV"):
        pool = players.sort_values("expected_value", ascending=False).reset_index(drop=True)
    elif order.startswith("Prezzo"):
        pool = players.sort_values("base_price", ascending=False).reset_index(drop=True)
    else:
        pool = players.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    log = []
    for _, p in pool.iterrows():
        # chi può partecipare?
        eligible = [b for b in bots if can_bid_on(p, b)]
        if not eligible:
            continue

        # calcolo riserve
        reserves = {b["name"]: reservation_price(p, b, market_tightness) for b in eligible}
        current = max(1, int(p["base_price"]))
        bidder_cycle = eligible.copy()
        last_bidder = None

        # almeno uno è disposto a partire?
        if all(current >= reserves[b["name"]] for b in eligible):
            # nessuno interessato al prezzo base
            continue

        # loop d'asta
        rounds = 0
        while True:
            rounds += 1
            if rounds > 500:  # safety
                break

            someone_bid = False
            for b in list(bidder_cycle):
                name = b["name"]
                if b["budget"] < current + price_increment:
                    continue
                if current + price_increment <= reserves[name]:
                    current += price_increment
                    last_bidder = b
                    someone_bid = True
                # se nessuno rilancia nel giro completo => assegna
            if not someone_bid:
                break

        # assegna al last_bidder se c'è
        if last_bidder is None:
            continue

        winner = last_bidder
        pay = current
        role = p["position"]

        # aggiorna winner
        winner["budget"] -= pay
        winner["roster"][role].append(p["name"])
        winner["quota_left"][role] = max(0, winner["quota_left"][role]-1)

        log.append({
            "player": p["name"],
            "team": p["team"],
            "position": role,
            "base_price": int(p["base_price"]),
            "expected_value": int(p["expected_value"]),
            "winner": winner["name"],
            "paid": int(pay)
        })

    log_df = pd.DataFrame(log)
    return log_df, bots

# =========================
# UI
# =========================
st.title("🧮 Simulatore Asta Fantacalcio (rifatto)")

colL, colR = st.columns([2,1])
with colL:
    uploaded = st.file_uploader("Carica giocatori (CSV o XLSX)", type=["csv","xlsx","xls"])
    use_demo = st.checkbox("Usa dati demo", value=uploaded is None)
with colR:
    st.markdown("**Formato minimo richiesto:**")
    st.code("name, team, position, base_price, expected_value", language="text")

# Parametri asta
with st.sidebar:
    st.header("⚙️ Impostazioni Asta")
    n_bots = st.slider("Numero bot", 2, 12, 6)
    budget = st.number_input("Budget per bot", 100, 2000, 500, step=10)

    st.subheader("Quote Rosa (per bot)")
    colq1, colq2 = st.columns(2)
    with colq1:
        qP = st.number_input("Portieri (P)", 1, 5, DEFAULT_QUOTA["P"])
        qD = st.number_input("Difensori (D)", 4, 12, DEFAULT_QUOTA["D"])
    with colq2:
        qC = st.number_input("Centrocampisti (C)", 4, 12, DEFAULT_QUOTA["C"])
        qA = st.number_input("Attaccanti (A)", 2, 9, DEFAULT_QUOTA["A"])
    roster_quota = {"P":int(qP),"D":int(qD),"C":int(qC),"A":int(qA)}

    price_inc = st.number_input("Incremento minimo", 1, 10, 1, step=1)
    order = st.selectbox("Ordine chiamata", ["EV (alto→basso)", "Prezzo base (alto→basso)", "Casuale"])
    styles = st.multiselect("Stili bot", ["Aggressive","Balanced","Saver","Sniper"], default=["Aggressive","Balanced","Saver","Sniper"])
    market_tightness = st.slider("Tensione mercato (0.8 = soft, 1.2 = hot)", 0.8, 1.2, 1.0, 0.01)
    seed = st.number_input("Seed casuale (0 = random)", 0, 999999, 0, step=1)
    start_btn = st.button("🚀 Avvia simulazione")

# Caricamento dati
try:
    players = load_players(uploaded, use_demo=use_demo)
    st.success(f"Giocatori caricati: {len(players)}")
    with st.expander("Anteprima giocatori"):
        st.dataframe(players.head(30), use_container_width=True)
except Exception as e:
    st.error("Errore nel caricamento. Verifica il file e riprova.")
    st.exception(e)
    st.stop()

# Avvio simulazione
if start_btn:
    if not styles:
        st.error("Seleziona almeno uno stile bot.")
        st.stop()

    bots = make_bots(n_bots, budget, roster_quota, styles)
    log_df, bots_out = run_auction(
        players=players,
        bots=bots,
        price_increment=int(price_inc),
        order=order,
        seed=None if seed==0 else int(seed),
        market_tightness=float(market_tightness)
    )

    st.subheader("📜 Log Asta")
    if log_df.empty:
        st.warning("Nessun acquisto effettuato (budget troppo basso o quote piene). Prova a ridurre le quote o aumentare il budget.")
    else:
        st.dataframe(log_df, use_container_width=True, height=360)

        # KPI
        st.subheader("📊 Riepilogo Bot")
        rows = []
        for b in bots_out:
            total_players = sum(len(b["roster"][r]) for r in ROLE_ORDER)
            spent = sum(
                log_df.loc[log_df["winner"]==b["name"], "paid"].astype(int)
            ) if not log_df.empty else 0
            rows.append({
                "bot": b["name"],
                "stile": b["style"],
                "acquisti": total_players,
                "speso": int(spent),
                "budget_residuo": int(b["budget"]),
                **{f"{r}_presi": len(b["roster"][r]) for r in ROLE_ORDER}
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Rose per bot
        st.subheader("🧾 Rose finali")
        tabs = st.tabs([b["name"] for b in bots_out])
        for tab, b in zip(tabs, bots_out):
            with tab:
                parts = []
                for r in ROLE_ORDER:
                    names = b["roster"][r]
                    if names:
                        parts.append(f"**{r}**: " + ", ".join(names))
                if parts:
                    st.markdown("  \n".join(parts))
                else:
                    st.write("Nessun giocatore acquistato.")

        # Download
        st.subheader("⬇️ Esporta")
        st.download_button(
            "Scarica LOG asta (CSV)",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="auction_log.csv",
            mime="text/csv"
        )
        # Roster flattened
        roster_rows = []
        for b in bots_out:
            for r in ROLE_ORDER:
                for n in b["roster"][r]:
                    roster_rows.append({"bot": b["name"], "role": r, "player": n})
        roster_df = pd.DataFrame(roster_rows)
        st.download_button(
            "Scarica ROSE (CSV)",
            data=roster_df.to_csv(index=False).encode("utf-8"),
            file_name="rosters.csv",
            mime="text/csv"
        )

else:
    st.info("Imposta i parametri nella sidebar e premi **Avvia simulazione**.")
