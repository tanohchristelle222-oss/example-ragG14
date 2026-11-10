"""Génère benchmark_llm.pdf, un rapport court, à partir de benchmark/results/.

Usage, depuis la racine du repo :  python benchmark/make_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from questions import QUESTIONS

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmark" / "results"
PDF_PATH = ROOT / "benchmark_llm.pdf"

TEAM = "Équipe 14"
QUALITY_WEIGHT = 0.6
SPEED_WEIGHT = 0.4
MIN_RULE_RATE = 0.75
MONTHS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
HEADER_BG = colors.HexColor("#f1f0ec")
LINE = colors.HexColor("#d5d4cf")
HIGHLIGHT = colors.HexColor("#e6f0fb")


def register_fonts() -> tuple[str, str]:
    """Arial sous Windows, sinon DejaVu Sans fournie avec matplotlib."""
    windows = Path("C:/Windows/Fonts")
    regular, bold = windows / "arial.ttf", windows / "arialbd.ttf"
    if not (regular.exists() and bold.exists()):
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        regular, bold = ttf / "DejaVuSans.ttf", ttf / "DejaVuSans-Bold.ttf"
    pdfmetrics.registerFont(TTFont("Corps", str(regular)))
    pdfmetrics.registerFont(TTFont("Corps-Gras", str(bold)))
    pdfmetrics.registerFontFamily("Corps", normal="Corps", bold="Corps-Gras",
                                  italic="Corps", boldItalic="Corps-Gras")
    return "Corps", "Corps-Gras"


FONT, FONT_BOLD = register_fonts()
BODY = ParagraphStyle("body", fontName=FONT, fontSize=10.5, leading=15, spaceAfter=7, textColor=colors.HexColor(INK))
H1 = ParagraphStyle("h1", parent=BODY, fontName=FONT_BOLD, fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=BODY, fontName=FONT_BOLD, fontSize=10.5, leading=14, spaceBefore=4, spaceAfter=3)
TITLE = ParagraphStyle("title", parent=BODY, fontName=FONT_BOLD, fontSize=19, leading=24, alignment=TA_CENTER, spaceAfter=6)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, alignment=TA_CENTER, textColor=colors.HexColor(INK_SECONDARY), spaceAfter=12)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=9.5, leading=12, spaceAfter=0)
CELL_HEAD = ParagraphStyle("cellhead", parent=CELL, fontName=FONT_BOLD)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(item, BULLET, bulletText="•") for item in items]


def table(header: list[str], rows: list[list[str]], widths_cm: list[float], highlight_row: int | None) -> Table:
    data = [[p(h, CELL_HEAD) for h in header]] + [[p(str(c), CELL) for c in row] for row in rows]
    result = Table(data, colWidths=[w * cm for w in widths_cm], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row + 1), (-1, highlight_row + 1), HIGHLIGHT))
    result.setStyle(TableStyle(style))
    return result


def footer(canvas, doc) -> None:
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor(INK_SECONDARY))
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")


# --- Données -----------------------------------------------------------------


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        sys.exit(f"{path} est introuvable. Lancez d'abord : python benchmark/run_benchmark.py")
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def read_manual_scores() -> pd.Series:
    """Note moyenne par modèle. Tolère un CSV réenregistré par Excel."""
    path = RESULTS_DIR / "manual_scores.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            scores = pd.read_csv(path, sep=";", encoding=encoding, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    else:
        return pd.Series(dtype=float)
    scores["note"] = pd.to_numeric(scores["score_sur_5"].str.replace(",", "."), errors="coerce")
    scores = scores[scores["note"].between(0, 5)]
    return scores.groupby("model")["note"].mean()


def summarise_llm(llm: pd.DataFrame, models: pd.DataFrame, manual: pd.Series) -> pd.DataFrame:
    rows = []
    for _, info in models.iterrows():
        model_rows = llm[llm["model"] == info["model"]]
        ok = model_rows[model_rows["statut"] == "ok"]
        if ok.empty or len(ok) < len(model_rows):
            continue  # Un modèle incomplet n'est pas comparable aux autres.
        row = {
            "model": info["model"],
            "ram_mo": info.get("ram_mo"),
            "note": manual.get(info["model"]),
            "temps_s": ok["total_duration_s"].mean(),
            "cite": as_bool(ok["cite_arome"]).mean(),
            "invente": as_bool(ok["invente_arome"]).mean(),
            "francais": as_bool(ok["en_francais"]).mean(),
            "question": as_bool(ok["termine_par_question"]).mean(),
        }
        row["consignes"] = (row["cite"] + (1 - row["invente"]) + row["francais"] + row["question"]) / 4
        row["qualite"] = row["consignes"] if pd.isna(row["note"]) else (row["consignes"] + row["note"] / 5) / 2
        rows.append(row)
    if not rows:
        sys.exit("Aucun LLM n'a répondu aux 8 questions : rapport impossible.")
    return pd.DataFrame(rows)


def max_precision() -> float:
    """Precision@3 maximale possible : certaines questions ont moins de 3 arômes attendus."""
    return sum(min(len(q["attendus"]), 3) / 3 for q in QUESTIONS) / len(QUESTIONS)


def score_pairs(summary: pd.DataFrame, embed: pd.DataFrame) -> pd.DataFrame:
    pmax = max_precision()
    rows = []
    for _, llm in summary.iterrows():
        for _, emb in embed[embed["statut"] == "ok"].iterrows():
            embed_quality = (emb["hit_at_3"] + min(emb["precision_at_3"] / pmax, 1)) / 2
            rows.append({
                "llm": llm["model"],
                "embedding": emb["model"],
                "qualite": (llm["qualite"] + embed_quality) / 2,
                "temps_s": llm["temps_s"] + emb["temps_moyen_requete_s"],
            })
    pairs = pd.DataFrame(rows)
    pairs["vitesse"] = pairs["temps_s"].min() / pairs["temps_s"]
    pairs["score"] = QUALITY_WEIGHT * pairs["qualite"] + SPEED_WEIGHT * pairs["vitesse"]
    return pairs.sort_values("score", ascending=False, ignore_index=True)


def missed_rules(row: pd.Series) -> list[str]:
    rules = {"citer un arôme du catalogue": row["cite"], "répondre en français": row["francais"],
             "finir par une question": row["question"], "ne pas inventer d'arôme": 1 - row["invente"]}
    return [label for label, rate in rules.items() if rate < MIN_RULE_RATE]


# --- Formats et graphiques ---------------------------------------------------


def fmt(value, pattern: str = "{:.1f}", suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return pattern.format(value).replace(".", ",") + suffix


def pct(value) -> str:
    return fmt(100 * value, "{:.0f}", " %")


def french_date(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


# --- Construction du PDF -----------------------------------------------------


def choose_pair(pairs: pd.DataFrame, summary: pd.DataFrame, used_embed: str) -> tuple[pd.Series, bool]:
    """Meilleur score parmi les modèles qui respectent les consignes, testés de bout en bout."""
    compliant = [r["model"] for _, r in summary.iterrows() if not missed_rules(r)]
    candidates = pairs[pairs["llm"].isin(compliant) & (pairs["embedding"] == used_embed)]
    if candidates.empty:
        return pairs.iloc[0], False
    return candidates.iloc[0], True


def build() -> None:
    embed = read_csv("embed_results.csv")
    run_info = json.loads((RESULTS_DIR / "run_info.json").read_text(encoding="utf-8"))
    manual = read_manual_scores()
    summary = summarise_llm(read_csv("llm_results.csv"), read_csv("llm_models.csv"), manual)
    pairs = score_pairs(summary, embed)
    used_embed = run_info["embedding_retenu"]
    machine = run_info["machine"]
    has_manual = not manual.empty

    choice, compliant = choose_pair(pairs, summary, used_embed)
    llm = summary[summary["model"] == choice["llm"]].iloc[0]
    emb = embed[embed["model"] == used_embed].iloc[0]
    others = embed[(embed["statut"] == "ok") & (embed["model"] != used_embed)]
    summary_view = summary.sort_values("temps_s", ignore_index=True)
    fastest = summary_view.iloc[0]
    chosen_row = int(summary_view.index[summary_view["model"] == choice["llm"]][0])

    story: list = [
        p("Benchmark des modèles Ollama — Plateforme de commande intelligente Givaudan", TITLE),
        p(f"{TEAM} · {french_date(date.today())}", SUBTITLE),

        p("1. Contexte", H1),
        p("Un client, souvent novice, décrit en français l'arôme qu'il cherche, par exemple "
          "« un arôme fruité pour un yaourt à boire ». Notre plateforme retrouve les arômes adaptés dans le "
          "catalogue, puis un modèle de langage lui répond comme un conseiller. "
          "Tout doit fonctionner en local, sans service externe."),
        p("Il nous faut donc choisir deux modèles : un pour <b>chercher</b> les bons arômes, "
          "et un pour <b>répondre</b> au client."),

        p("2. Méthodologie", H1),
        *bullets([
            f"Nous avons posé les mêmes {len(QUESTIONS)} demandes clients à chaque modèle, "
            "sur un catalogue fictif de 48 arômes.",
            f"Tout a tourné sur notre ordinateur portable ({fmt(machine.get('ram_go'))} Go de mémoire, sans carte graphique dédiée).",
            "<b>Recherche</b> : le bon arôme se trouve-t-il dans les 3 premiers résultats ?",
            "<b>Réponse</b> : les 4 modèles reçoivent les mêmes arômes. Nous mesurons le temps de réponse et le "
            "respect de 4 consignes : citer un arôme du catalogue, ne pas en inventer, répondre en français, "
            "finir par une question.",
        ]),

        p("3. Résultats", H1),
    ]

    other_text = "".join(
        f" {r['model']} fait moins bien, avec {pct(r['precision_at_3'])} de bonnes propositions."
        for _, r in others.iterrows() if r["precision_at_3"] < emb["precision_at_3"]
    )
    embed_ok = embed[embed["statut"] == "ok"].reset_index(drop=True)
    embed_rows = [[r["model"], pct(r["hit_at_3"]), pct(r["precision_at_3"])] for _, r in embed_ok.iterrows()]
    embed_chosen = int(embed_ok.index[embed_ok["model"] == used_embed][0])
    story += [
        KeepTogether([
            p("Chercher les bons arômes", H2),
            table(["Modèle", "Trouve au moins un bon arôme", "Bonnes propositions"], embed_rows, [6, 5.5, 5.5],
                  highlight_row=embed_chosen),
            Spacer(1, 6),
            p(f"<b>{used_embed}</b> trouve au moins un bon arôme pour {pct(emb['hit_at_3'])} des demandes, et "
              f"{pct(emb['precision_at_3'])} de ses propositions sont justes (le maximum possible sur ce test est de "
              f"{pct(max_precision())}).{other_text}"),
        ]),
    ]

    has_note = has_manual
    header = ["Modèle", "Temps de réponse", "Mémoire utilisée", "Consignes respectées"] + (["Note équipe"] if has_note else [])
    widths = [5, 3.6, 3.6, 4.8] if not has_note else [4.4, 3.2, 3.2, 3.6, 2.6]
    rows = []
    for _, r in summary_view.iterrows():
        row = [r["model"], fmt(r["temps_s"], "{:.0f}", " s"), fmt(r["ram_mo"] / 1024, "{:.1f}", " Go"), pct(r["consignes"])]
        rows.append(row + ([fmt(r["note"], "{:.1f}", "/5")] if has_note else []))

    fastest_missed = missed_rules(fastest)
    speed_text = f"<b>{fastest['model']}</b> est le plus rapide, avec {fmt(fastest['temps_s'], '{:.0f}')} s en moyenne"
    if fastest_missed:
        speed_text += f", mais il n'arrive pas à : {', '.join(fastest_missed)}."
    else:
        speed_text += "."
    story += [
        KeepTogether([
            p("Répondre au client", H2),
            table(header, rows, widths, highlight_row=chosen_row),
            Spacer(1, 6),
            p(speed_text),
        ]),
    ]

    answer_time = fmt(llm["temps_s"], "{:.0f}")
    choice_story = [
        p("4. Notre choix", H1),
        p(f"Nous choisissons <b>{choice['llm']}</b> pour répondre et <b>{choice['embedding']}</b> pour chercher."),
    ]
    if compliant:
        reasons = [
            "<b>Adapté à un client novice</b> : il répond en français, s'appuie sur les arômes du catalogue "
            "et finit par une question pour préciser le besoin.",
            f"<b>Le plus rapide des modèles qui respectent toutes les consignes</b> : environ {answer_time} s par réponse.",
            f"<b>Compatible avec notre contrainte locale</b> : il utilise {fmt(llm['ram_mo'] / 1024, '{:.1f}')} Go de mémoire sur un ordinateur sans carte graphique dédiée.",
        ]
        choice_story += bullets(reasons)
    else:
        choice_story.append(p("Aucun modèle ne respecte toutes les consignes : ce choix est le meilleur compromis."))
    choice_story.append(Spacer(1, 4))
    choice_story.append(p(
        f"Notre score compte {QUALITY_WEIGHT:.0%} pour la qualité et {SPEED_WEIGHT:.0%} pour la vitesse. "
        f"{choice['llm']} obtient {fmt(100 * choice['score'], '{:.0f}')}/100.".replace("%", " %")
        + (f" {pairs.iloc[0]['llm']} a un score plus élevé grâce à sa vitesse, mais nous l'écartons car il ne "
           "respecte pas les consignes." if compliant and pairs.iloc[0]["llm"] != choice["llm"] else "")
    ))
    choice_story.append(p(
        f"Ce résultat vient de {len(QUESTIONS)} questions, d'un catalogue fictif et d'une seule machine : "
        "il reste à confirmer sur de vrais usages."
    ))
    story.append(KeepTogether(choice_story))

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Benchmark des modèles Ollama — Plateforme de commande intelligente Givaudan",
                            author=TEAM)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    print(f"PDF généré : {PDF_PATH}")
    print(f"Choix : {choice['llm']} + {choice['embedding']} (score {100 * choice['score']:.0f}/100)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    build()
