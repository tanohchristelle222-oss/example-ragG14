"""Génère benchmark_llm.pdf à la racine du repo à partir de benchmark/results/.

Usage, depuis la racine du repo :  python benchmark/make_report.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from questions import QUESTIONS

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmark" / "results"
PDF_PATH = ROOT / "benchmark_llm.pdf"

TEAM = "Équipe XXX"
QUALITY_WEIGHT = 0.6
SPEED_WEIGHT = 0.4
EXAMPLE_QUESTION = "Q1"
EXAMPLE_MAX_WORDS = 110
MIN_RULE_RATE = 0.75
VEGETARIAN_QUESTION = "Q7"
MEAT_AROMA = "Poulet rôti"
MONTHS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
HEADER_BG = colors.HexColor("#f1f0ec")
LINE = colors.HexColor("#d5d4cf")
HIGHLIGHT = colors.HexColor("#e6f0fb")


# --- Polices et styles -------------------------------------------------------


def register_fonts() -> tuple[str, str]:
    """Arial sous Windows, sinon DejaVu Sans fournie avec matplotlib."""
    windows = Path("C:/Windows/Fonts")
    regular, bold = windows / "arial.ttf", windows / "arialbd.ttf"
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    if not (regular.exists() and bold.exists()):
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        regular, bold = ttf / "DejaVuSans.ttf", ttf / "DejaVuSans-Bold.ttf"
        plt.rcParams["font.family"] = ["DejaVu Sans"]
    pdfmetrics.registerFont(TTFont("Corps", str(regular)))
    pdfmetrics.registerFont(TTFont("Corps-Gras", str(bold)))
    pdfmetrics.registerFontFamily("Corps", normal="Corps", bold="Corps-Gras",
                                  italic="Corps", boldItalic="Corps-Gras")
    return "Corps", "Corps-Gras"


FONT, FONT_BOLD = register_fonts()
BODY = ParagraphStyle("body", fontName=FONT, fontSize=10, leading=14.5, spaceAfter=7, textColor=colors.HexColor(INK))
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11.5, textColor=colors.HexColor(INK_SECONDARY))
H1 = ParagraphStyle("h1", parent=BODY, fontName=FONT_BOLD, fontSize=15, leading=19, spaceBefore=8, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=BODY, fontName=FONT_BOLD, fontSize=11, leading=14, spaceBefore=6, spaceAfter=4)
TITLE = ParagraphStyle("title", parent=BODY, fontName=FONT_BOLD, fontSize=25, leading=31, alignment=TA_CENTER, spaceAfter=20)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, fontSize=13, leading=18, alignment=TA_CENTER, textColor=colors.HexColor(INK_SECONDARY))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
CELL_HEAD = ParagraphStyle("cellhead", parent=CELL, fontName=FONT_BOLD)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)
ANSWER = ParagraphStyle("answer", parent=BODY, fontSize=9.5, leading=13, leftIndent=10, borderPadding=0)


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(item, BULLET, bulletText="•") for item in items]


def table(header: list[str], rows: list[list[str]], widths_cm: list[float], highlight_row: int | None = None) -> Table:
    data = [[p(h, CELL_HEAD) for h in header]] + [[p(str(c), CELL) for c in row] for row in rows]
    result = Table(data, colWidths=[w * cm for w in widths_cm], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row + 1), (-1, highlight_row + 1), HIGHLIGHT))
    result.setStyle(TableStyle(style))
    return result


def footer(canvas, doc) -> None:
    if doc.page == 1:
        return
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor(INK_SECONDARY))
    canvas.drawString(2 * cm, 1.2 * cm, "Benchmark des modèles Ollama")
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
        row = {
            "model": info["model"],
            "statut": info["statut"],
            "n_ok": len(ok),
            "n_total": len(model_rows),
            "ram_mo": info.get("ram_mo"),
            "taille_disque_mo": info.get("taille_disque_mo"),
            "chargement_s": info.get("chargement_s"),
            "note": manual.get(info["model"]),
        }
        if len(ok):
            row |= {
                "temps_s": ok["total_duration_s"].mean(),
                "tokens_s": ok["eval_count"].sum() / ok["eval_duration_s"].sum(),
                "mots": ok["nb_mots"].mean(),
                "cite": as_bool(ok["cite_arome"]).mean(),
                "invente": as_bool(ok["invente_arome"]).mean(),
                "francais": as_bool(ok["en_francais"]).mean(),
                "question": as_bool(ok["termine_par_question"]).mean(),
                "tronquee": as_bool(ok["tronquee"]).mean(),
            }
            auto = (row["cite"] + (1 - row["invente"]) + row["francais"] + row["question"]) / 4
            row["qualite_auto"] = auto
            row["qualite"] = auto if pd.isna(row["note"]) else (auto + row["note"] / 5) / 2
        rows.append(row)
    return pd.DataFrame(rows)


def max_precision() -> float:
    """Precision@3 maximale possible : certaines questions ont moins de 3 arômes attendus."""
    return sum(min(len(q["attendus"]), 3) / 3 for q in QUESTIONS) / len(QUESTIONS)


def score_pairs(summary: pd.DataFrame, embed: pd.DataFrame) -> pd.DataFrame:
    llms = summary[summary["n_ok"] == summary["n_total"]]
    if llms.empty:
        llms = summary[summary["n_ok"] > 0]
    embeds = embed[embed["statut"] == "ok"]
    pmax = max_precision()
    rows = []
    for _, llm in llms.iterrows():
        for _, emb in embeds.iterrows():
            embed_quality = (emb["hit_at_3"] + min(emb["precision_at_3"] / pmax, 1)) / 2
            rows.append({
                "llm": llm["model"],
                "embedding": emb["model"],
                "qualite_llm": llm["qualite"],
                "qualite_embedding": embed_quality,
                "qualite": (llm["qualite"] + embed_quality) / 2,
                "temps_s": llm["temps_s"] + emb["temps_moyen_requete_s"],
            })
    pairs = pd.DataFrame(rows)
    pairs["vitesse"] = pairs["temps_s"].min() / pairs["temps_s"]
    pairs["score"] = QUALITY_WEIGHT * pairs["qualite"] + SPEED_WEIGHT * pairs["vitesse"]
    return pairs.sort_values("score", ascending=False, ignore_index=True)


# --- Formats -----------------------------------------------------------------


def fmt(value, pattern: str = "{:.1f}", suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return pattern.format(value).replace(".", ",") + suffix


def pct(value) -> str:
    return fmt(None if value is None or pd.isna(value) else 100 * value, "{:.0f}", " %")


def go(mo) -> str:
    return fmt(None if mo is None or pd.isna(mo) else mo / 1024, "{:.1f}", " Go")


def status_text(row: pd.Series) -> str:
    if row["n_ok"] == row["n_total"]:
        return ""
    return f"{row['n_ok']}/{row['n_total']} réponses ({escape(str(row['statut'])[:70])})"


def printable(text: str) -> str:
    """Remplace les caractères absents de la police, puis échappe pour reportlab."""
    text = "".join(c if ord(c) < 0x2100 else "" for c in text)
    return escape(text)


def shorten(answer: str, max_words: int) -> str:
    tokens = answer.split(" ")
    if len(tokens) <= max_words:
        return answer
    return " ".join(tokens[:max_words]) + " […]"


def answer_markup(answer: str) -> str:
    html = printable(answer)
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html)
    html = re.sub(r"(?m)^\s*[-*•]\s+", "– ", html)
    return html.replace("*", "").replace("\n\n", "<br/>").replace("\n", "<br/>")


# --- Graphiques --------------------------------------------------------------


def style_axes(ax, axis: str) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK_SECONDARY, labelsize=9)


def save_figure(fig, name: str) -> Path:
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def embed_chart(embed: pd.DataFrame) -> Path:
    ok = embed[embed["statut"] == "ok"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    width = 0.28
    gap = 0.02
    for offset, column, label, color in (
        (-(width + gap) / 2, "hit_at_3", "Hit@3", SERIES_1),
        ((width + gap) / 2, "precision_at_3", "Precision@3", SERIES_2),
    ):
        bars = ax.bar(ok.index + offset, ok[column], width, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.2f}".replace(".", ",") for v in ok[column]],
                     padding=3, fontsize=9, color=INK)
    ax.set_xticks(ok.index, ok["model"], color=INK, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    style_axes(ax, "y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=2, frameon=False, fontsize=9, labelcolor=INK)
    return save_figure(fig, "chart_embeddings.png")


def bar_chart(data: pd.DataFrame, column: str, unit: str, name: str, ascending: bool) -> Path:
    # barh trace de bas en haut : le dernier de la liste apparaît en haut.
    ordered = data.dropna(subset=[column]).sort_values(column, ascending=ascending)
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(ordered) + 0.5))
    bars = ax.barh(ordered["model"], ordered[column], height=0.5, color=SERIES_1)
    ax.bar_label(bars, labels=[f"{v:.1f} {unit}".replace(".", ",") for v in ordered[column]],
                 padding=4, fontsize=9, color=INK)
    ax.set_xlim(0, ordered[column].max() * 1.18)
    style_axes(ax, "x")
    ax.tick_params(axis="y", labelcolor=INK)
    ax.grid(False)  # Chaque barre porte sa valeur : l'axe serait redondant.
    ax.set_xticks([])
    return save_figure(fig, name)


def image(path: Path, width_cm: float = 15.5) -> Image:
    pixel_width, pixel_height = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * pixel_height / pixel_width)


# --- Recommandation ----------------------------------------------------------


def recommendation_text(pairs: pd.DataFrame, summary: pd.DataFrame, embed: pd.DataFrame, used_embed: str, has_manual: bool) -> list:
    best = pairs.iloc[0]
    llm = summary[summary["model"] == best["llm"]].iloc[0]
    emb = embed[embed["model"] == best["embedding"]].iloc[0]
    fastest = summary.dropna(subset=["temps_s"]).sort_values("temps_s").iloc[0]
    best_quality = summary.dropna(subset=["qualite"]).sort_values("qualite", ascending=False).iloc[0]

    story = [p(f"<b>Couple recommandé : {best['llm']} + {best['embedding']}</b>, score {fmt(100 * best['score'], '{:.0f}')}/100.")]
    quality_source = "des indicateurs automatiques et des notes de l'équipe" if has_manual else "des indicateurs automatiques"
    story.append(p(
        f"Sur cette machine, {best['llm']} répond en {fmt(llm['temps_s'])} s en moyenne à "
        f"{fmt(llm['tokens_s'])} tokens/s, et sa qualité atteint {fmt(100 * llm['qualite'], '{:.0f}')}/100 "
        f"(moyenne {quality_source}). Il cite un arôme fourni dans {pct(llm['cite'])} des réponses, "
        f"répond en français dans {pct(llm['francais'])} des cas, finit par une question dans "
        f"{pct(llm['question'])} des cas et invente un arôme dans {pct(llm['invente'])} des cas."
    ))
    story.append(p(
        f"Côté recherche, {best['embedding']} obtient un Hit@3 de {fmt(emb['hit_at_3'], '{:.2f}')} et une "
        f"Precision@3 de {fmt(emb['precision_at_3'], '{:.2f}')}, pour environ "
        f"{fmt(1000 * emb['temps_moyen_requete_s'], '{:.0f}')} ms par requête."
    ))

    comments = []
    top_quality = summary[summary["qualite"] >= best_quality["qualite"] - 1e-9].sort_values("temps_s")
    if best["llm"] not in set(top_quality["model"]):
        names = ", ".join(f"{r['model']} ({fmt(r['temps_s'])} s)" for _, r in top_quality.iterrows())
        comments.append(f"La meilleure qualité ({fmt(100 * best_quality['qualite'], '{:.0f}')}/100) revient à : {names}. "
                        "Leur temps de réponse pèse sur leur score.")
    if fastest["model"] != best["llm"]:
        comments.append(f"{fastest['model']} est le plus rapide ({fmt(fastest['temps_s'])} s), "
                        f"avec une qualité de {fmt(100 * fastest['qualite'], '{:.0f}')}/100.")
    other_llm = pairs[pairs["llm"] != best["llm"]]
    if not other_llm.empty:
        runner_up = other_llm.iloc[0]
        gap = 100 * (best["score"] - runner_up["score"])
        if gap < 5:
            comments.append(f"L'écart avec {runner_up['llm']} est faible ({fmt(gap)} points) : "
                            "les notes de l'équipe peuvent changer le classement.")
    if llm.get("tronquee", 0) and llm["tronquee"] > 0:
        comments.append(f"{pct(llm['tronquee'])} de ses réponses ont atteint la limite de longueur.")
    if comments:
        story.append(p(" ".join(comments)))

    def missed_rules(row: pd.Series) -> list[str]:
        rules = {"citer un arôme fourni": row["cite"], "répondre en français": row["francais"],
                 "finir par une question": row["question"], "ne pas inventer d'arôme": 1 - row["invente"]}
        return [f"{label} ({pct(rate)} des réponses)" for label, rate in rules.items() if rate < MIN_RULE_RATE]

    missed = missed_rules(llm)
    if missed:
        warning = (f"<b>Point de vigilance :</b> {best['llm']} gagne surtout grâce à sa vitesse, mais il respecte "
                   f"mal cette consigne : {', '.join(missed)}.")
        compliant_models = [r["model"] for _, r in summary.dropna(subset=["qualite"]).iterrows() if not missed_rules(r)]
        compliant = pairs[pairs["llm"].isin(compliant_models)]
        if not compliant.empty:
            alt = compliant.iloc[0]
            warning += (f" Si cette consigne est indispensable, le meilleur couple qui la respecte est "
                        f"<b>{alt['llm']} + {alt['embedding']}</b> (score {fmt(100 * alt['score'], '{:.0f}')}/100).")
        story.append(p(warning))

    if best["embedding"] != used_embed:
        story.append(p(
            f"Attention : les réponses des LLM ont été évaluées avec le contexte de {used_embed}, "
            f"et non de {best['embedding']}. La combinaison recommandée n'a donc pas été testée de bout en bout."
        ))
    if not has_manual:
        story.append(p(
            "Les notes manuelles de l'équipe ne sont pas encore renseignées : ce classement repose "
            "uniquement sur des indicateurs automatiques, qui ne jugent pas la pertinence du conseil."
        ))
    return story


# --- Construction du PDF -----------------------------------------------------


def french_date(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def build() -> None:
    embed = read_csv("embed_results.csv")
    llm = read_csv("llm_results.csv")
    models = read_csv("llm_models.csv")
    run_info = json.loads((RESULTS_DIR / "run_info.json").read_text(encoding="utf-8"))
    manual = read_manual_scores()
    summary = summarise_llm(llm, models, manual)
    usable = summary[summary["n_ok"] > 0]
    if usable.empty:
        sys.exit("Aucun LLM n'a produit de réponse : rapport impossible.")
    pairs = score_pairs(summary, embed)
    used_embed = run_info["embedding_retenu"]
    machine = run_info["machine"]
    has_manual = not manual.empty

    story: list = [
        Spacer(1, 6 * cm),
        p("Benchmark des modèles Ollama — Plateforme de commande intelligente Givaudan", TITLE),
        Spacer(1, 1.5 * cm),
        p(TEAM, SUBTITLE),
        p(french_date(date.today()), SUBTITLE),
        PageBreak(),
    ]

    story += [
        p("1. Contexte et besoin", H1),
        p("Un client de Givaudan, souvent novice, décrit son besoin en français, par exemple "
          "« un arôme fruité pour un yaourt à boire ». La plateforme doit retrouver les arômes adaptés "
          "dans un catalogue, puis répondre comme un conseiller."),
        p("Le système repose sur deux types de modèles :"),
        *bullets([
            "<b>Un modèle d'embeddings</b> transforme les textes en vecteurs pour retrouver les arômes proches de la demande (RAG).",
            "<b>Un LLM</b> rédige la réponse : dialogue en français, ton accueillant, 2 ou 3 arômes justifiés, puis une question pour préciser le besoin.",
        ]),
        p("La contrainte principale est de fonctionner à 100 % en local, sans aucune API externe, "
          "sur un ordinateur portable modeste."),
        p("2. Méthodologie", H1),
        p("Machine de test", H2),
        table(["Élément", "Valeur détectée"], [
            ["Système", machine.get("os", "—")],
            ["Processeur", f"{machine.get('cpu', '—')} ({machine.get('coeurs_logiques', '—')} cœurs logiques)"],
            ["Mémoire vive", f"{fmt(machine.get('ram_go'))} Go"],
            ["Carte graphique", f"{machine.get('gpu', '—')} (pas de GPU dédié, calcul sur CPU)"],
        ], [4, 13]),
        Spacer(1, 8),
        p("Modèles testés", H2),
        table(["Modèle", "Rôle", "Taille sur disque"],
              [[r["model"], "Embeddings", go(r.get("taille_disque_mo"))] for _, r in embed.iterrows()]
              + [[r["model"], "LLM", go(r["taille_disque_mo"])] for _, r in summary.iterrows()],
              [7, 5, 5]),
        Spacer(1, 8),
        p("Protocole", H2),
        *bullets([
            f"{len(QUESTIONS)} demandes clients en français, chacune avec ses arômes attendus (vérité terrain), sur un catalogue fictif de 48 arômes.",
            "Embeddings : top 3 par similarité cosinus. Hit@3 : au moins un arôme attendu dans le top 3. Precision@3 : part des 3 résultats qui sont attendus.",
            f"LLM : même contexte pour tous (les 3 arômes trouvés par {used_embed}, le meilleur en Hit@3 puis Precision@3), même consigne système.",
            f"Température {fmt(run_info['temperature'])}, mode thinking désactivé, réponse limitée à {run_info['num_predict']} tokens par sécurité.",
            "Un seul modèle en mémoire à la fois : un appel de chauffe non compté avant chaque modèle, puis déchargement après.",
            "Mesures issues d'Ollama : durée totale, chargement, tokens générés, vitesse et mémoire occupée.",
            "Qualité automatique : cite un arôme fourni, invente un arôme absent du catalogue (détection simple), répond en français, finit par une question.",
        ]),
    ]

    embed_rows = []
    for _, r in embed.iterrows():
        if r["statut"] == "ok":
            embed_rows.append([r["model"], fmt(r["hit_at_3"], "{:.2f}"), fmt(r["precision_at_3"], "{:.2f}"),
                               fmt(r["temps_indexation_s"], "{:.1f}", " s"),
                               fmt(1000 * r["temps_moyen_requete_s"], "{:.0f}", " ms"), go(r["ram_mo"])])
        else:
            embed_rows.append([r["model"], escape(str(r["statut"])), "—", "—", "—", "—"])
    story += [
        p("3. Résultats des embeddings", H1),
        table(["Modèle", "Hit@3", "Precision@3", "Indexation (48 arômes)", "Temps par requête", "RAM"],
              embed_rows, [5, 2, 2.4, 3, 2.6, 2]),
        Spacer(1, 8),
        p("Hit@3 et Precision@3 par modèle d'embeddings (plus c'est haut, mieux c'est)", H2),
        image(embed_chart(embed)),
        p(f"La Precision@3 ne peut pas dépasser {fmt(max_precision(), '{:.2f}')} sur ce jeu de test, "
          "car certaines questions ont moins de 3 arômes attendus.", SMALL),
        p(f"Le modèle retenu pour fournir le contexte aux LLM est <b>{used_embed}</b>."),
    ]

    llm_rows = []
    for _, r in summary.iterrows():
        if r["n_ok"]:
            llm_rows.append([r["model"], fmt(r["temps_s"], "{:.1f}", " s"), fmt(r["tokens_s"]), go(r["ram_mo"]),
                             pct(r["cite"]), pct(r["invente"]), pct(r["francais"]), pct(r["question"]),
                             fmt(r["note"], "{:.1f}", "/5")])
        else:
            llm_rows.append([r["model"], "échec", "—", "—", "—", "—", "—", "—", "—"])
    notes = [f"{r['model']} : {status_text(r)}" for _, r in summary.iterrows() if status_text(r)]
    story += [
        PageBreak(),
        p("4. Résultats des LLM", H1),
        table(["Modèle", "Temps moyen", "Tokens /s", "RAM", "Cite un arôme", "Invente", "Français", "Finit par une question", "Note équipe"],
              llm_rows, [3.4, 1.8, 1.5, 1.5, 1.7, 1.6, 1.7, 2, 1.8]),
        Spacer(1, 4),
        p("Temps de chargement à froid : " + ", ".join(
            f"{r['model']} {fmt(r['chargement_s'], '{:.1f}', ' s')}" for _, r in usable.iterrows()) + ".", SMALL),
        p(("Note équipe : pas encore renseignée dans manual_scores.csv." if not has_manual
           else "Note équipe : moyenne des notes de manual_scores.csv."), SMALL),
    ]
    if notes:
        story.append(p("Modèles incomplets : " + " ; ".join(notes) + ".", SMALL))
    story += [
        KeepTogether([p("Vitesse de génération en tokens par seconde (plus c'est long, mieux c'est)", H2),
                      image(bar_chart(usable, "tokens_s", "tokens/s", "chart_vitesse.png", ascending=True))]),
        KeepTogether([p("Temps de réponse moyen en secondes (plus c'est court, mieux c'est)", H2),
                      image(bar_chart(usable, "temps_s", "s", "chart_temps.png", ascending=False))]),
    ]

    question = next(q for q in QUESTIONS if q["id"] == EXAMPLE_QUESTION)
    story += [PageBreak(), p("5. Exemple de réponses", H1), p(f"Question : « {escape(question['question'])} »")]
    example_rows = llm[(llm["question_id"] == EXAMPLE_QUESTION) & (llm["statut"] == "ok")]
    for _, r in example_rows.iterrows():
        text = "" if pd.isna(r["reponse"]) else str(r["reponse"])
        story += [
            p(escape(r["model"]), H2),
            p(answer_markup(shorten(text, EXAMPLE_MAX_WORDS)) or "(réponse vide)", ANSWER),
        ]

    vegetarian = llm[(llm["question_id"] == VEGETARIAN_QUESTION) & (llm["statut"] == "ok")]
    meat_count = int(vegetarian["aromes_cites"].fillna("").str.contains(MEAT_AROMA).sum())
    limits_extra = []
    if meat_count:
        limits_extra.append(
            f"Pour le bouillon végétarien, {meat_count} modèle(s) sur {len(vegetarian)} citent « {MEAT_AROMA} » : "
            "ce contresens n'est pas détecté par les indicateurs automatiques, d'où l'intérêt des notes de l'équipe."
        )

    pair_rows = [[r["llm"], r["embedding"], fmt(100 * r["qualite"], "{:.0f}"), fmt(100 * r["vitesse"], "{:.0f}"),
                  fmt(100 * r["score"], "{:.0f}")] for _, r in pairs.iterrows()]
    story += [
        PageBreak(),
        p("6. Recommandation", H1),
        p(f"Chaque couple LLM + embeddings reçoit un score sur 100 : {QUALITY_WEIGHT:.0%} de qualité et "
          f"{SPEED_WEIGHT:.0%} de vitesse.".replace("%", " %")),
        *bullets([
            "<b>Qualité</b> : moyenne de la qualité du LLM (ses 4 indicateurs automatiques, et les notes de l'équipe pour moitié si elles existent) et de la qualité des embeddings (Hit@3 et Precision@3 ramenée à son maximum possible).",
            "<b>Vitesse</b> : temps total de réponse du couple, ramené au couple le plus rapide (100 = le plus rapide).",
        ]),
        Spacer(1, 4),
        table(["LLM", "Embeddings", "Qualité", "Vitesse", "Score"], pair_rows, [5, 5, 2.3, 2.3, 2.4], highlight_row=0),
        Spacer(1, 10),
        *recommendation_text(pairs, summary, embed, used_embed, has_manual),
        p("7. Limites", H1),
        *bullets([
            "Le catalogue est fictif : 48 arômes créés pour la démo, sans données réelles Givaudan.",
            f"Le jeu de test est petit ({len(QUESTIONS)} questions) : un écart de quelques points n'est pas significatif.",
            "Les mesures viennent d'une seule machine, qui exécutait d'autres applications : les temps changeraient sur un autre ordinateur.",
            "Les indicateurs automatiques sont simples : ils peuvent manquer un arôme inventé ou en signaler un à tort, et ne jugent pas la pertinence du conseil.",
            "Chaque LLM a été évalué avec le contexte d'un seul modèle d'embeddings.",
            *limits_extra,
        ]),
    ]

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Benchmark des modèles Ollama — Plateforme de commande intelligente Givaudan",
                            author=TEAM)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    print(f"PDF généré : {PDF_PATH}")
    print("\nRécapitulatif :")
    print(pairs[["llm", "embedding", "qualite", "vitesse", "score"]].round(3).to_string(index=False))
    best = pairs.iloc[0]
    print(f"\nCouple recommandé : {best['llm']} + {best['embedding']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    build()
