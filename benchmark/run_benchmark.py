"""Benchmark des modèles Ollama pour la plateforme de commande intelligente.

Usage, depuis la racine du repo :  python benchmark/run_benchmark.py

A. Embeddings : retrouve-t-on les bons arômes du catalogue ?
B. LLM : chaque modèle répond avec exactement les mêmes 3 arômes de contexte.

Les modèles sont testés un par un et déchargés entre chaque test : deux modèles
ne sont jamais en mémoire en même temps. Résultats dans benchmark/results/.
"""

from __future__ import annotations

import csv
import ctypes
import difflib
import json
import os
import platform
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import ollama

from questions import QUESTIONS

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_PATH = ROOT / "data" / "catalogue_aromes.csv"
RESULTS_DIR = ROOT / "benchmark" / "results"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODELS = ["nomic-embed-text", "bge-m3"]
LLM_MODELS = ["qwen2.5:1.5b", "qwen3.5:4b", "llama3.2:3b", "gemma3:4b"]

TOP_K = 3
TEMPERATURE = 0.2
# Garde-fou : environ 2 fois la longueur demandée, pour qu'un modèle qui part
# en boucle ne bloque pas le benchmark.
NUM_PREDICT = 400
REQUEST_TIMEOUT_S = 900
KEEP_ALIVE_DURING_TEST = "10m"
MAX_CONSECUTIVE_ERRORS = 2

SYSTEM_PROMPT = """Tu es un conseiller Givaudan spécialisé en arômes alimentaires. \
Ton ton est professionnel et accueillant, et tu t'adresses à un client novice : \
utilise des mots simples.

Règles à respecter :
- Réponds UNIQUEMENT à partir des arômes fournis dans le message, n'en invente jamais d'autres.
- Propose 2 ou 3 arômes en citant leur nom exact, avec une courte justification pour chacun.
- Termine par UNE seule question pour préciser le besoin du client.
- Réponds en français, en moins de 150 mots."""

# Ingrédients courants, pour repérer un arôme inventé. Ceux qui figurent déjà
# dans le catalogue sont retirés automatiquement (voir build_invention_detector).
FLAVOUR_LEXICON = """banane kiwi melon pasteque papaye goyave grenade figue datte prune
mirabelle raisin myrtille groseille canneberge cranberry clementine mandarine bergamote
tonka amaretto rhum whisky nougat praline reglisse anis safran muscade girofle poivre
curcuma cumin ail tomate thym romarin persil coriandre bacon jambon boeuf lardon truffe
sesame cacahuete amande pecan macadamia violette jasmin tilleul verveine sureau rhubarbe
nectarine cacao""".split()

# Premiers mots de noms du catalogue trop génériques pour signaler un arôme.
GENERIC_NAME_WORDS = {
    "fruits", "fruit", "biscuit", "the", "fume", "beurre", "creme", "lait", "fleur", "rose", "noix",
}
NAME_STOPWORDS = {"arome", "aromes", "de", "des", "du", "d", "la", "le", "les", "l", "au", "aux", "et", "a"}

FR_WORDS = set(
    "le la les un une des du de et est pour avec votre vous nous que qui dans sur ce cette ou au aux".split()
)
EN_WORDS = set("the and for with is are you your we that this of to in or a an".split())
NON_LATIN = re.compile(r"[Ѐ-ӿ؀-ۿ぀-ヿ㐀-鿿가-힯]")

client = ollama.Client(host=OLLAMA_HOST, timeout=REQUEST_TIMEOUT_S)


# --- Outils généraux ---------------------------------------------------------


def norm(text: str) -> str:
    """Minuscules, sans accents, apostrophes droites."""
    text = unicodedata.normalize("NFKD", text.replace("’", "'"))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(text))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    # utf-8-sig : UTF-8 avec BOM, pour qu'Excel affiche bien les accents.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_catalogue() -> list[dict]:
    with CATALOGUE_PATH.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def full_name(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def installed_models() -> dict[str, int]:
    return {m.model: m.size or 0 for m in client.list().models}


def to_mb(size_bytes: float) -> int:
    return round(size_bytes / 1024**2)


def unload(model: str) -> None:
    """Demande à Ollama de libérer le modèle de la RAM."""
    try:
        client.generate(model=model, prompt="", keep_alive=0)
    except ollama.ResponseError:
        # Les modèles d'embeddings ne savent pas « générer ».
        try:
            client.embed(model=model, input="", keep_alive=0)
        except Exception as exc:  # noqa: BLE001
            print(f"   ! Déchargement de {model} impossible : {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"   ! Déchargement de {model} impossible : {exc}")


def unload_everything(timeout_s: float = 60) -> None:
    """Vide la mémoire d'Ollama et attend que ce soit effectif."""
    deadline = time.monotonic() + timeout_s
    requested: set[str] = set()
    while time.monotonic() < deadline:
        loaded = [m.model for m in client.ps().models]
        if not loaded:
            return
        for model in loaded:
            if model not in requested:
                unload(model)
                requested.add(model)
        time.sleep(0.5)
    print("   ! Un modèle est encore en mémoire après 60 s, on continue quand même.")


def loaded_ram_mb(model: str) -> int | None:
    for m in client.ps().models:
        if m.model == full_name(model):
            return to_mb(m.size or 0)
    return None


def detect_machine() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "cpu": platform.processor() or "non détecté",
        "coeurs_logiques": os.cpu_count(),
        "ram_go": None,
        "gpu": "non détecté",
    }
    if platform.system() != "Windows":
        return info

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        info["ram_go"] = round(status.ullTotalPhys / 1024**3, 1)

    def wmi(class_name: str) -> str:
        command = f"(Get-CimInstance {class_name}).Name -join ', '"
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True, text=True, timeout=30, check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    info["cpu"] = wmi("Win32_Processor") or info["cpu"]
    info["gpu"] = wmi("Win32_VideoController") or info["gpu"]
    return info


# --- A. Embeddings -----------------------------------------------------------


def aroma_text(aroma: dict) -> str:
    fields = ("nom", "famille", "description", "notes_gout", "applications")
    return ". ".join(aroma[field] for field in fields)


def embed(model: str, texts: list[str], prefix: str) -> np.ndarray:
    vectors = np.array(
        client.embed(model=model, input=[prefix + t for t in texts],
                     keep_alive=KEEP_ALIVE_DURING_TEST)["embeddings"],
        dtype=float,
    )
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def benchmark_embedding_model(model: str, catalogue: list[dict]) -> tuple[dict, list[dict]]:
    doc_prefix = "search_document: " if "nomic" in model else ""
    query_prefix = "search_query: " if "nomic" in model else ""

    embed(model, ["chauffe"], doc_prefix)  # Appel de chauffe, non compté.
    ram = loaded_ram_mb(model)

    start = time.perf_counter()
    doc_vectors = embed(model, [aroma_text(a) for a in catalogue], doc_prefix)
    index_time = time.perf_counter() - start

    details, hits, precisions, query_times = [], [], [], []
    for q in QUESTIONS:
        start = time.perf_counter()
        query_vector = embed(model, [q["question"]], query_prefix)[0]
        query_times.append(time.perf_counter() - start)

        scores = doc_vectors @ query_vector
        top = np.argsort(-scores)[:TOP_K]
        top_ids = [catalogue[i]["id"] for i in top]
        n_good = sum(i in q["attendus"] for i in top_ids)
        hits.append(n_good > 0)
        precisions.append(n_good / TOP_K)
        for rank, i in enumerate(top, start=1):
            details.append({
                "model": model,
                "question_id": q["id"],
                "question": q["question"],
                "rang": rank,
                "id": catalogue[i]["id"],
                "nom": catalogue[i]["nom"],
                "score_cosinus": round(float(scores[i]), 4),
                "attendu": catalogue[i]["id"] in q["attendus"],
            })

    summary = {
        "model": model,
        "statut": "ok",
        "hit_at_3": round(float(np.mean(hits)), 4),
        "precision_at_3": round(float(np.mean(precisions)), 4),
        "temps_indexation_s": round(index_time, 3),
        "temps_moyen_requete_s": round(float(np.mean(query_times)), 4),
        "dimension": doc_vectors.shape[1],
        "ram_mo": ram,
    }
    return summary, details


def run_embeddings(catalogue: list[dict], installed: dict[str, int]) -> list[dict]:
    print("\n=== A. Benchmark des embeddings ===")
    summaries, details = [], []
    for n, model in enumerate(EMBED_MODELS, start=1):
        print(f"[{n}/{len(EMBED_MODELS)}] {model}")
        size = installed.get(full_name(model))
        if size is None:
            print(f"   ! {model} n'est pas installé : ollama pull {model}")
            summaries.append({"model": model, "statut": "non installé"})
            continue
        unload_everything()
        try:
            summary, model_details = benchmark_embedding_model(model, catalogue)
            summary["taille_disque_mo"] = to_mb(size)
            summaries.append(summary)
            details.extend(model_details)
            print(f"   Hit@3 = {summary['hit_at_3']:.2f}  Precision@3 = {summary['precision_at_3']:.2f}"
                  f"  indexation = {summary['temps_indexation_s']:.1f} s")
        except Exception as exc:  # noqa: BLE001
            print(f"   ! Échec de {model} : {exc}")
            summaries.append({"model": model, "statut": f"erreur : {exc}"})
        finally:
            unload_everything()

    write_csv(RESULTS_DIR / "embed_results.csv", summaries, [
        "model", "statut", "hit_at_3", "precision_at_3", "temps_indexation_s",
        "temps_moyen_requete_s", "dimension", "taille_disque_mo", "ram_mo",
    ])
    write_csv(RESULTS_DIR / "embed_top3.csv", details, [
        "model", "question_id", "question", "rang", "id", "nom", "score_cosinus", "attendu",
    ])
    return summaries


def best_embedding(summaries: list[dict]) -> str | None:
    usable = [s for s in summaries if s["statut"] == "ok"]
    if not usable:
        return None
    # Départage final par le temps de requête, pour un choix toujours déterminé.
    return max(usable, key=lambda s: (s["hit_at_3"], s["precision_at_3"], -s["temps_moyen_requete_s"]))["model"]


def retrieve_contexts(model: str, catalogue: list[dict]) -> dict[str, list[dict]]:
    """Top 3 de chaque question, figé pour que tous les LLM aient le même contexte."""
    with (RESULTS_DIR / "embed_top3.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter=";") if r["model"] == model]
    by_id = {a["id"]: a for a in catalogue}
    contexts: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: int(r["rang"])):
        contexts.setdefault(row["question_id"], []).append(by_id[row["id"]])
    return contexts


# --- B. Indicateurs qualité --------------------------------------------------


def build_invention_detector(catalogue: list[dict]):
    catalogue_vocab: set[str] = set()
    names: list[list[str]] = []
    for aroma in catalogue:
        catalogue_vocab.update(words(" ".join(
            aroma[f] for f in ("nom", "famille", "description", "notes_gout", "applications")
        )))
        names.append([w for w in words(aroma["nom"]) if w not in NAME_STOPWORDS])
    outside_catalogue = set(FLAVOUR_LEXICON) - catalogue_vocab
    head_words = {name[0] for name in names} - GENERIC_NAME_WORDS

    def is_known(candidate: list[str]) -> bool:
        for name in names:
            if set(candidate) <= set(name) or set(name) <= set(candidate):
                return True
            if difflib.SequenceMatcher(None, " ".join(candidate), " ".join(name)).ratio() >= 0.85:
                return True
        return False

    def detect(answer: str, question: str, context: list[dict]) -> list[str]:
        allowed = set(words(question))
        for aroma in context:
            allowed.update(words(" ".join(aroma.values())))
        found = sorted(set(words(answer)) & outside_catalogue - allowed)

        # Noms mis en avant : texte en gras ou début de ligne de liste.
        spans = re.findall(r"\*\*(.+?)\*\*", answer)
        spans += re.findall(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+(?:\*\*)?([^:\n–—(*]{2,60})", answer)
        for span in spans:
            candidate = [w for w in words(span) if w not in NAME_STOPWORDS]
            if (0 < len(candidate) <= 5 and candidate[0] in head_words | outside_catalogue
                    and not is_known(candidate)):
                found.append(span.strip())
        return sorted(set(found))

    return detect


def is_french(answer: str) -> bool:
    if NON_LATIN.search(answer):
        return False
    tokens = re.findall(r"[a-zàâäçéèêëîïôöùûüœ']+", answer.lower())
    if not tokens:
        return False
    french = sum(t.split("'")[-1] in FR_WORDS or t.split("'")[0] in FR_WORDS for t in tokens)
    english = sum(t in EN_WORDS for t in tokens)
    return french / len(tokens) >= 0.12 and french > english * 2


def ends_with_question(answer: str) -> bool:
    text = answer.rstrip()
    while text and not (text[-1].isalnum() or text[-1] in ".?!…:;,"):
        text = text[:-1]  # Retire markdown, guillemets et emojis de fin.
    return text.endswith("?")


def word_count(answer: str) -> int:
    return len(re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", answer))


def clean_answer(answer: str) -> str:
    return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()


# --- B. LLM ------------------------------------------------------------------


def format_context(aromas: list[dict]) -> str:
    return "\n".join(
        f"- {a['nom']} (famille : {a['famille']}) : {a['description']}. "
        f"Notes de goût : {a['notes_gout']}. Applications : {a['applications']}. "
        f"Forme : {a['forme']}. Dosage conseillé : {a['dosage_conseille']}. "
        f"Certifications : {a['certifications']}. "
        f"Prix indicatif : {a['prix_indicatif_eur_kg']} EUR/kg. "
        f"Quantité minimale : {a['quantite_min_kg']} kg."
        for a in aromas
    )


def build_messages(question: str, aromas: list[dict]) -> list[dict]:
    user = f"Arômes disponibles :\n{format_context(aromas)}\n\nDemande du client : {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def chat(model: str, messages: list[dict], think_supported: bool, num_predict: int = NUM_PREDICT):
    options = {"temperature": TEMPERATURE, "num_predict": num_predict}
    kwargs = {"think": False} if think_supported else {}
    return client.chat(model=model, messages=messages, options=options,
                       keep_alive=KEEP_ALIVE_DURING_TEST, **kwargs)


def warm_up(model: str):
    """Appel de chauffe non compté. Détermine si think=False est accepté."""
    messages = [{"role": "user", "content": "Bonjour"}]
    try:
        return chat(model, messages, True, num_predict=5), True
    except ollama.ResponseError as exc:
        if "think" not in str(exc).lower():
            raise
        print("   (le mode thinking n'est pas pris en charge par ce modèle)")
        return chat(model, messages, False, num_predict=5), False


def seconds(nanoseconds: int | None) -> float:
    return round((nanoseconds or 0) / 1e9, 3)


def run_question(model: str, q: dict, aromas: list[dict], think_supported: bool, detect) -> dict:
    response = chat(model, build_messages(q["question"], aromas), think_supported)
    answer = clean_answer(response.message.content or "")
    names = [a["nom"] for a in aromas]
    cited = [n for n in names if norm(n) in norm(answer)]
    invented = detect(answer, q["question"], aromas)
    eval_seconds = seconds(response.eval_duration)
    return {
        "statut": "ok",
        "reponse": answer,
        "total_duration_s": seconds(response.total_duration),
        "load_duration_s": seconds(response.load_duration),
        "prompt_eval_duration_s": seconds(response.prompt_eval_duration),
        "eval_duration_s": eval_seconds,
        "eval_count": response.eval_count,
        "tokens_par_s": round((response.eval_count or 0) / eval_seconds, 2) if eval_seconds else 0,
        "cite_arome": bool(cited),
        "aromes_cites": ", ".join(cited),
        "invente_arome": bool(invented),
        "aromes_inventes": ", ".join(invented),
        "en_francais": is_french(answer),
        "termine_par_question": ends_with_question(answer),
        "nb_mots": word_count(answer),
        "tronquee": response.done_reason == "length",
    }


def error_rows(model: str, reason: str) -> list[dict]:
    return [{"model": model, "question_id": q["id"], "question": q["question"], "statut": reason} for q in QUESTIONS]


def benchmark_llm_model(model, contexts, detect, row_saver) -> tuple[list[dict], dict]:
    unload_everything()
    print("   Chauffe et chargement du modèle (non compté)...")
    warm, think_supported = warm_up(model)
    info = {
        "statut": "ok",
        "chargement_s": seconds(warm.load_duration),
        "ram_mo": loaded_ram_mb(model),
        "think_desactive": think_supported,
    }

    rows, consecutive_errors = [], 0
    for n, q in enumerate(QUESTIONS, start=1):
        print(f"   Question {n}/{len(QUESTIONS)} ({q['id']})...", end=" ", flush=True)
        base = {"model": model, "question_id": q["id"], "question": q["question"]}
        try:
            row = base | run_question(model, q, contexts[q["id"]], think_supported, detect)
            consecutive_errors = 0
            print(f"{row['total_duration_s']:.1f} s, {row['tokens_par_s']:.1f} tok/s")
        except Exception as exc:  # noqa: BLE001
            row = base | {"statut": f"erreur : {exc}"}
            consecutive_errors += 1
            print(f"ÉCHEC : {exc}")
        rows.append(row)
        row_saver(row)
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS and n < len(QUESTIONS):
            print("   ! Trop d'erreurs d'affilée, on passe au modèle suivant.")
            remaining = {r["question_id"] for r in rows}
            for later in error_rows(model, "non exécutée (erreurs répétées)"):
                if later["question_id"] not in remaining:
                    rows.append(later)
                    row_saver(later)
            info["statut"] = "erreurs répétées"
            break
    return rows, info


LLM_FIELDS = [
    "model", "question_id", "question", "statut", "total_duration_s", "load_duration_s",
    "prompt_eval_duration_s", "eval_duration_s", "eval_count", "tokens_par_s", "cite_arome",
    "aromes_cites", "invente_arome", "aromes_inventes", "en_francais", "termine_par_question",
    "nb_mots", "tronquee", "reponse",
]


def run_llms(contexts, catalogue, installed) -> list[dict]:
    print("\n=== B. Benchmark des LLM ===")
    detect = build_invention_detector(catalogue)
    all_rows: list[dict] = []
    model_infos: list[dict] = []

    def save_row(row: dict) -> None:
        # Sauvegarde à chaque réponse : une coupure ne perd pas ce qui est fait.
        all_rows.append(row)
        write_csv(RESULTS_DIR / "llm_results.csv", all_rows, LLM_FIELDS)

    for n, model in enumerate(LLM_MODELS, start=1):
        print(f"\n[{n}/{len(LLM_MODELS)}] {model}")
        info = {"model": model}
        size = installed.get(full_name(model))
        if size is None:
            print(f"   ! {model} n'est pas installé : ollama pull {model}")
            info["statut"] = "non installé"
            for row in error_rows(model, "modèle non installé"):
                save_row(row)
        else:
            info["taille_disque_mo"] = to_mb(size)
            try:
                _, details = benchmark_llm_model(model, contexts, detect, save_row)
                info |= details
            except Exception as exc:  # noqa: BLE001
                print(f"   ! Échec de {model} (manque de RAM ?) : {exc}")
                info["statut"] = f"erreur : {exc}"
                for row in error_rows(model, f"erreur : {exc}"):
                    save_row(row)
            finally:
                unload_everything()
        model_infos.append(info)
        write_csv(RESULTS_DIR / "llm_models.csv", model_infos, [
            "model", "statut", "taille_disque_mo", "ram_mo", "chargement_s", "think_desactive",
        ])
    return all_rows


def write_answers(rows: list[dict], contexts, embed_model: str) -> None:
    lines = [
        "# Réponses des modèles, à noter par l'équipe",
        "",
        f"Contexte retrouvé avec `{embed_model}`. Notez chaque réponse sur 5 dans "
        "`manual_scores.csv`.",
    ]
    for q in QUESTIONS:
        lines += ["", f"## {q['id']} — {q['question']}", ""]
        lines.append("Arômes fournis : " + ", ".join(a["nom"] for a in contexts[q["id"]]))
        for row in (r for r in rows if r["question_id"] == q["id"]):
            lines += ["", f"### {row['model']}", ""]
            if row["statut"] != "ok":
                lines.append(f"*Pas de réponse : {row['statut']}*")
                continue
            lines += [
                row["reponse"] or "*(réponse vide)*",
                "",
                f"*{row['total_duration_s']:.1f} s · {row['tokens_par_s']:.1f} tokens/s · "
                f"{row['nb_mots']} mots · arômes cités : {row['aromes_cites'] or 'aucun'} · "
                f"inventés : {row['aromes_inventes'] or 'aucun'}*",
            ]
    (RESULTS_DIR / "llm_answers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_manual_scores_template() -> None:
    path = RESULTS_DIR / "manual_scores.csv"
    if path.exists():
        print(f"{path.name} existe déjà, il n'est pas modifié.")
        return
    rows = [{"model": m, "question_id": q["id"]} for m in LLM_MODELS for q in QUESTIONS]
    write_csv(path, rows, ["model", "question_id", "score_sur_5", "commentaire"])


def print_summary(rows: list[dict]) -> None:
    print("\n=== Récapitulatif LLM ===")
    print(f"{'Modèle':<16}{'Temps moy.':>12}{'Tokens/s':>10}{'Cite':>7}{'Invente':>9}{'Français':>10}{'Question':>10}")
    for model in LLM_MODELS:
        ok = [r for r in rows if r["model"] == model and r["statut"] == "ok"]
        if not ok:
            print(f"{model:<16}   aucun résultat")
            continue
        def pct(key: str) -> str:
            return f"{100 * sum(bool(r[key]) for r in ok) / len(ok):.0f} %"
        time_mean = np.mean([r["total_duration_s"] for r in ok])
        speed_mean = np.mean([r["tokens_par_s"] for r in ok])
        print(f"{model:<16}{time_mean:>10.1f} s{speed_mean:>10.1f}{pct('cite_arome'):>7}"
              f"{pct('invente_arome'):>9}{pct('en_francais'):>10}{pct('termine_par_question'):>10}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    catalogue = load_catalogue()
    print(f"Catalogue : {len(catalogue)} arômes, {len(QUESTIONS)} questions.")

    try:
        installed = installed_models()
    except Exception as exc:  # noqa: BLE001
        print(f"Ollama ne répond pas ({exc}). Lancez l'application Ollama puis recommencez.")
        return 1

    run_started = time.perf_counter()
    embed_summaries = run_embeddings(catalogue, installed)
    embed_model = best_embedding(embed_summaries)
    if embed_model is None:
        print("Aucun modèle d'embeddings n'a fonctionné : benchmark LLM impossible.")
        return 1
    print(f"\nMeilleur modèle d'embeddings : {embed_model}")

    contexts = retrieve_contexts(embed_model, catalogue)
    rows = run_llms(contexts, catalogue, installed)
    write_answers(rows, contexts, embed_model)
    create_manual_scores_template()

    run_info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": detect_machine(),
        "ollama_python": ollama.__version__ if hasattr(ollama, "__version__") else None,
        "embedding_retenu": embed_model,
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "duree_totale_min": round((time.perf_counter() - run_started) / 60, 1),
        "contextes": {qid: [a["id"] for a in aromas] for qid, aromas in contexts.items()},
    }
    (RESULTS_DIR / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print_summary(rows)
    print(f"\nTerminé en {run_info['duree_totale_min']} min. Résultats dans {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
