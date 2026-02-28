#!/usr/bin/env python3
"""Run persona-basis experiments with activation extraction, PCA, and steering tests.

This script implements an end-to-end reproducible pipeline:
1) Load local persona datasets.
2) Extract hidden-state persona vectors from a real causal LM.
3) Fit PCA and quantify geometry/stability.
4) Perform steering interventions with top PCs vs random directions.
5) Test alignment with external Big-5 labels.
6) Save metrics, plots, and intermediate artifacts.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import DatasetDict, load_from_disk
from scipy.linalg import subspace_angles
from scipy.stats import pearsonr, shapiro, wilcoxon
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ExperimentConfig:
    seed: int = 42
    model_name: str = "Qwen/Qwen2.5-0.5B"
    persona_samples: int = 800
    steering_eval_prompts: int = 48
    trait_samples: int = 800
    max_length_extract: int = 128
    max_new_tokens: int = 24
    batch_size: int = 64
    pca_components: int = 20
    steering_alpha: float = 8.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for path in ["results", "results/plots", "figures", "logs", "src"]:
        Path(path).mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/experiment.log"),
            logging.StreamHandler(),
        ],
    )


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def encode_trait_answer(answer: str) -> Dict[str, int]:
    # answer format: '{"O":"n", "C":"n", ...}'
    parsed = json.loads(answer)
    return {k: int(v.lower() == "y") for k, v in parsed.items()}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(samp)))
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_data(config: ExperimentConfig) -> Dict[str, DatasetDict]:
    persona = load_from_disk("datasets/personahub_persona")
    persona_chat = load_from_disk("datasets/persona_chat")
    myp = load_from_disk("datasets/mypersonality")

    return {"persona": persona, "persona_chat": persona_chat, "myp": myp}


def data_quality_report(data: Dict[str, DatasetDict], config: ExperimentConfig) -> Dict[str, object]:
    persona_texts = [clean_text(t) for t in data["persona"]["train"]["persona"][: config.persona_samples * 2]]
    persona_texts = [t for t in persona_texts if t]

    missing_persona = sum(1 for t in persona_texts if not t)
    dup_persona = len(persona_texts) - len(set(persona_texts))
    lengths = np.array([len(t.split()) for t in persona_texts], dtype=float)

    chat_train = data["persona_chat"]["train"]
    chat_missing = 0
    for i in range(min(len(chat_train), 500)):
        row = chat_train[i]
        if not row["personality"] or not row["utterances"]:
            chat_missing += 1

    myp_train = data["myp"]["train"]
    trait_fail = 0
    for i in range(min(len(myp_train), 500)):
        try:
            _ = encode_trait_answer(myp_train[i]["answer"])
        except Exception:
            trait_fail += 1

    report = {
        "persona_rows_checked": len(persona_texts),
        "persona_missing": missing_persona,
        "persona_duplicates": dup_persona,
        "persona_len_mean": float(lengths.mean()),
        "persona_len_std": float(lengths.std(ddof=1)),
        "persona_len_min": int(lengths.min()),
        "persona_len_max": int(lengths.max()),
        "persona_outliers_gt_3sigma": int(np.sum(lengths > lengths.mean() + 3 * lengths.std(ddof=1))),
        "persona_chat_rows_checked": min(len(chat_train), 500),
        "persona_chat_missing_rows": chat_missing,
        "myp_rows_checked": min(len(myp_train), 500),
        "myp_parse_failures": trait_fail,
    }
    return report


def sample_persona_texts(data: Dict[str, DatasetDict], n: int, seed: int) -> List[str]:
    texts = [clean_text(t) for t in data["persona"]["train"]["persona"]]
    texts = [t for t in texts if t]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    return [texts[i] for i in idx]


def build_persona_prompts(texts: List[str]) -> List[str]:
    return [f"System: You are defined by this persona: {t}\nUser: Introduce yourself briefly.\nAssistant:" for t in texts]


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def extract_mean_hidden(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    device: str,
    batch_size: int,
    max_len: int,
) -> np.ndarray:
    feats: List[np.ndarray] = []

    for batch in batch_iter(prompts, batch_size):
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(**enc, output_hidden_states=True, use_cache=False)

        hs = out.hidden_states[layer_idx + 1]  # hidden_states[0] is embeddings
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        feats.append(pooled.detach().float().cpu().numpy())

    return np.concatenate(feats, axis=0)


def fit_pca(vectors: np.ndarray, n_components: int) -> Tuple[PCA, np.ndarray]:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(centered)
    return pca, scores


def subspace_stability(vectors: np.ndarray, n_components: int, n_trials: int = 5) -> Dict[str, float]:
    rng = np.random.default_rng(42)
    n = vectors.shape[0]
    overlaps = []

    for _ in range(n_trials):
        idx1 = rng.choice(n, size=n // 2, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[idx1] = False
        idx2 = np.where(mask)[0]

        p1 = PCA(n_components=n_components, random_state=42).fit(vectors[idx1])
        p2 = PCA(n_components=n_components, random_state=42).fit(vectors[idx2])

        # principal angles between two k-dim subspaces
        angles = subspace_angles(p1.components_.T, p2.components_.T)
        overlaps.append(float(np.mean(np.cos(angles))))

    arr = np.array(overlaps)
    return {
        "subspace_overlap_mean": float(arr.mean()),
        "subspace_overlap_std": float(arr.std(ddof=1)),
        "subspace_overlap_ci_low": float(np.quantile(arr, 0.025)),
        "subspace_overlap_ci_high": float(np.quantile(arr, 0.975)),
    }


def random_orthonormal_basis(dim: int, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(dim, k))
    q, _ = np.linalg.qr(mat)
    return q.T


def compute_random_basis_evr(vectors: np.ndarray, k: int, trials: int = 200) -> Dict[str, float]:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    total_var = float(np.sum(np.var(centered, axis=0, ddof=1)))

    vals = []
    for i in range(trials):
        basis = random_orthonormal_basis(centered.shape[1], k, seed=100 + i)
        proj = centered @ basis.T
        var = float(np.sum(np.var(proj, axis=0, ddof=1)))
        vals.append(var / total_var)

    arr = np.array(vals)
    return {
        "random_basis_k_var_mean": float(arr.mean()),
        "random_basis_k_var_std": float(arr.std(ddof=1)),
        "random_basis_k_var_ci_low": float(np.quantile(arr, 0.025)),
        "random_basis_k_var_ci_high": float(np.quantile(arr, 0.975)),
    }


def build_pc_lexicons(texts: List[str], scores: np.ndarray, pc_idx: int, top_n: int = 120, lex_n: int = 25) -> Tuple[List[str], List[str]]:
    order = np.argsort(scores[:, pc_idx])
    low_texts = [texts[i] for i in order[:top_n]]
    high_texts = [texts[i] for i in order[-top_n:]]

    vec = CountVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(high_texts + low_texts)
    vocab = np.array(vec.get_feature_names_out())

    high_mean = np.asarray(X[:top_n].mean(axis=0)).ravel()
    low_mean = np.asarray(X[top_n:].mean(axis=0)).ravel()
    delta = high_mean - low_mean

    pos_words = vocab[np.argsort(delta)[-lex_n:]].tolist()
    neg_words = vocab[np.argsort(delta)[:lex_n]].tolist()
    return pos_words, neg_words


def extract_eval_prompts_from_personachat(data: Dict[str, DatasetDict], n: int, seed: int) -> List[str]:
    rows = data["persona_chat"]["validation"]
    prompts = []
    for i in range(len(rows)):
        hist = rows[i]["utterances"][0]["history"]
        if hist:
            prompts.append(f"User: {hist[-1]}\nAssistant:")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(prompts), size=min(n, len(prompts)), replace=False)
    return [prompts[i] for i in idx]


def token_count_matches(words: List[str], text: str) -> int:
    lower = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    toks = lower.split()
    word_set = set(words)
    return sum(1 for t in toks if t in word_set)


def build_add_direction_hook(direction: torch.Tensor, alpha: float):
    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs2 = hs.clone()
            hs2[:, -1, :] = hs2[:, -1, :] + alpha * direction.to(hs2.device, hs2.dtype)
            return (hs2,) + output[1:]

        hs = output
        hs2 = hs.clone()
        hs2[:, -1, :] = hs2[:, -1, :] + alpha * direction.to(hs2.device, hs2.dtype)
        return hs2

    return hook_fn


def generate_with_optional_hook(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int,
    batch_size: int = 16,
) -> List[str]:
    results = []
    for batch in batch_iter(prompts, batch_size):
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        results.extend(texts)

    return results


def steering_experiment(
    model,
    tokenizer,
    prompts: List[str],
    layer_module,
    pc_direction: np.ndarray,
    random_direction: np.ndarray,
    alpha: float,
    pos_words: List[str],
    neg_words: List[str],
    device: str,
    max_new_tokens: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    dvec = torch.tensor(pc_direction, dtype=torch.float32, device=device)
    rvec = torch.tensor(random_direction, dtype=torch.float32, device=device)

    dvec = dvec / torch.norm(dvec)
    rvec = rvec / torch.norm(rvec)

    # Base
    base_texts = generate_with_optional_hook(model, tokenizer, prompts, device, max_new_tokens)

    # PC+
    h = layer_module.register_forward_hook(build_add_direction_hook(dvec, alpha))
    pc_plus = generate_with_optional_hook(model, tokenizer, prompts, device, max_new_tokens)
    h.remove()

    # PC-
    h = layer_module.register_forward_hook(build_add_direction_hook(dvec, -alpha))
    pc_minus = generate_with_optional_hook(model, tokenizer, prompts, device, max_new_tokens)
    h.remove()

    # Random+
    h = layer_module.register_forward_hook(build_add_direction_hook(rvec, alpha))
    rnd_plus = generate_with_optional_hook(model, tokenizer, prompts, device, max_new_tokens)
    h.remove()

    rows = []
    for i, prompt in enumerate(prompts):
        for cond, text in [
            ("base", base_texts[i]),
            ("pc_plus", pc_plus[i]),
            ("pc_minus", pc_minus[i]),
            ("random_plus", rnd_plus[i]),
        ]:
            pos = token_count_matches(pos_words, text)
            neg = token_count_matches(neg_words, text)
            rows.append(
                {
                    "prompt": prompt,
                    "condition": cond,
                    "text": text,
                    "pos_hits": pos,
                    "neg_hits": neg,
                    "score": pos - neg,
                }
            )

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="prompt", columns="condition", values="score", aggfunc="mean")

    # Statistical comparisons
    pc_plus_vs_base = pivot["pc_plus"].values - pivot["base"].values
    pc_plus_vs_rand = pivot["pc_plus"].values - pivot["random_plus"].values
    pc_minus_vs_base = pivot["pc_minus"].values - pivot["base"].values

    # Use Shapiro as light normality check; fallback to Wilcoxon.
    shapiro_p = float(shapiro(pc_plus_vs_base).pvalue) if len(pc_plus_vs_base) <= 5000 else 0.0

    w1 = wilcoxon(pc_plus_vs_base, alternative="greater", zero_method="wilcox")
    w2 = wilcoxon(pc_plus_vs_rand, alternative="greater", zero_method="wilcox")
    w3 = wilcoxon(pc_minus_vs_base, alternative="less", zero_method="wilcox")

    ci1 = bootstrap_ci(pc_plus_vs_base)
    ci2 = bootstrap_ci(pc_plus_vs_rand)
    ci3 = bootstrap_ci(pc_minus_vs_base)

    metrics = {
        "steering_n_prompts": int(len(pivot)),
        "normality_shapiro_p_pcplus_minus_base": shapiro_p,
        "pc_plus_vs_base_mean_delta": float(np.mean(pc_plus_vs_base)),
        "pc_plus_vs_base_ci95": [ci1[0], ci1[1]],
        "pc_plus_vs_base_p": float(w1.pvalue),
        "pc_plus_vs_base_effect_d": cohens_d_paired(pivot["pc_plus"].values, pivot["base"].values),
        "pc_plus_vs_random_mean_delta": float(np.mean(pc_plus_vs_rand)),
        "pc_plus_vs_random_ci95": [ci2[0], ci2[1]],
        "pc_plus_vs_random_p": float(w2.pvalue),
        "pc_plus_vs_random_effect_d": cohens_d_paired(pivot["pc_plus"].values, pivot["random_plus"].values),
        "pc_minus_vs_base_mean_delta": float(np.mean(pc_minus_vs_base)),
        "pc_minus_vs_base_ci95": [ci3[0], ci3[1]],
        "pc_minus_vs_base_p": float(w3.pvalue),
        "pc_minus_vs_base_effect_d": cohens_d_paired(pivot["pc_minus"].values, pivot["base"].values),
        "score_means_by_condition": {
            c: float(df[df["condition"] == c]["score"].mean()) for c in ["base", "pc_plus", "pc_minus", "random_plus"]
        },
    }

    return df, metrics


def trait_alignment_analysis(
    data: Dict[str, DatasetDict],
    model,
    tokenizer,
    device: str,
    layer_idx: int,
    pca: PCA,
    vec_mean: np.ndarray,
    config: ExperimentConfig,
) -> Dict[str, object]:
    ds = data["myp"]["train"]
    n = min(config.trait_samples, len(ds))

    rng = np.random.default_rng(config.seed)
    idx = rng.choice(len(ds), size=n, replace=False)

    texts = [clean_text(ds[int(i)]["text"]) for i in idx]
    answers = [encode_trait_answer(ds[int(i)]["answer"]) for i in idx]

    prompts = [f"User post: {t}\nAssistant summary:" for t in texts]
    reps = extract_mean_hidden(
        model,
        tokenizer,
        prompts,
        layer_idx=layer_idx,
        device=device,
        batch_size=config.batch_size,
        max_len=config.max_length_extract,
    )

    centered = reps - vec_mean.reshape(1, -1)
    proj = pca.transform(centered)[:, :5]

    traits = ["O", "C", "E", "A", "N"]
    rows = []
    pvals = []
    for pc in range(proj.shape[1]):
        for tr in traits:
            y = np.array([a[tr] for a in answers], dtype=float)
            r, p = pearsonr(proj[:, pc], y)
            rows.append({"pc": pc + 1, "trait": tr, "r": float(r), "p": float(p)})
            pvals.append(p)

    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    for i, row in enumerate(rows):
        row["p_fdr_bh"] = float(p_adj[i])
        row["significant"] = bool(reject[i])

    out = pd.DataFrame(rows)
    sig = out[out["significant"]]

    return {
        "trait_alignment_rows": rows,
        "trait_alignment_significant_count": int(sig.shape[0]),
        "trait_alignment_top_abs_corr": float(out["r"].abs().max()),
    }


def save_plots(
    pca: PCA,
    steering_df: pd.DataFrame,
    trait_rows: List[Dict[str, object]],
) -> None:
    sns.set_theme(style="whitegrid")

    # Scree plot
    evr = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), marker="o")
    plt.title("Cumulative Explained Variance by PC")
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Explained Variance")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/plots/pca_scree.png", dpi=200)
    plt.close()

    # Steering distribution
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=steering_df, x="condition", y="score")
    plt.title("Steering Score by Intervention Condition")
    plt.xlabel("Condition")
    plt.ylabel("Lexical Persona Score (pos_hits - neg_hits)")
    plt.tight_layout()
    plt.savefig("results/plots/steering_scores_boxplot.png", dpi=200)
    plt.close()

    # Trait correlation heatmap
    tr = pd.DataFrame(trait_rows)
    piv = tr.pivot(index="trait", columns="pc", values="r")
    plt.figure(figsize=(8, 4))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Trait Correlation (r) with PC Projections")
    plt.xlabel("Principal Component")
    plt.ylabel("Big-5 Trait")
    plt.tight_layout()
    plt.savefig("results/plots/trait_pc_heatmap.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()
    setup_logging()

    config = ExperimentConfig()
    set_seed(config.seed)
    device = get_device()

    start = time.time()
    logging.info("Using device: %s", device)

    data = load_data(config)
    quality = data_quality_report(data, config)

    persona_texts = sample_persona_texts(data, config.persona_samples, config.seed)
    prompts = build_persona_prompts(persona_texts)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer_mid = n_layers // 2
    layer_last = n_layers - 1

    logging.info("Extracting persona vectors at layers %d and %d", layer_mid, layer_last)
    vec_mid = extract_mean_hidden(
        model,
        tokenizer,
        prompts,
        layer_idx=layer_mid,
        device=device,
        batch_size=config.batch_size,
        max_len=config.max_length_extract,
    )
    vec_last = extract_mean_hidden(
        model,
        tokenizer,
        prompts,
        layer_idx=layer_last,
        device=device,
        batch_size=config.batch_size,
        max_len=config.max_length_extract,
    )

    # PCA on last layer for main analysis.
    pca, scores = fit_pca(vec_last, config.pca_components)
    stability = subspace_stability(vec_last, n_components=min(10, config.pca_components), n_trials=6)
    rand_k = compute_random_basis_evr(vec_last, k=10, trials=200)

    evr10 = float(np.sum(pca.explained_variance_ratio_[:10]))
    evr20 = float(np.sum(pca.explained_variance_ratio_[: min(20, len(pca.explained_variance_ratio_))]))

    # Cross-layer subspace similarity
    p_mid = PCA(n_components=10, random_state=42).fit(vec_mid)
    p_last = PCA(n_components=10, random_state=42).fit(vec_last)
    cross_angles = subspace_angles(p_mid.components_.T, p_last.components_.T)
    cross_overlap = float(np.mean(np.cos(cross_angles)))

    # Build lexical probes from PC1 extremes
    pos_words, neg_words = build_pc_lexicons(persona_texts, scores, pc_idx=0)

    eval_prompts = extract_eval_prompts_from_personachat(data, config.steering_eval_prompts, config.seed)
    pc1_direction = pca.components_[0]

    # Random baseline direction with matched norm
    rng = np.random.default_rng(config.seed)
    rnd = rng.normal(size=pc1_direction.shape)
    rnd = rnd / np.linalg.norm(rnd)

    layer_module = model.model.layers[layer_last] if hasattr(model.model, "layers") else model.transformer.h[layer_last]

    steering_df, steering_metrics = steering_experiment(
        model,
        tokenizer,
        eval_prompts,
        layer_module,
        pc_direction=pc1_direction,
        random_direction=rnd,
        alpha=config.steering_alpha,
        pos_words=pos_words,
        neg_words=neg_words,
        device=device,
        max_new_tokens=config.max_new_tokens,
    )

    # Trait alignment uses PCA basis from last-layer persona vectors.
    trait_metrics = trait_alignment_analysis(
        data,
        model,
        tokenizer,
        device,
        layer_idx=layer_last,
        pca=pca,
        vec_mean=vec_last.mean(axis=0),
        config=config,
    )

    metrics = {
        "config": asdict(config),
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else [],
            "device": device,
            "batch_size": config.batch_size,
            "mixed_precision": bool(device == "cuda"),
            "model_name": config.model_name,
            "layers": n_layers,
            "layer_mid": layer_mid,
            "layer_last": layer_last,
        },
        "data_quality": quality,
        "geometry": {
            "persona_n": len(persona_texts),
            "hidden_dim": int(vec_last.shape[1]),
            "evr_top10": evr10,
            "evr_top20": evr20,
            "pc1_evr": float(pca.explained_variance_ratio_[0]),
            "pc2_evr": float(pca.explained_variance_ratio_[1]),
            "cross_layer_subspace_overlap_top10": cross_overlap,
            **stability,
            **rand_k,
        },
        "steering": steering_metrics,
        "trait_alignment": {
            k: v for k, v in trait_metrics.items() if k != "trait_alignment_rows"
        },
        "runtime_seconds": float(time.time() - start),
    }

    # Save artifacts
    pd.DataFrame({"persona_text": persona_texts, "pc1_score": scores[:, 0]}).to_csv(
        "results/persona_scores.csv", index=False
    )
    steering_df.to_csv("results/steering_outputs.csv", index=False)

    with open("results/trait_alignment_rows.json", "w", encoding="utf-8") as f:
        json.dump(trait_metrics["trait_alignment_rows"], f, indent=2)

    with open("results/data_quality.json", "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    with open("results/config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open("results/examples.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "persona_examples": persona_texts[:3],
                "pc1_positive_lexicon": pos_words,
                "pc1_negative_lexicon": neg_words,
                "steering_prompt_examples": eval_prompts[:3],
            },
            f,
            indent=2,
        )

    save_plots(pca, steering_df, trait_metrics["trait_alignment_rows"])

    logging.info("Experiment complete. Runtime: %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
