import json
import os
from typing import List, Tuple, Optional, Callable
from tqdm import tqdm
import pandas as pd
import pickle
from datetime import datetime


def make_instance(instruction: str, text: str) -> str:
    return instruction + text + "\n\n<Tuple List>\n"


def build_pred_tuples(
    response_list: List[List[dict]]
) -> List[List[Tuple[str, str, str, Optional[float]]]]:
    pred_tuple: List[List[Tuple[str, str, str, Optional[float]]]] = []
    for r in response_list:
        if not r:
            pred_tuple.append([])
            continue

        seen = set()
        ordered: List[Tuple[str, str, str, Optional[float]]] = []

        for it in r:
            e = (str(it.get("entity", "")).strip().lower() or "null")
            f = (str(it.get("feature", "")).strip().lower() or "null")
            o = (str(it.get("opinion", "")).strip().lower() or "null")
            try:
                ov = float(it.get("opinion_value"))
                ov = max(-1.0, min(1.0, ov))
            except Exception:
                ov = None

            key = (e, f, o, ov)
            if key not in seen:
                seen.add(key)
                ordered.append(key)

        pred_tuple.append(ordered)

    return pred_tuple


def add_tuples_to_df(
    df: pd.DataFrame,
    response_list: List[List[dict]],
) -> pd.DataFrame:
    out = df.copy()
    out["pred_tuples"] = build_pred_tuples(response_list)
    return out


def save_pred_tuples_to_pickle(
    output_dir: str,
    model_name: str,
    pred_tuples: List[List[Tuple[str, str, str, Optional[float]]]],
    base_name: str = "inference_result",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    out_pickle = os.path.join(output_dir, f"{model_short}_{base_name}.pickle")
    
    with open(out_pickle, "wb") as f:
        pickle.dump(pred_tuples, f)
    
    print("Inference completed. Results saved at:", out_pickle)


def save_dataset(
    df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    base_name: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    current_time = datetime.now().strftime("%Y%m%d-%H%M")

    out_path = os.path.join(output_dir, f"{base_name}_{model_short}_{current_time}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def default_type_rewrite(content_type: str) -> str:
    mapping = {
        "youtube": "youtube video thread comments",
        "reddit": "car community thread comments",
        "blog": "blog post",
        "review_site": "car user review text",
        "laptops": "laptop comments",
        "restaurants": "restaurants comments",
        "electronics": "electronic products reviews",
    }
    return mapping.get(content_type, "user comments")


def run_inference(
    *,
    df: pd.DataFrame,
    text_col: str, # Column containing the text of user reviews/comments.
    tokenizer,
    run_local_model_chat: Callable[[str, str], str],
    instruction: str,
    content_type: Optional[str] = "generic",  # ex.: "reddit"
    MODEL_NAME: str,
    system: str = "You are an opinion mining assistant.",
    MAX_TOKENS: int = 7500,
    SAFE_TOKENS: int = 5000,
    max_attempts: int = 2,
) -> List[List[dict]]:
    if text_col not in df.columns:
        raise ValueError(f"text_col='{text_col}' not found in DataFrame.")

    response_list: List[List[dict]] = []

    for i in tqdm(range(len(df)), desc=f"Processing {MODEL_NAME}"):
        foe_prompt = instruction.replace('{REPLACE}', default_type_rewrite(content_type))
        row_text = str(df.iat[i, df.columns.get_loc(text_col)])
        input_text_ = make_instance(foe_prompt, row_text)

        len_tokenize = len(tokenizer.tokenize(input_text_))
        if len_tokenize > MAX_TOKENS:
            print(f"Warning: Instance {i} too long ({len_tokenize} tokens) -> truncating to {SAFE_TOKENS}")
            input_text = tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(input_text_)[:SAFE_TOKENS]
            )
        else:
            input_text = input_text_

        attempt = 0
        success = False
        result_json = {"opinion_tuple": []}

        while attempt < max_attempts and not success:
            try:
                raw_content = run_local_model_chat(system, input_text)

                try:
                    result_json = json.loads(raw_content)
                except json.JSONDecodeError:
                    corrected_content = raw_content.replace("'", '"').strip()
                    result_json = json.loads(corrected_content)

                success = True
                print(f"Output sample: {raw_content[:100]}...")
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt >= max_attempts:
                    print("Max attempts reached. Skipping this instance.")
                    result_json = {"opinion_tuple": []}
                    break

        response_list.append(result_json.get('opinion_tuple', []))

    return response_list


