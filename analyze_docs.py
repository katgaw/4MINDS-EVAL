#!/usr/bin/env python3
"""Ask the 4minds model about a document, then score its answers locally.

The model only receives the PDF and the questions. Ground-truth answers from
questions.txt are used after the API calls return, and are never sent to 4MINDS.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DOCUMENT = ROOT / "data" / "va-124-soubor-vozidla.pdf"
DEFAULT_QUESTIONS = ROOT / "questions.txt"
DEFAULT_OUTPUT = ROOT / "eval_results.json"
BASE_URL = "https://api.4minds.ai"
MODEL_NAME = "4minds"
DATASET_NAME = "va-124-soubor-vozidla"

READY_STATUSES = {"ready", "completed", "complete", "active", "online", "deployed", "available", "success"}
FAILED_STATUSES = {"failed", "error", "cancelled", "canceled"}
BUILDING_STATUSES = {
    "building",
    "training",
    "processing",
    "pending",
    "queued",
    "new",
    "preparing",
    "in_progress",
    "running",
    "creating",
    "",
}

STOPWORDS = {
    "aby", "ale", "ani", "az", "bez", "bude", "budou", "byl", "byla", "bylo",
    "byt", "ci", "co", "do", "ho", "jak", "jako", "je", "jeho", "jej", "jeji",
    "jejich", "jen", "jeste", "ji", "jiz", "jsem", "jsi", "jsou", "jsme",
    "ktera", "ktere", "ktery", "kdyz", "kde", "kdo", "mezi", "muj", "na",
    "nad", "nam", "nas", "nebo", "neboť", "nebot", "nej", "neni", "nic",
    "nove", "od", "pak", "po", "pod", "podle", "pokud", "pouze", "pro",
    "proc", "proto", "protoze", "pred", "pri", "sice", "si", "se", "svuj",
    "sve", "ta", "tak", "take", "tam", "ten", "tento", "tato", "toto", "tim",
    "toho", "tom", "tomu", "tu", "tuto", "ty", "uz", "vam", "vas", "ve",
    "vice", "vsak", "vse", "za", "ze", "zpet",
}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def content_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", fold(text))
    return [word for word in words if len(word) >= 4 and word not in STOPWORDS]


def load_qa(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    key_at = next(
        (
            index
            for index, line in enumerate(lines)
            if "odpovedni klic" in fold(line) or fold(line.strip()) in {"answer key", "ground truth"}
        ),
        None,
    )
    if key_at is None:
        raise SystemExit(f"{path} has no answer-key section")

    questions = [
        line.strip()
        for line in lines[:key_at]
        if line.strip() and fold(line.strip()) not in {"otazky", "otazka", "questions", "question"}
    ]
    answers = [line.strip() for line in lines[key_at + 1 :] if line.strip()]
    if not questions or len(questions) != len(answers):
        raise SystemExit(
            f"Expected paired questions and answers, found {len(questions)} questions and {len(answers)} answers"
        )
    return list(zip(questions, answers))


def build_query(question: str) -> str:
    return (
        "Odpověz výhradně podle nahraného dokumentu s pojistnými podmínkami vozidel. "
        "Nepoužívej informace, které v dokumentu nejsou. "
        "Začni krátkým verdiktem a potom uveď stručné odůvodnění podle ustanovení dokumentu. "
        "Verdikt může být ano, ne, nebo přesné rozlišení dvou různých situací, pokud se otázka ptá na obě.\n\n"
        f"Otázka: {question}"
    )


def build_ocr_query(question: str) -> str:
    return (
        "Answer only from the uploaded receipt images. Do not use outside knowledge. "
        "Use only the receipt named in the question. "
        "Reply with the printed amounts, counts, and names. Do not add values that are not on that receipt.\n\n"
        f"Question: {question}"
    )


def assert_ground_truth_stays_local(query: str, answers: list[str]) -> None:
    for answer in answers:
        snippet = answer[:120]
        if snippet and snippet in query:
            raise RuntimeError("Refusing to send a ground-truth answer to 4MINDS")


def decision_head(gold: str) -> str:
    return re.split(r"[.!?]", gold.strip(), maxsplit=1)[0].strip()


def decision_kind(gold: str) -> str:
    head = fold(decision_head(gold))
    if head.startswith("prvni skoda ano"):
        return "split"
    if re.match(r"ne\b", head):
        return "no"
    if re.match(r"ano\b", head):
        return "yes"
    return "other"


def lead(text: str, sentences: int = 2) -> str:
    parts = [part.strip() for part in re.split(r"[.!?]", text.strip()) if part.strip()]
    return ". ".join(parts[:sentences])


def has_negation(text: str) -> bool:
    folded = fold(text)
    return bool(
        re.search(
            r"(?:^|[\s,;(])(?:ne|nikoliv|nikoli)\b|"
            r"neni|nejsou|nebude|nebudou|neposkyt|nekry|nehrad|neuplatn|"
            r"nescit|nesect|nezahrn|nepromit|neovliv|odepr",
            folded,
        )
    )


def has_affirmation(text: str) -> bool:
    folded = fold(text)
    if re.match(r"ano\b", folded.strip()):
        return True
    return bool(
        re.search(
            r"(?<!ne)(?:poskytn|uhrad|kryj\w*|kryta|kryte|kryto|zanikn|zahrn)\w*",
            folded,
        )
    )


def token_f1(prediction: str, gold: str) -> float:
    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in stems(prediction):
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in stems(gold):
        gold_counts[token] = gold_counts.get(token, 0) + 1
    if not pred_counts or not gold_counts:
        return 0.0
    overlap = sum(min(pred_counts[token], gold_counts[token]) for token in pred_counts if token in gold_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(gold_counts.values())
    return 2 * precision * recall / (precision + recall)


def clean(text: str) -> str:
    text = re.sub(r"[*_`>#]+", " ", text)
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", text).strip()


def stems(text: str) -> list[str]:
    folded = []
    for token in content_tokens(clean(text)):
        folded.append(token[:6] if len(token) > 6 else token)
    return folded


def content_recall(prediction: str, reference: str) -> float:
    expected = stems(reference)
    if not expected:
        return 0.0
    found = set(stems(prediction))
    return sum(1 for token in expected if token in found) / len(expected)


def leading_polarity(text: str) -> str:
    opening = fold(clean(lead(text, 1)))
    if re.match(r"(ano|yes)\b", opening):
        return "yes"
    if re.match(r"(ne|no)\b", opening):
        return "no"
    return ""


def decision_matches(prediction: str, gold: str) -> bool:
    kind = decision_kind(gold)
    cleaned = clean(prediction)
    window = cleaned[:1200]
    if kind == "split":
        folded = fold(window)
        return "prvni" in folded and "druh" in folded and has_negation(folded) and has_affirmation(folded)
    if kind == "no":
        return leading_polarity(cleaned) != "yes" and has_negation(clean(lead(cleaned, 2)))
    if kind == "yes":
        if leading_polarity(cleaned) == "no":
            return False
        head = decision_head(gold)
        if len(stems(head)) >= 2:
            return content_recall(window, head) >= 0.34
        return leading_polarity(cleaned) == "yes" or has_affirmation(window)
    return content_recall(window, decision_head(gold)) >= 0.34


def canon_number(token: str) -> str:
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token


def significant_numbers(text: str) -> list[str]:
    flattened = clean(text)
    flattened = re.sub(r"(\d),(\d{3})\b", r"\1\2", flattened)
    kept = []
    for token in re.findall(r"\d+\.\d+|\d+", flattened):
        canonical = canon_number(token)
        if "." in token or (canonical.isdigit() and int(canonical) >= 10):
            kept.append(canonical)
    return kept


def number_recall(prediction: str, gold: str) -> tuple[float, list[str]]:
    expected = significant_numbers(gold)
    if not expected:
        return 0.0, []
    found = set(significant_numbers(prediction))
    missing = [token for token in expected if token not in found]
    return (len(expected) - len(missing)) / len(expected), missing


def score_ocr_pair(prediction: str, gold: str) -> dict[str, object]:
    recall, missing = number_recall(prediction, gold)
    lexical = token_f1(prediction, gold)
    return {
        "passed": recall >= 0.75,
        "decision_match": recall >= 0.75,
        "decision": "amounts",
        "expected_decision": decision_head(gold),
        "lexical_f1": round(lexical, 4),
        "number_recall": round(recall, 4),
        "missing_numbers": missing,
    }


def score_pair(prediction: str, gold: str) -> dict[str, object]:
    lexical = token_f1(prediction, gold)
    matched = decision_matches(prediction, gold)
    return {
        "passed": matched or lexical >= 0.45,
        "decision_match": matched,
        "decision": decision_kind(gold),
        "expected_decision": decision_head(gold),
        "lexical_f1": round(lexical, 4),
    }


class FourMinds:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 120,
    ) -> tuple[str, bytes]:
        url = BASE_URL + path
        hdrs = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        data = body
        if json_body is not None:
            data = json.dumps(json_body).encode()
            hdrs["Content-Type"] = "application/json"
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.headers.get("Content-Type", ""), response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"{method} {path} failed ({exc.code}): {detail}") from None

    def get_json(self, path: str) -> object:
        _, raw = self.request("GET", path)
        return json.loads(raw.decode())

    def post_json(self, path: str, payload: dict, timeout: int = 180) -> tuple[str, bytes]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self.request("POST", path, json_body=payload, timeout=timeout)
            except RuntimeError as exc:
                last_error = exc
                if attempt == 2 or not any(code in str(exc) for code in ("429", "500", "502", "503")):
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(str(last_error))


def unwrap(payload: object) -> object:
    if isinstance(payload, dict) and payload.get("status") == "error":
        raise RuntimeError(str(payload.get("message") or payload))
    if isinstance(payload, dict) and payload.get("status") == "success" and "data" in payload:
        return payload["data"]
    return payload


def as_items(payload: object) -> list[dict]:
    data = unwrap(payload)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("items", "models", "datasets", "results", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if "id" in data:
            return [data]
    return []


def item_name(item: dict) -> str:
    for key in ("name", "model_name", "dataset_name"):
        if item.get(key):
            return str(item[key])
    return ""


def find_named(items: list[dict], name: str) -> dict | None:
    target = name.casefold()
    for item in items:
        if item_name(item).casefold() == target:
            return item
    return None


def entity_id(item: dict) -> int | str:
    if "id" not in item:
        raise RuntimeError(f"API response has no id: {json.dumps(item, ensure_ascii=False)[:400]}")
    value = item["id"]
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def status_of(model: dict, key: str) -> str:
    return str(model.get(key) or "").strip().lower()


def is_failed(model: dict) -> bool:
    return status_of(model, "status") in FAILED_STATUSES or status_of(model, "graph_status") in FAILED_STATUSES


def is_ready(model: dict) -> bool:
    status = status_of(model, "status")
    graph = status_of(model, "graph_status")
    if status in READY_STATUSES:
        return True
    if graph in READY_STATUSES and status not in BUILDING_STATUSES:
        return True
    progress = model.get("graph_build_progress")
    return graph in READY_STATUSES and progress in (100, "100")


def multipart(fields: dict[str, str], file_field: str, filename: str, content: bytes) -> tuple[str, bytes]:
    boundary = "----4minds" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        )
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
            f"Content-Type: application/pdf\r\n\r\n"
        ).encode()
        + content
        + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode())
    return boundary, b"".join(chunks)


def post_pdf(client: FourMinds, path: str, fields: dict[str, str], file_field: str, pdf: Path) -> dict:
    boundary, body = multipart(fields, file_field, pdf.name, pdf.read_bytes())
    _, raw = client.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=180,
    )
    payload = unwrap(json.loads(raw.decode()))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected upload response: {raw[:400]!r}")
    return payload


def create_dataset(client: FourMinds, pdf: Path) -> dict:
    description = "Pojistne podminky vozidel, va-124-soubor-vozidla.pdf"
    attempts = [
        ("/api/v1/user/dataset", "files", {"dataset_name": DATASET_NAME, "description": description}),
        ("/api/v1/user/dataset", "file", {"name": DATASET_NAME, "description": description}),
        ("/api/v1/user/dataset", "files", {"name": DATASET_NAME, "description": description}),
    ]
    errors: list[str] = []
    for endpoint, field, fields in attempts:
        try:
            return post_pdf(client, endpoint, fields, field, pdf)
        except RuntimeError as exc:
            errors.append(str(exc))
            if not any(code in str(exc) for code in ("400", "422")):
                raise
    raise RuntimeError("Could not upload the PDF:\n" + "\n".join(errors))


def dataset_has_files(dataset: dict) -> bool:
    if int(dataset.get("file_count") or 0) > 0:
        return True
    return int(dataset.get("size") or dataset.get("total_bytes") or 0) > 0


def ensure_dataset(client: FourMinds, pdf: Path) -> dict:
    existing = find_named(as_items(client.get_json("/api/v1/user/dataset")), DATASET_NAME)
    if existing and dataset_has_files(existing):
        print(f"Reusing dataset {existing.get('id')} ({item_name(existing)})")
        return existing
    if existing:
        print(f"Uploading {pdf.name} into dataset {existing.get('id')}")
        post_pdf(
            client,
            "/api/v1/user/dataset/upload",
            {"dataset_id": str(entity_id(existing))},
            "files",
            pdf,
        )
        return existing
    print(f"Uploading {pdf.name}")
    created = create_dataset(client, pdf)
    print(f"Created dataset {created.get('id')}")
    return created


def create_model(client: FourMinds, dataset_id: int | str) -> dict:
    description = "4minds model over va-124-soubor-vozidla.pdf"
    bodies = [
        {"model_name": MODEL_NAME, "dataset_id": dataset_id, "description": description},
        {"name": MODEL_NAME, "dataset_id": dataset_id, "description": description},
        {"model_name": MODEL_NAME, "dataset_id": str(dataset_id), "description": description},
    ]
    errors: list[str] = []
    for body in bodies:
        try:
            _, raw = client.post_json("/api/v1/user/model", body)
            payload = unwrap(json.loads(raw.decode()))
            if isinstance(payload, dict):
                return payload
        except RuntimeError as exc:
            errors.append(str(exc))
            if not any(code in str(exc) for code in ("400", "422")):
                raise
    raise RuntimeError("Could not create the 4minds model:\n" + "\n".join(errors))


def get_model(client: FourMinds, model_id: int | str) -> dict:
    payload = unwrap(client.get_json(f"/api/v1/user/model/{model_id}"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected model payload for {model_id}")
    return payload


def wait_until_ready(client: FourMinds, model_id: int | str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last: tuple | None = None
    while time.time() < deadline:
        model = get_model(client, model_id)
        snapshot = (model.get("status"), model.get("graph_status"), model.get("graph_build_progress"))
        if snapshot != last:
            print(
                f"Model {model_id}: status={snapshot[0]} graph={snapshot[1]} progress={snapshot[2]}"
            )
            last = snapshot
        if is_failed(model):
            raise RuntimeError(f"Model {model_id} failed while preparing: {snapshot}")
        if is_ready(model):
            return model
        time.sleep(15)
    raise RuntimeError(f"Model {model_id} was not ready after {timeout_s} seconds")


def dataset_contains_file(client: FourMinds, dataset_id: int | str, filename: str) -> bool:
    payload = unwrap(client.get_json(f"/api/v1/user/dataset/{dataset_id}"))
    if not isinstance(payload, dict):
        return False
    target = filename.casefold()
    files = payload.get("files") or []
    return any(str(item.get("name", "")).casefold() == target for item in files if isinstance(item, dict))


def model_for_document(client: FourMinds, pdf: Path) -> dict | None:
    models = as_items(client.get_json("/api/v1/user/model"))
    named = find_named(models, MODEL_NAME)
    datasets = as_items(client.get_json("/api/v1/user/dataset"))
    matching_ids = {
        str(dataset.get("id"))
        for dataset in datasets
        if dataset.get("id") is not None and dataset_contains_file(client, dataset["id"], pdf.name)
    }
    if named and (not matching_ids or str(named.get("dataset_id")) in matching_ids):
        return named
    attached = [model for model in models if str(model.get("dataset_id")) in matching_ids]
    ready = [model for model in attached if is_ready(model)]
    if ready:
        return ready[0]
    return attached[0] if attached else None


def ensure_model(client: FourMinds, pdf: Path, model_id: str | None, timeout_s: int) -> tuple[int | str, str]:
    if model_id:
        print(f"Using model {model_id}")
        model = wait_until_ready(client, model_id, timeout_s)
        chosen = int(model_id) if str(model_id).isdigit() else model_id
        return chosen, item_name(model) or str(chosen)

    existing = model_for_document(client, pdf)
    if existing:
        chosen = entity_id(existing)
        print(f"Reusing model {chosen} ({item_name(existing)}) already trained on {pdf.name}")
        model = wait_until_ready(client, chosen, timeout_s)
        return chosen, item_name(model) or item_name(existing)

    dataset = ensure_dataset(client, pdf)
    dataset_id = entity_id(dataset)
    print(f"Creating model {MODEL_NAME} on dataset {dataset_id}")
    created = create_model(client, dataset_id)
    chosen = entity_id(created)
    model = wait_until_ready(client, chosen, timeout_s)
    return chosen, item_name(model) or MODEL_NAME


def parse_sse(text: str) -> str:
    messages: list[str] = []
    tokens: list[str] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            tokens.append(data)
            continue
        if not isinstance(obj, dict):
            continue
        kind = str(obj.get("type") or "")
        if kind in {"reasoning", "context", "status"}:
            continue
        content = obj.get("content") or obj.get("response") or obj.get("answer") or obj.get("text")
        if isinstance(content, dict):
            content = content.get("content") or content.get("text")
        if kind in {"token", "delta"} and isinstance(content, str):
            tokens.append(content)
            continue
        if isinstance(content, str) and content.strip() and kind in {"message", "response", ""}:
            messages.append(content)
            continue
        delta = obj.get("token") or obj.get("delta")
        if isinstance(delta, dict):
            delta = delta.get("content") or delta.get("text")
        if isinstance(delta, str) and delta:
            tokens.append(delta)
    if len(messages) > 1 and all(messages[i + 1].startswith(messages[i]) for i in range(len(messages) - 1)):
        return messages[-1].strip()
    if messages:
        return "".join(messages).strip()
    return "".join(tokens).strip()


def parse_inference(content_type: str, raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type.lower() or text.lstrip().startswith("data:"):
        answer = parse_sse(text)
        if answer:
            return answer
    payload = unwrap(json.loads(text))
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("response", "answer", "message", "content", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"].strip()
    raise RuntimeError(f"Unrecognized inference response: {text[:500]}")


def ask(
    client: FourMinds,
    model_id: int | str,
    question: str,
    answers: list[str],
    *,
    ocr: bool = False,
) -> str:
    query = build_ocr_query(question) if ocr else build_query(question)
    assert_ground_truth_stays_local(query, answers)
    content_type, raw = client.post_json(
        "/api/v1/user/inference",
        {
            "query": query,
            "model_id": int(model_id) if str(model_id).isdigit() else model_id,
            "temperature": 0,
            "enable_web_search": False,
            "max_tokens": 800,
            "response_length_preference": "detailed",
            "thread_id": str(uuid.uuid4()),
        },
        timeout=300,
    )
    answer = parse_inference(content_type, raw).strip()
    if not answer:
        raise RuntimeError("4MINDS returned an empty answer")
    return answer


def evaluate(pairs: list[tuple[str, str]], predictions: list[str], *, ocr: bool = False) -> dict:
    items = []
    for (question, gold), prediction in zip(pairs, predictions):
        scored = score_ocr_pair(prediction, gold) if ocr else score_pair(prediction, gold)
        items.append(
            {
                "question": question,
                "ground_truth": gold,
                "model_answer": prediction,
                **scored,
            }
        )
    passed = sum(1 for item in items if item["passed"])
    mean_f1 = sum(float(item["lexical_f1"]) for item in items) / len(items)
    report = {
        "question_count": len(items),
        "passed": passed,
        "accuracy": round(passed / len(items), 4),
        "mean_lexical_f1": round(mean_f1, 4),
        "items": items,
    }
    if ocr:
        report["mean_number_recall"] = round(
            sum(float(item["number_recall"]) for item in items) / len(items),
            4,
        )
    return report


def print_report(report: dict) -> None:
    for index, item in enumerate(report["items"], start=1):
        mark = "PASS" if item["passed"] else "FAIL"
        extra = ""
        if "number_recall" in item:
            extra = f" numbers={item['number_recall']}"
            if item.get("missing_numbers"):
                extra += f" missing={','.join(item['missing_numbers'])}"
        print(
            f"\n[{index}/{report['question_count']}] {mark} "
            f"decision={item['decision']} match={item['decision_match']} f1={item['lexical_f1']}{extra}"
        )
        print(f"Q: {item['question']}")
        print(f"4minds: {item['model_answer']}")
        print(f"ground truth: {item['ground_truth']}")
    summary = (
        f"\nDecision accuracy: {report['passed']}/{report['question_count']} "
        f"({report['accuracy']:.0%}), mean lexical F1 {report['mean_lexical_f1']:.2f}"
    )
    if "mean_number_recall" in report:
        summary += f", mean number recall {report['mean_number_recall']:.2f}"
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a document with the 4minds model and score the answers.")
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=os.environ.get("4MINDS_MODEL_ID"))
    parser.add_argument("--model-name", default=None, help="Reuse a ready model with this name, for example model-ocr")
    parser.add_argument("--ocr", action="store_true", help="Score receipt amounts against questions_ocr.txt")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds to wait for the model to become ready")
    return parser.parse_args()


def main() -> None:
    load_env(ROOT / ".env")
    args = parse_args()
    api_key = os.environ.get("4MINDS_API_KEY") or os.environ.get("FOURMINDS_API_KEY")
    if not api_key:
        raise SystemExit("Set 4MINDS_API_KEY in the environment or in .env")
    if args.ocr:
        if args.questions == DEFAULT_QUESTIONS:
            args.questions = ROOT / "questions_ocr.txt"
        if args.output == DEFAULT_OUTPUT:
            args.output = ROOT / "eval_ocr_results.json"
        if not args.model_name and not args.model_id:
            args.model_name = "model-ocr"

    if not args.ocr and not args.document.is_file():
        raise SystemExit(f"Document not found: {args.document}")
    if not args.questions.is_file():
        raise SystemExit(f"Questions not found: {args.questions}")

    pairs = load_qa(args.questions)
    questions = [question for question, _ in pairs]
    answers = [answer for _, answer in pairs]
    print(f"Loaded {len(pairs)} questions. Ground truth stays local and is not sent to the model.")

    client = FourMinds(api_key)
    if args.model_name and not args.model_id:
        named = find_named(as_items(client.get_json("/api/v1/user/model")), args.model_name)
        if named is None:
            raise SystemExit(f"No 4MINDS model named {args.model_name}")
        args.model_id = str(entity_id(named))
        print(f"Found model {args.model_name} as id {args.model_id}")
    model_id, model_name = ensure_model(client, args.document, args.model_id, args.timeout)

    predictions = []
    for index, question in enumerate(questions, start=1):
        print(f"Asking {index}/{len(questions)}")
        predictions.append(ask(client, model_id, question, answers, ocr=args.ocr))

    report = evaluate(pairs, predictions, ocr=args.ocr)
    report["model_id"] = model_id
    report["model_name"] = model_name
    report["document"] = str(args.document)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
