import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from lab.metrics import evaluate, hop_type
from lab.prepare import fetch_politely, page_offset_report, write_jsonl
from lab.report import report
from lab.run import read_jsonl, run_experiment
from lab.stages import PageIndex, bm25, neighbors

PAGES = [
    {"document": "a.pdf", "page": 1, "text": "Annual revenue of Acme was 12 million dollars"},
    {"document": "a.pdf", "page": 2, "text": "The chief executive of Acme is Jane Roe"},
    {"document": "a.pdf", "page": 3, "text": "Unrelated notes about office furniture"},
    {"document": "b.pdf", "page": 1, "text": "Jane Roe was born in Lisbon"},
    {"document": "b.pdf", "page": 2, "text": ""},
]
QUESTIONS = [
    {"id": "dev/0", "question": "What was the revenue of Acme?", "evidence": [{"document": "a.pdf", "page": 1}]},
    {
        "id": "dev/1",
        "question": "Where was the chief executive of Acme born?",
        "evidence": [{"document": "a.pdf", "page": 2}, {"document": "b.pdf", "page": 1}],
    },
    {
        "id": "dev/2",
        "question": "Acme revenue and furniture",
        "evidence": [{"document": "a.pdf", "page": 1}, {"document": "a.pdf", "page": 3}],
    },
]


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    write_jsonl(data / "questions" / "dev.jsonl", QUESTIONS)
    write_jsonl(data / "pages" / "pdf_text.jsonl", PAGES)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "name": "bm25",
                "pages": "pdf_text",
                "pipeline": [{"stage": "bm25", "top_k": 5}],
                "eval_k": [1, 5],
            }
        )
    )
    return data, config


def test_bm25_ranks_the_matching_page_first(tmp_path):
    index = PageIndex(tmp_path / "i.sqlite", PAGES)
    assert bm25("Where was Jane Roe born in Lisbon?", [], index)[0] == ("b.pdf", 1)


def test_bm25_survives_punctuation_and_stopword_only_queries(tmp_path):
    index = PageIndex(tmp_path / "i.sqlite", PAGES)
    assert bm25('Acme\'s "revenue" (AND) OR NOT*', [], index)[0] == ("a.pdf", 1)
    assert bm25("what is the", [], index) == []


def test_interrupted_index_build_is_not_reused(tmp_path):
    path = tmp_path / "i.sqlite"

    def failing_pages():
        yield PAGES[0]
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        PageIndex(path, failing_pages())
    assert not path.exists()
    assert bm25("Jane Roe Lisbon", [], PageIndex(path, PAGES))[0] == ("b.pdf", 1)


def test_neighbors_stay_inside_the_document(tmp_path):
    index = PageIndex(tmp_path / "i.sqlite", PAGES)
    assert neighbors("q", [("a.pdf", 1)], index) == [("a.pdf", 1), ("a.pdf", 2)]
    assert neighbors("q", [("a.pdf", 2)], index) == [("a.pdf", 2), ("a.pdf", 3), ("a.pdf", 1)]


def test_metrics_and_hop_types():
    assert [hop_type(q) for q in QUESTIONS] == ["single", "cross_doc", "cross_page"]
    result = evaluate([(QUESTIONS[1], [("a.pdf", 2), ("x.pdf", 1)])], [1, 2])
    assert result["overall"]["page_recall@1"] == 50.0
    assert result["overall"]["all_evidence@2"] == 0.0
    assert result["cross_doc"]["n"] == 1
    assert result["single"]["n"] == 0


def test_questions_without_evidence_are_skipped():
    no_gold = {"id": "t", "question": "q", "evidence": []}
    assert evaluate([(no_gold, [])], [1])["overall"] == {"n": 0}


def test_run_logs_once_and_reuses_identical_runs(workspace):
    data, config = workspace
    first = run_experiment(config, "dev", data)
    second = run_experiment(config, "dev", data)
    assert (first["cached"], second["cached"]) == (False, True)
    assert first["metrics"]["overall"]["n"] == 3
    assert len(read_jsonl(data / "results" / "ledger.jsonl")) == 1
    assert "bm25" in report(data, "dev", 5)


def test_changed_pages_invalidate_the_cached_result(workspace):
    data, config = workspace
    run_experiment(config, "dev", data)
    write_jsonl(data / "pages" / "pdf_text.jsonl", PAGES[:3])
    assert run_experiment(config, "dev", data)["cached"] is False


def test_report_hides_runs_on_older_pages(workspace):
    data, config = workspace
    run_experiment(config, "dev", data)
    write_jsonl(data / "pages" / "pdf_text.jsonl", PAGES[:3])
    assert "bm25" not in report(data, "dev", 5).split("name", 1)[1]
    run_experiment(config, "dev", data)
    assert "bm25" in report(data, "dev", 5).split("name", 1)[1]


def _blocked(url):
    raise HTTPError(url, 429, "Too Many Requests", None, None)


def test_downloader_stops_when_blocked_and_resumes(tmp_path):
    names = [f"{i}.pdf" for i in range(10)]
    urls = {name: name for name in names}
    counts = fetch_politely(names, urls, tmp_path, _blocked, max_blocked=3, sleep=lambda _: None)
    assert counts["failed"] == 3 and counts["blocked_stop"] == 1
    counts = fetch_politely(names, urls, tmp_path, lambda u: b"pdf", sleep=lambda _: None)
    assert counts["downloaded"] == 10
    counts = fetch_politely(names, urls, tmp_path, _blocked, sleep=lambda _: None)
    assert counts["skipped"] == 10 and counts["failed"] == 0


def test_runs_record_a_profile_of_each_function(workspace):
    data, config = workspace
    profile = run_experiment(config, "dev", data)["profile"]
    assert profile["run_pipeline"]["calls"] == 3
    assert profile["bm25"]["calls"] == 3
    assert profile["PageIndex.bm25"]["calls"] == 3
    assert {"calls", "total_ms", "mean_ms", "p95_ms", "max_ms"} <= set(profile["bm25"])


def test_force_reruns_and_report_keeps_one_row(workspace):
    data, config = workspace
    run_experiment(config, "dev", data)
    assert run_experiment(config, "dev", data, force=True)["cached"] is False
    assert report(data, "dev", 5).count("bm25") == 1


def test_test_split_is_refused(workspace):
    data, config = workspace
    with pytest.raises(ValueError):
        run_experiment(config, "test", data)


def test_page_offset_report_detects_zero_based_gold():
    pages = [
        {"document": "a.pdf", "page": 1, "text": "nothing"},
        {"document": "a.pdf", "page": 2, "text": "answer lives here"},
    ]
    question = {"answer_variants": [["answer"]], "evidence": [{"document": "a.pdf", "page": 1}]}
    assert page_offset_report([question], pages) == {-1: 0.0, 0: 0.0, 1: 100.0}


def test_parse_answer_handles_model_output_shapes():
    from lab.read import parse_answer

    assert parse_answer('{"answer": ["Senior Scientist"]}') == ["Senior Scientist"]
    assert parse_answer('<think>hmm {"x":1}</think>\n```json\n{"answer": "12"}\n```') == ["12"]
    assert parse_answer("plain text") == ["plain text"]
    assert parse_answer("") == []


def test_reader_scores_with_madqa_metric_and_profiles(tmp_path):
    from lab.profiling import reset, snapshot
    from lab.read import run_reader, score

    questions = [
        {"id": "a", "question": "Who?", "evidence": [{"document": "a.pdf", "page": 1}], "answer_variants": [["Jane Roe"]]},
        {"id": "b", "question": "Where?", "evidence": [{"document": "a.pdf", "page": 1}], "answer_variants": [["Lisbon"]]},
    ]
    replies = {"Who?": "Jane Roe", "Where?": "Tokyo"}

    def fake(content):
        answer = next(v for k, v in replies.items() if k in content[-1]["text"])
        return {"choices": [{"message": {"content": json.dumps({"answer": [answer]})}}]}

    reset()
    outputs = run_reader(questions, tmp_path, "text", {("a.pdf", 1): "page text"}, fake, workers=2)
    assert [o["id"] for o in outputs] == ["a", "b"]
    assert "page text" not in outputs[0]["answer"]
    metrics = score(questions, outputs)
    assert metrics["overall"]["accuracy"] == 50.0
    assert metrics["single"]["n"] == 2
    assert snapshot()["question_total"]["calls"] == 2


def test_render_jpeg_produces_an_image_from_a_real_pdf(tmp_path):
    import base64

    import pypdfium2

    from lab.read import render_jpeg

    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(300, 400)
    pdf.save(tmp_path / "a.pdf")
    pdf.close()
    assert base64.b64decode(render_jpeg(tmp_path / "a.pdf", 1)).startswith(b"\xff\xd8")


def test_parse_reply_maps_labels_to_pages_and_drops_invented_ones():
    from lab.answer import parse_reply

    pages = [("a.pdf", 4), ("b.pdf", 9)]
    answer, citations = parse_reply('```json\n{"answer": ["x"], "pages": [2, 2, 7, "1"]}\n```', pages)
    assert answer == ["x"]
    assert citations == [{"file": "b.pdf", "page": 9}]
    assert parse_reply("not json", pages) == (["not json"], [])


def test_end_to_end_answers_and_scores_with_citations(tmp_path):
    import pypdfium2

    from lab.answer import answer_questions, score

    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(100, 100)
    pdf.save(tmp_path / "pdfs_a.pdf")
    pdf.close()
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs_a.pdf").rename(tmp_path / "pdfs" / "a.pdf")
    questions = [
        {"id": "q", "question": "Who?", "answer_variants": [["Jane"]], "evidence": [{"document": "a.pdf", "page": 1}]},
        {"id": "r", "question": "Nothing?", "answer_variants": [["No"]], "evidence": [{"document": "a.pdf", "page": 1}]},
    ]
    reply = {"choices": [{"message": {"content": '{"answer": ["Jane"], "pages": [1]}'}}]}
    retrieve = lambda text: [("a.pdf", 1)] if text == "Who?" else []
    outputs = answer_questions(questions, retrieve, lambda content: reply, tmp_path, 5, workers=2)
    assert outputs[0]["citations"] == [{"file": "a.pdf", "page": 1}]
    assert outputs[1]["answer"] == [] and outputs[1]["iterations"] == 1
    metrics = score(questions, outputs)["overall"]
    assert metrics["accuracy"] == 50.0
    assert metrics["page_f1"] == 50.0


def test_order_by_scores_only_reorders_the_scored_head():
    from lab.stages import order_by_scores

    pages = [("a", 1), ("a", 2), ("a", 3), ("a", 4)]
    assert order_by_scores(pages, [0.1, 0.9, 0.5], 3) == [("a", 2), ("a", 3), ("a", 1), ("a", 4)]
    assert order_by_scores(pages[:2], [0.5, 0.5], 3) == pages[:2]


def test_index_text_reads_page_text(tmp_path):
    write_jsonl(tmp_path / "pages.jsonl", PAGES)
    index = PageIndex(tmp_path / "index" / "i.sqlite", PAGES)
    index.pages_path = tmp_path / "pages.jsonl"
    assert "Jane Roe" in index.text(("a.pdf", 2))


def test_reader_caps_images_at_the_server_limit():
    from lab.read import MAX_IMAGES, build_content

    question = {"question": "q", "evidence": [{"document": "a.pdf", "page": i} for i in range(1, MAX_IMAGES + 5)]}
    content = build_content(question, Path("unused"), "text", {})
    assert content[-1]["text"].count("[a.pdf page") == MAX_IMAGES


def test_parsers_ignore_reasoning_with_only_a_closing_think_tag():
    from lab.answer import parse_reply
    from lab.read import parse_answer

    text = 'Look at {this} and {"answer": ["draft"]}.\n</think>\n{"answer": ["final"], "pages": [1]}'
    assert parse_answer(text) == ["final"]
    assert parse_reply(text, [("a.pdf", 2)]) == (["final"], [{"file": "a.pdf", "page": 2}])


def test_select_questions_uses_the_id_set_and_rejects_unknown_ids(tmp_path):
    from lab.run import select_questions

    questions = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    ids = tmp_path / "set.json"
    ids.write_text(json.dumps({"ids": {"x": ["a"], "y": ["c"]}}))
    assert [q["id"] for q in select_questions(questions, ids)] == ["a", "c"]
    assert select_questions(questions, None) == questions
    ids.write_text(json.dumps({"ids": {"x": ["z"]}}))
    with pytest.raises(ValueError):
        select_questions(questions, ids)


def test_hard20_set_has_twenty_real_dev_ids():
    from lab.run import select_questions

    ids = Path("experiments/hard20.json")
    groups = json.loads(ids.read_text())["ids"]
    wanted = [i for group in groups.values() for i in group]
    assert len(wanted) == len(set(wanted)) == 20
    assert len(select_questions([{"id": i} for i in wanted], ids)) == 20


def _indexed(tmp_path):
    write_jsonl(tmp_path / "pages.jsonl", PAGES)
    index = PageIndex(tmp_path / "index" / "i.sqlite", PAGES)
    index.pages_path = tmp_path / "pages.jsonl"
    return index


def test_expansion_terms_skip_question_words_and_favour_names():
    from lab.stages import expansion_terms

    terms = expansion_terms("Jane Roe met the tall gardener", "Who met the gardener?", {"tall": 1}, 100, 3)
    assert "met" not in terms and "gardener" not in terms
    assert terms[0] in {"jane", "roe"}


def test_expand_follows_a_name_to_a_linked_page(tmp_path):
    from lab.stages import expand

    index = _indexed(tmp_path)
    # The CEO page names Jane Roe, which leads to the Lisbon page in another document.
    question = "Where was the chief executive of Acme born?"
    result = expand(question, [("a.pdf", 2)], index, seeds=1, terms=5, per_seed=5)
    assert result[0] == ("a.pdf", 2)
    assert ("b.pdf", 1) in result
    assert len(result) == len(set(result))


def test_name_phrase_finds_multi_word_names_only():
    from lab.stages import NAME_PHRASE

    assert NAME_PHRASE.findall("the chief was Jane Roe, who met John Smith today") == ["Jane Roe", "John Smith"]
    assert NAME_PHRASE.findall("nothing here") == []


def test_add_context_gives_each_page_its_own_documents_opening():
    from lab.contextualize import add_context

    pages = [
        {"document": "b.pdf", "page": 2, "text": "Exhibit A"},
        {"document": "a.pdf", "page": 1, "text": "Other agreement"},
        {"document": "b.pdf", "page": 1, "text": "Confidentiality Agreement"},
    ]
    result = {(p["document"], p["page"]): p["text"] for p in add_context(pages, 20)}
    assert result[("b.pdf", 2)].startswith("Confidentiality Agre")
    assert result[("b.pdf", 2)].endswith("Exhibit A")
    assert "Other" not in result[("b.pdf", 2)]
    assert len(result) == 3


def test_index_text_can_read_another_text_source(tmp_path):
    write_jsonl(tmp_path / "enriched.jsonl", [{"document": "a.pdf", "page": 1, "text": "with context"}])
    write_jsonl(tmp_path / "plain.jsonl", [{"document": "a.pdf", "page": 1, "text": "plain"}])
    index = PageIndex(tmp_path / "index" / "i.sqlite", [{"document": "a.pdf", "page": 1, "text": "with context"}])
    index.pages_path = tmp_path / "enriched.jsonl"
    assert index.text(("a.pdf", 1)) == "with context"
    assert index.text(("a.pdf", 1), "plain") == "plain"


def test_parse_judgment_reads_the_last_verdict_after_reasoning():
    from lab.judge import parse_judgment

    text = 'Step 1 {"judgment": "correct"} was a draft. {not json}\n{"judgment": "partial", "main_issue": "x"}'
    assert parse_judgment(text) == "partial"
    assert parse_judgment("no verdict here") is None
    assert parse_judgment('{"judgment": "maybe"}') is None


def test_judge_follows_madqa_rules_and_caches_verdicts(tmp_path):
    from lab.judge import Judge, judged_score

    calls = []

    def fake(prompt):
        calls.append(prompt)
        return '{"judgment": "correct"}'

    judge = Judge(fake, tmp_path / "cache.jsonl")
    question = {"question": "How much?", "answer_variants": [["$30 million"]]}
    assert judged_score(judge, question, []) == (0.0, 0.0, False)
    assert judged_score(judge, question, ["$30 million"])[0] == 1.0
    assert calls == []
    assert judged_score(judge, question, ["$30,000,000"])[0] == 1.0
    assert judged_score(judge, question, ["$30,000,000"])[0] == 1.0
    assert len(calls) == 1
    reloaded = Judge(fake, tmp_path / "cache.jsonl")
    assert judged_score(reloaded, question, ["$30,000,000"])[0] == 1.0
    assert len(calls) == 1


def test_judge_failure_is_reported_not_silently_counted(tmp_path):
    from lab.judge import Judge, judged_score

    judge = Judge(lambda prompt: "no json", tmp_path / "cache.jsonl", retries=1)
    question = {"question": "q", "answer_variants": [["yes"]]}
    assert judged_score(judge, question, ["nope"])[2] is True
    assert judge.errors == 1


def test_to_messages_content_converts_images_and_keeps_order():
    from lab.commandcode import to_messages_content

    blocks = to_messages_content(
        [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            {"type": "text", "text": "q"},
        ]
    )
    assert blocks[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}}
    assert blocks[1] == {"type": "text", "text": "q"}


def test_with_neighbors_keeps_given_pages_first_and_only_adds_real_pages():
    from lab.read import with_neighbors

    known = {("a", 1), ("a", 2), ("a", 3), ("b", 1)}
    assert with_neighbors([("a", 2)], 1, known) == [("a", 2), ("a", 3), ("a", 1)]
    assert with_neighbors([("b", 1)], 1, known) == [("b", 1)]
    assert with_neighbors([("a", 1), ("a", 2)], 1, known) == [("a", 1), ("a", 2), ("a", 3)]
    assert with_neighbors([("a", 2)], 0, known) == [("a", 2)]


def test_collect_stream_rebuilds_text_and_usage():
    from lab.read import collect_stream

    lines = [
        b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n',
        b"\n",
        b'data: {"choices": [{"delta": {"content": "lo"}}]}\n',
        b'data: {"choices": [], "usage": {"completion_tokens": 2}}\n',
        b"data: [DONE]\n",
    ]
    reply = collect_stream(lines)
    assert reply["choices"][0]["message"]["content"] == "Hello"
    assert reply["usage"]["completion_tokens"] == 2


def test_render_jpeg_respects_max_side(tmp_path):
    import base64
    import io

    import pypdfium2
    from PIL import Image

    from lab.read import render_jpeg

    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(300, 400)
    pdf.save(tmp_path / "a.pdf")
    pdf.close()
    image = Image.open(io.BytesIO(base64.b64decode(render_jpeg(tmp_path / "a.pdf", 1, 2000))))
    assert max(image.size) == 2000


def test_v3_rules_keep_units_and_are_available():
    from lab.read import RULES

    assert "keep units" in RULES["v3"]
    assert RULES["v1"] == ""


def test_ocr_merge_keeps_stamps_and_leaves_full_pages_alone():
    from lab.ocr import merge

    pages = [
        {"document": "a", "page": 1, "text": ""},
        {"document": "a", "page": 2, "text": "CONFIDENTIAL STAMP ON A SCAN"},
        {"document": "a", "page": 3, "text": "x" * 500},
        {"document": "a", "page": 4, "text": "tiny"},
    ]
    ocr = {("a", 1): "scan one", ("a", 2): "scan two", ("a", 3): "ignored", ("a", 4): "scan four"}
    result = {p["page"]: p["text"] for p in merge(pages, ocr, 200)}
    assert result[1] == "scan one"
    assert result[2] == "CONFIDENTIAL STAMP ON A SCAN\nscan two"
    assert result[3] == "x" * 500
    assert result[4] == "scan four"
    assert merge(pages[:1], {}, 200)[0]["text"] == ""


def test_answer_labels_name_the_file_only_when_asked(tmp_path):
    import pypdfium2

    from lab.answer import build_content

    (tmp_path / "pdfs").mkdir()
    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(100, 100)
    pdf.save(tmp_path / "pdfs" / "a.pdf")
    pdf.close()
    plain = build_content("q", [("a.pdf", 1)], tmp_path)
    labelled = build_content("q", [("a.pdf", 1)], tmp_path, doc_labels=True)
    assert plain[0]["text"] == "Page 1:"
    assert labelled[0]["text"] == "Page 1 (file a.pdf, its page 1):"


def test_one_failed_request_scores_empty_and_the_run_continues(tmp_path):
    import urllib.error

    import pypdfium2

    from lab.answer import answer_questions

    (tmp_path / "pdfs").mkdir()
    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(100, 100)
    pdf.save(tmp_path / "pdfs" / "a.pdf")
    pdf.close()
    questions = [{"id": "a", "question": "bad"}, {"id": "b", "question": "good"}]

    def call(content):
        if "bad" in content[-1]["text"]:
            raise urllib.error.HTTPError("u", 400, "Bad Request", None, None)
        return {"choices": [{"message": {"content": '{"answer": ["x"], "pages": [1]}'}}]}

    outputs = answer_questions(questions, lambda q: [("a.pdf", 1)], call, tmp_path, 5, workers=1)
    assert outputs[0]["answer"] == []
    assert outputs[1]["answer"] == ["x"]


def test_validate_reports_every_submission_problem():
    from lab.submit import to_submission_row, validate

    questions = [{"id": "a", "question": "qa"}, {"id": "b", "question": "qb"}]
    good = {"id": "a", "question": "qa", "answer": ["x"], "citations": [{"file": "f.pdf", "page": 1}], "iterations": 2, "usage": {}}
    row = to_submission_row(good)
    assert set(row) == {"question", "answer", "citations", "iterations", "id"}
    known = {("f.pdf", 1)}
    assert validate([row, {**row, "id": "b", "question": "qb"}], questions, known) == []
    problems = validate([{**row, "answer": [], "citations": [{"file": "z", "page": 9}]}], questions, known)
    joined = " ".join(problems)
    assert "1 questions unanswered" in joined and "empty answer" in joined and "unknown page" in joined


def test_plan_falls_back_to_the_question_on_bad_output():
    from lab.agent import plan_searches

    assert plan_searches("q", lambda p: '{"searches": ["a", "b", "a"]}') == ["a", "b"]
    assert plan_searches("q", lambda p: "thinking {x}</think>{\"searches\": [\"c\"]}") == ["c"]
    assert plan_searches("q", lambda p: "nonsense") == ["q"]
    assert plan_searches("q", lambda p: '{"searches": []}') == ["q"]

    def down(prompt):
        raise OSError("server down")

    assert plan_searches("q", down) == ["q"]


def test_select_pages_shares_room_between_searches_then_adds_neighbours():
    from lab.agent import AgentSettings, select_pages

    known = {("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2)}
    lists = [[("a", 1), ("a", 2)], [("b", 1)]]
    settings = AgentSettings(pages_per_search=2, window=1, max_images=4)
    assert select_pages(lists, settings, known) == [("a", 1), ("b", 1), ("a", 2), ("b", 2)]


def _agent_setup(tmp_path):
    import pypdfium2

    (tmp_path / "pdfs").mkdir()
    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(100, 100)
    pdf.new_page(100, 100)
    pdf.save(tmp_path / "pdfs" / "a.pdf")
    pdf.close()
    return {("a.pdf", 1), ("a.pdf", 2)}


def test_agent_answers_simple_question_in_one_round(tmp_path):
    from lab.agent import AgentSettings, run_agent

    known = _agent_setup(tmp_path)
    reply = '{"answer": ["x"], "pages": [1], "sufficient": true, "next_search": ""}'
    result = run_agent(
        "q", lambda s: [("a.pdf", 1)], lambda p: '{"searches": ["q"]}', lambda c: reply, tmp_path, known, AgentSettings(window=0)
    )
    assert (result.answer, result.iterations) == (["x"], 1)
    assert result.citations == [{"file": "a.pdf", "page": 1}]


def test_agent_searches_again_until_sufficient_and_respects_the_round_limit(tmp_path):
    from lab.agent import AgentSettings, run_agent

    known = _agent_setup(tmp_path)
    searched = []

    def search(query):
        searched.append(query)
        return [("a.pdf", 2)] if query == "more" else [("a.pdf", 1)]

    replies = iter(
        [
            '{"answer": ["draft"], "pages": [1], "sufficient": false, "next_search": "more"}',
            '{"answer": ["final"], "pages": [1], "sufficient": true, "next_search": ""}',
        ]
    )
    result = run_agent("q", search, lambda p: '{"searches": ["q"]}', lambda c: next(replies), tmp_path, known, AgentSettings(window=0))
    assert result.answer == ["final"] and result.iterations == 2 and searched == ["q", "more"]
    assert result.citations == [{"file": "a.pdf", "page": 2}]

    # Asking for new searches forever must still stop, and keep the latest answer.
    count = iter(range(100))

    def endless(content):
        return json.dumps({"answer": ["a"], "pages": [1], "sufficient": False, "next_search": f"s{next(count)}"})

    capped = run_agent("q", search, lambda p: '{"searches": ["q"]}', endless, tmp_path, known, AgentSettings(window=0, max_rounds=3))
    assert capped.iterations == 3 and capped.answer == ["a"]


def test_agent_stops_if_it_repeats_a_search(tmp_path):
    from lab.agent import AgentSettings, run_agent

    known = _agent_setup(tmp_path)
    reply = '{"answer": ["a"], "pages": [1], "sufficient": false, "next_search": "q"}'
    result = run_agent("q", lambda s: [("a.pdf", 1)], lambda p: '{"searches": ["q"]}', lambda c: reply, tmp_path, known, AgentSettings(window=0))
    assert result.iterations == 1


def test_agent_carries_confirmed_facts_to_the_next_round(tmp_path):
    from lab.agent import AgentSettings, run_agent

    known = _agent_setup(tmp_path)
    seen = []
    replies = iter(
        [
            '{"answer": ["draft"], "pages": [1], "confirmed": ["Ann is the CEO"], "missing": ["birth"], "sufficient": false, "next_search": "Ann birth"}',
            '{"answer": ["Lisbon"], "pages": [2], "confirmed": [], "missing": [], "sufficient": true, "next_search": ""}',
        ]
    )

    def vision(content):
        seen.append(content[-1]["text"])
        return next(replies)

    result = run_agent("q", lambda s: [("a.pdf", 1)], lambda p: '{"searches": ["q"]}', vision, tmp_path, known, AgentSettings(window=0))
    assert result.answer == ["Lisbon"]
    assert "Ann is the CEO" not in seen[0]
    assert "Ann is the CEO" in seen[1] and "Ann birth" in seen[1]


def test_llm_selection_orders_chosen_pages_and_ignores_bad_labels():
    from lab.stages import apply_selection, parse_selection

    pages = [("a", 1), ("a", 2), ("a", 3)]
    assert parse_selection('<x>{"pages": [3, 3, 1, 9, "2"]}', 3) == [2, 0]
    assert parse_selection("reasoning</think>{\"pages\": [2]}", 3) == [1]
    assert parse_selection("no json", 3) == []
    assert apply_selection(pages, [2, 0]) == [("a", 3), ("a", 1), ("a", 2)]
    assert apply_selection(pages, []) == pages


def test_agent_search_steps_leave_neighbours_to_the_agent():
    from lab.run_agent import search_steps

    steps = [{"stage": "bm25"}, {"stage": "rerank"}, {"stage": "neighbors"}, {"stage": "llm_select"}]
    assert [s["stage"] for s in search_steps(steps)] == ["bm25", "rerank", "llm_select"]


def test_agent_runner_scores_a_failed_request_as_empty(tmp_path):
    import urllib.error

    from lab.agent import AgentSettings
    from lab.run_agent import answer_one

    known = _agent_setup(tmp_path)

    def broken(content):
        raise urllib.error.HTTPError("u", 400, "Bad", None, None)

    row = answer_one(
        {"id": "a", "question": "q"}, lambda s: [("a.pdf", 1)], lambda p: '{"searches": ["q"]}', broken, tmp_path, known, AgentSettings(window=0)
    )
    assert row["answer"] == [] and row["failed"] is True and row["iterations"] == 1


def test_verify_keeps_draft_unless_there_is_a_real_correction():
    from lab.verify import merge_verdict

    draft = {"id": "a", "answer": ["old"], "citations": [{"file": "f", "page": 9}]}
    pages = [("f", 1), ("f", 2)]
    fixed = merge_verdict(draft, '{"quote": "q", "draft_correct": false, "answer": ["new"], "pages": [2]}', pages)
    assert fixed["answer"] == ["new"] and fixed["changed"] is True
    assert fixed["draft_answer"] == ["old"] and fixed["citations"] == [{"file": "f", "page": 2}]
    same = merge_verdict(draft, '{"quote": "q", "draft_correct": true, "answer": ["new"], "pages": [1]}', pages)
    assert same["answer"] == ["old"] and same["changed"] is False
    assert same["citations"] == [{"file": "f", "page": 9}]
    assert merge_verdict(draft, "not json", pages)["answer"] == ["old"]
    empty = merge_verdict(draft, '{"draft_correct": false, "answer": [], "pages": []}', pages)
    assert empty["answer"] == ["old"] and empty["changed"] is False


def test_verify_survives_a_failed_request(tmp_path):
    import urllib.error

    import pypdfium2

    from lab.verify import verify_questions

    (tmp_path / "pdfs").mkdir()
    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(100, 100)
    pdf.save(tmp_path / "pdfs" / "a.pdf")
    pdf.close()

    def down(content):
        assert "Draft answer" in content[-1]["text"]
        raise urllib.error.URLError("down")

    drafts = [{"id": "a", "question": "q", "answer": ["x"], "citations": []}]
    out = verify_questions(drafts, {"a": [("a.pdf", 1)]}, down, tmp_path, 9, "", None, 1)
    assert out[0]["answer"] == ["x"] and out[0]["changed"] is False


def test_parse_number_handles_printed_formats_and_refuses_ambiguity():
    from lab.calc import parse_number

    assert parse_number("$13,417,108") == 13417108
    assert parse_number("(5,506,557)") == -5506557
    assert parse_number("-3.5") == -3.5
    assert parse_number("$13.65 million") == 13650000
    assert parse_number("2,400 cwt") == 2400
    assert parse_number("3 of 5") is None
    assert parse_number("n/a") is None


def test_parse_date_handles_common_formats():
    from datetime import date

    from lab.calc import parse_date

    assert parse_date("July 24, 2020") == date(2020, 7, 24)
    assert parse_date("24th July 2020") == date(2020, 7, 24)
    assert parse_date("7/24/2020") == date(2020, 7, 24)
    assert parse_date("2020-07-24") == date(2020, 7, 24)
    assert parse_date("sometime") is None


def test_compute_covers_each_operation_and_refuses_when_unsure():
    from lab.calc import compute

    v = lambda *pairs: [{"label": a, "value": b} for a, b in pairs]
    assert compute("sum", v(("a", "$23,000"), ("b", "$73,680"))) == "$96,680"
    assert compute("difference", v(("a", "10"), ("b", "4"))) == "6"
    assert compute("count", v(("a", "x"), ("b", "y"), ("c", "z"))) == "3"
    assert compute("highest", v(("Ann", "5"), ("Bob", "9"))) == "Bob"
    assert compute("lowest", v(("Ann", "5"), ("Bob", "9"))) == "Ann"
    assert compute("earliest", v(("A", "Aug 4, 2020"), ("B", "July 24, 2020"))) == "B"
    assert compute("latest", v(("A", "Aug 4, 2020"), ("B", "July 24, 2020"))) == "A"
    assert compute("years_between", v(("birth", "May 2, 1990"), ("death", "May 1, 2015"))) == "24"
    assert compute("sum", v(("a", "2,400 cwt"), ("b", "100 cwt")), "cwt") == "2,500 cwt"
    assert compute("highest", v(("Ann", "9"), ("Bob", "9"))) is None
    assert compute("sum", v(("a", "unknown"), ("b", "4"))) is None
    assert compute("sum", v(("a", "4"))) is None
    assert compute("none", v(("a", "4"))) is None
    assert compute("__import__('os')", v(("a", "4"))) is None


def test_apply_calc_only_replaces_on_a_real_calculation():
    from lab.calc import apply_calc

    draft = {"id": "a", "answer": ["$1"], "citations": []}
    pages = [("f.pdf", 3), ("f.pdf", 4)]
    text = '{"operation": "sum", "values": [{"label": "a", "value": "$1", "page": 1}, {"label": "b", "value": "$2", "page": 2}], "unit": ""}'
    result = apply_calc(draft, text, pages)
    assert result["answer"] == ["$3"] and result["calc_changed"] is True
    assert result["citations"] == [{"file": "f.pdf", "page": 3}, {"file": "f.pdf", "page": 4}]
    none = apply_calc(draft, '{"operation": "none", "values": [], "unit": ""}', pages)
    assert none["answer"] == ["$1"] and none["calc_changed"] is False
    assert apply_calc(draft, "garbage", pages)["answer"] == ["$1"]


def test_escalation_only_runs_on_disagreement_and_keeps_usable_answers(tmp_path):
    import pypdfium2

    from lab.escalate import disagree, escalate_questions

    assert not disagree(["A, b"], ["a b"])
    assert not disagree(["x", "y"], ["y", "x"])
    assert disagree(["1"], ["2"])
    (tmp_path / "pdfs").mkdir()
    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(100, 100)
    pdf.save(tmp_path / "pdfs" / "a.pdf")
    pdf.close()
    first = [
        {"id": "same", "question": "q", "answer": ["1"], "citations": []},
        {"id": "diff", "question": "q", "answer": ["1"], "citations": []},
    ]
    second = {"same": {"answer": ["1"]}, "diff": {"answer": ["2"]}}
    calls = []

    def call(content):
        calls.append(1)
        return {"choices": [{"message": {"content": 'reasoning</think>{"answer": ["3"], "pages": [1]}'}}]}

    ranked = {"same": [("a.pdf", 1)], "diff": [("a.pdf", 1)]}
    out = escalate_questions(first, second, ranked, call, tmp_path, 9, "", None, 1)
    assert len(calls) == 1
    assert out[0]["answer"] == ["1"] and out[0]["escalated"] is False
    assert out[1]["answer"] == ["3"] and out[1]["before_escalation"] == ["1"]


def test_looks_unanswered_spots_empty_and_not_found_replies():
    from lab.agent import looks_unanswered

    assert looks_unanswered([])
    assert looks_unanswered(["", " "])
    assert looks_unanswered(["Not stated in the document"])
    assert looks_unanswered(["The document does not provide this"])
    assert not looks_unanswered(["Raymond D. Johnson"])
    assert not looks_unanswered(["No"])


def test_empty_answer_triggers_a_different_search_even_when_marked_sufficient(tmp_path):
    from lab.agent import AgentSettings, run_agent

    known = _agent_setup(tmp_path)
    searched = []

    def search(query):
        searched.append(query)
        return [("a.pdf", 2)] if query == "Tedd Druck" else [("a.pdf", 1)]

    def text_model(prompt):
        return '{"search": "Tedd Druck"}' if "different approach" in prompt else '{"searches": ["address of Tedd"]}'

    replies = iter(
        [
            '{"answer": [], "pages": [], "sufficient": true, "next_search": ""}',
            '{"answer": ["685 Strawberry Rd"], "pages": [1], "sufficient": true, "next_search": ""}',
        ]
    )
    result = run_agent("q", search, text_model, lambda c: next(replies), tmp_path, known, AgentSettings(window=0))
    assert searched == ["address of Tedd", "Tedd Druck"]
    assert result.answer == ["685 Strawberry Rd"] and result.iterations == 2
    assert result.citations == [{"file": "a.pdf", "page": 2}]


def test_not_found_never_replaces_a_real_answer(tmp_path):
    from lab.agent import AgentSettings, run_agent

    known = _agent_setup(tmp_path)
    replies = iter(
        [
            '{"answer": ["real"], "pages": [1], "sufficient": false, "next_search": "more"}',
            '{"answer": ["Not found in the document"], "pages": [], "sufficient": true, "next_search": ""}',
        ]
    )

    def text_model(prompt):
        return '{"search": "more"}' if "different approach" in prompt else '{"searches": ["q"]}'

    result = run_agent("q", lambda s: [("a.pdf", 1)], text_model, lambda c: next(replies), tmp_path, known, AgentSettings(window=0))
    assert result.answer == ["real"]


def test_exact_terms_finds_names_titles_and_identifiers():
    from lab.stages import exact_terms

    assert ["tedd", "druck"] in exact_terms("What is Tedd Druck's address?")
    assert ["humans", "of", "hud"] in exact_terms('What are the names of the "Humans of HUD"?')
    assert ["12", "2", "053"] in exact_terms("Was the case number 12-2-053 dismissed?")
    assert exact_terms("what is the total?") == []


def test_exact_seed_puts_phrase_matches_first_and_keeps_the_rest(tmp_path):
    from lab.stages import exact_seed

    index = _indexed(tmp_path)
    result = exact_seed("Where was Jane Roe born?", [("a.pdf", 1)], index)
    assert set(result[:2]) == {("a.pdf", 2), ("b.pdf", 1)}
    assert result[-1] == ("a.pdf", 1)
    assert exact_seed("what is this?", [("a.pdf", 1)], index) == [("a.pdf", 1)]


def test_blend_ranks_keeps_a_page_that_either_ranking_rates_highly():
    from lab.stages import blend_ranks

    reranked = [("a", 1), ("a", 2), ("a", 3), ("a", 4), ("a", 5)]
    bm25 = [("a", 5), ("a", 9)]
    result = blend_ranks(reranked, bm25, c=10, weight=2.0)
    assert result[0] == ("a", 5)
    assert result[1] == ("a", 1)
    assert ("a", 9) in result
    assert len(result) == len(set(result))
    assert blend_ranks(reranked, [], 10, 2.0) == reranked


def test_verify_never_replaces_an_answer_with_not_found():
    from lab.verify import merge_verdict

    draft = {"id": "a", "answer": ["4,235,643"], "citations": []}
    text = '{"quote": "", "draft_correct": false, "answer": ["Not found"], "pages": []}'
    result = merge_verdict(draft, text, [("f", 1)])
    assert result["answer"] == ["4,235,643"] and result["changed"] is False


def test_cascade_only_trusts_a_clear_confident_answer():
    from lab.cascade import combine, needs_full_read

    assert not needs_full_read(["Jane"], True)
    assert needs_full_read(["Jane"], False)
    assert needs_full_read(["Jane"], None)
    assert needs_full_read(["Jane"], "true")
    assert needs_full_read([], True)
    assert needs_full_read(["Not stated in the document"], True)
    full = {"id": "a", "question": "q", "answer": ["full"], "citations": [{"file": "f", "page": 9}], "iterations": 1}
    sure = combine({"answer": ["fast"], "citations": [], "confident": True}, full)
    assert sure["answer"] == ["fast"] and sure["iterations"] == 1 and sure["escalated"] is False
    unsure = combine({"answer": ["fast"], "citations": [], "confident": False}, full)
    assert unsure["answer"] == ["full"] and unsure["iterations"] == 2 and unsure["escalated"] is True
    assert unsure["citations"] == [{"file": "f", "page": 9}]


def test_read_fast_treats_a_failed_request_as_unsure(tmp_path):
    import urllib.error

    import pypdfium2

    from lab.cascade import needs_full_read, read_fast

    (tmp_path / "pdfs").mkdir()
    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(100, 100)
    pdf.save(tmp_path / "pdfs" / "a.pdf")
    pdf.close()

    def down(content):
        raise urllib.error.URLError("down")

    out = read_fast({"id": "a", "question": "q"}, [("a.pdf", 1)], down, tmp_path, "", 1568)
    assert needs_full_read(out["answer"], out["confident"])

    def ok(content):
        assert "confident" in content[-1]["text"]
        return {"choices": [{"message": {"content": '{"answer": ["x"], "pages": [1], "confident": true}'}}]}

    good = read_fast({"id": "a", "question": "q"}, [("a.pdf", 1)], ok, tmp_path, "", 1568)
    assert good["answer"] == ["x"] and good["confident"] is True
    assert good["citations"] == [{"file": "a.pdf", "page": 1}]


def test_classify_question_routes_each_type():
    from lab.specialists import classify_question as c

    assert c("Are both Workshop A and Board B open to the public?") == "yes_no"
    assert c("Was the defendant negligent?") == "yes_no"
    assert c("Who is older: Wanis or Julian?") == "comparison"
    assert c("Which event takes place at an earlier hour: A or B?") == "comparison"
    assert c("How many scholars attended the University of Oregon?") == "count_total"
    assert c("What was the total income of the spouse?") == "count_total"
    assert c("What are the names of the officers?") == "list"
    assert c("Which items were marked as OUT?") == "list"
    assert c("What is Tedd Druck's address?") == "lookup"


def test_specialist_rules_add_focus_without_losing_the_base():
    from lab.specialists import specialist_rules

    rules = specialist_rules("How many items are there?", "BASE RULES")
    assert rules.startswith("BASE RULES") and "count" in rules.lower()
    assert "BASE RULES" in specialist_rules("What is the address?", "BASE RULES")


def test_answer_accepts_a_prompt_per_question(tmp_path):
    import pypdfium2

    from lab.answer import answer_questions

    (tmp_path / "pdfs").mkdir()
    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(100, 100)
    pdf.save(tmp_path / "pdfs" / "a.pdf")
    pdf.close()
    seen = []

    def call(content):
        seen.append(content[-1]["text"])
        return {"choices": [{"message": {"content": '{"facts": [{"item": "x", "value": "1", "page": 1}], "answer": ["1"], "pages": [1]}'}}]}

    questions = [{"id": "a", "question": "Q?"}]
    out = answer_questions(questions, lambda q: [("a.pdf", 1)], call, tmp_path, 5, 1, lambda q: f"RULE FOR {q}")
    assert "RULE FOR Q?" in seen[0]
    assert out[0]["answer"] == ["1"] and out[0]["citations"] == [{"file": "a.pdf", "page": 1}]


def _adj_pdf(tmp_path):
    import pypdfium2

    (tmp_path / "pdfs").mkdir(exist_ok=True)
    pdf = pypdfium2.PdfDocument.new()
    pdf.new_page(100, 100)
    pdf.new_page(100, 100)
    pdf.save(tmp_path / "pdfs" / "a.pdf")
    pdf.close()


def test_adjudicate_skips_the_model_when_methods_agree(tmp_path):
    from lab.adjudicate import adjudicate_one

    rows = [
        {"id": "q", "question": "Q?", "answer": ["John  Smith"], "citations": [], "iterations": 1},
        {"id": "q", "question": "Q?", "answer": ["john smith"], "citations": [], "iterations": 1},
    ]

    def never(content):
        raise AssertionError("model should not run")

    assert adjudicate_one(rows, never, tmp_path, "", None)["adjudicated"] is False


def test_adjudicate_ignores_not_found_and_keeps_the_real_answer(tmp_path):
    from lab.adjudicate import adjudicate_one

    rows = [
        {"id": "q", "question": "Q?", "answer": ["Not found"], "citations": []},
        {"id": "q", "question": "Q?", "answer": ["real"], "citations": []},
    ]

    def never(content):
        raise AssertionError("model should not run")

    assert adjudicate_one(rows, never, tmp_path, "", None)["answer"] == ["real"]


def test_adjudicate_reads_all_cited_pages_and_records_its_decision(tmp_path):
    from lab.adjudicate import adjudicate_one

    _adj_pdf(tmp_path)
    rows = [
        {"id": "q", "question": "Q?", "answer": ["one"], "citations": [{"file": "a.pdf", "page": 1}], "iterations": 1},
        {"id": "q", "question": "Q?", "answer": ["two"], "citations": [{"file": "a.pdf", "page": 2}], "iterations": 2},
    ]
    seen = {}

    def call(content):
        seen["images"] = sum(block["type"] == "image_url" for block in content)
        seen["prompt"] = content[-1]["text"]
        return {"choices": [{"message": {"content": '{"quote": "x", "answer": ["two"], "pages": [2]}'}}]}

    out = adjudicate_one(rows, call, tmp_path, "", None)
    assert seen["images"] == 2 and '"one"' in seen["prompt"] and '"two"' in seen["prompt"]
    assert out["answer"] == ["two"] and out["adjudicated"] is True and out["iterations"] == 3
    assert out["citations"] == [{"file": "a.pdf", "page": 2}]


def test_adjudicate_falls_back_when_the_reply_is_unusable(tmp_path):
    import urllib.error

    from lab.adjudicate import adjudicate_one

    _adj_pdf(tmp_path)
    rows = [
        {"id": "q", "question": "Q?", "answer": ["one"], "citations": [{"file": "a.pdf", "page": 1}]},
        {"id": "q", "question": "Q?", "answer": ["two"], "citations": [{"file": "a.pdf", "page": 2}]},
    ]

    def bad(content):
        return {"choices": [{"message": {"content": "no json"}}]}

    def down(content):
        raise urllib.error.URLError("down")

    assert adjudicate_one(rows, bad, tmp_path, "", None)["answer"] == ["one"]
    assert adjudicate_one(rows, down, tmp_path, "", None)["answer"] == ["one"]


def test_adjudicate_retries_answer_first_after_a_runaway_quote(tmp_path):
    from lab.adjudicate import adjudicate_one

    _adj_pdf(tmp_path)
    rows = [
        {"id": "q", "question": "Q?", "answer": ["one"], "citations": [{"file": "a.pdf", "page": 1}]},
        {"id": "q", "question": "Q?", "answer": ["two"], "citations": [{"file": "a.pdf", "page": 2}]},
    ]
    prompts = []

    def call(content):
        prompts.append(content[-1]["text"])
        if len(prompts) == 1:
            return {"choices": [{"message": {"content": '{"quote": "Accounting . . . . . . . .'}}]}
        return {"choices": [{"message": {"content": '{"answer": ["two"], "pages": [2], "quote": "Acc . . . .'}}]}

    out = adjudicate_one(rows, call, tmp_path, "", None)
    assert out["citations"] == [{"file": "a.pdf", "page": 2}]
    assert len(prompts) == 2 and "answer first" in prompts[1]
    assert out["answer"] == ["two"] and out["adjudicated"] is True


def test_complete_finds_list_questions():
    from lab.complete import is_list_question

    assert is_list_question("What are the categories of risk?", ["a"])
    assert is_list_question("Who signed it?", ["a", "b"])
    assert not is_list_question("On what date was it filed?", ["10/30/2014"])


def test_complete_reads_cited_then_nearby_then_ranked_pages():
    from lab.complete import nearby_pages

    pages = nearby_pages([("a.pdf", 1), ("b.pdf", 5)], [("c.pdf", 2), ("a.pdf", 2)], lambda d: 5, 2, 8)
    assert pages[:2] == [("a.pdf", 1), ("b.pdf", 5)]
    assert ("a.pdf", 0) not in pages and ("b.pdf", 6) not in pages
    assert pages[2:5] == [("a.pdf", 2), ("b.pdf", 4), ("a.pdf", 3)]
    assert pages[5:] == [("b.pdf", 3), ("c.pdf", 2)]


def test_trigram_finds_words_that_ocr_ran_together(tmp_path):
    import json as _json

    from lab.stages import trigram

    rows = [
        {"document": "a.pdf", "page": 1, "text": "FullNameofContributor TeddDrucK 250.00 Strawberry Rd."},
        {"document": "b.pdf", "page": 1, "text": "Annual report of the county"},
    ]
    pages_path = tmp_path / "pages.jsonl"
    pages_path.write_text("".join(_json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    index = PageIndex(tmp_path / "index" / "i.sqlite", rows)
    index.pages_path = pages_path
    assert index.bm25("Tedd Druck address", 10) == []
    assert trigram("What is Tedd Druck's address?", [("b.pdf", 1)], index) == [("b.pdf", 1), ("a.pdf", 1)]


def test_incremental_finds_changed_top_pages_and_merges():
    from lab.incremental import changed_ids, merge_rows

    old = {"q1": [["a", 1], ["b", 2]], "q2": [["a", 1], ["c", 3]]}
    new = {"q1": [("a", 1), ("b", 2), ("z", 9)], "q2": [("c", 3), ("a", 1)]}
    assert changed_ids(old, new, 2) == ["q2"]
    rows = merge_rows([{"id": "q1", "answer": ["x"]}, {"id": "q2", "answer": ["y"]}], [{"id": "q2", "answer": ["new"]}])
    assert [r["answer"] for r in rows] == [["x"], ["new"]]


def test_trim_margins_keeps_content_and_scale():
    from PIL import Image, ImageDraw

    from lab.read import trim_margins

    image = Image.new("RGB", (400, 600), "white")
    ImageDraw.Draw(image).rectangle((100, 150, 200, 300), fill="black")
    cropped = trim_margins(image)
    assert cropped.size == (101 + 32, 151 + 32)
    blank = Image.new("RGB", (50, 50), "white")
    assert trim_margins(blank).size == (50, 50)


def test_units_only_touches_numbers_and_never_changes_digits():
    from lab.units import accept, is_number_answer

    assert is_number_answer(["$ 59,529"]) and is_number_answer(["925-960", "1800-1900"])
    assert not is_number_answer(["Camarones en Aguachile"]) and not is_number_answer(["J. Samuel McCue"])
    assert accept(["$ 59,529"], ["$59,529 million (Canadian dollars)"])
    assert not accept(["$ 59,529"], ["$59,530 million"])
    assert not accept(["925-960", "1800-1900"], ["925-960 MHz"])


def test_hop_reads_the_decision_and_swaps_in_new_pages():
    from lab.hop import hop_pages, parse_hop

    assert parse_hop('```json\n{"missing": true, "search": "Acme Corp principal executive offices"}\n```') == "Acme Corp principal executive offices"
    assert parse_hop('{"missing": false, "search": ""}') is None and parse_hop("no json") is None
    shown = [("a", 1), ("a", 2), ("b", 1), ("b", 2)]
    assert hop_pages(shown, [("a", 2), ("a", 9), ("a", 7)], 2, 2) == [("a", 1), ("a", 2), ("a", 9), ("a", 7)]
    assert hop_pages(shown, [("a", 1)], 2, 2) == shown


def test_zoom_reads_cited_pages_first():
    from lab.zoom import zoom_pages

    draft = {"citations": [{"file": "a.pdf", "page": 4}]}
    assert zoom_pages(draft, [["b.pdf", 1], ["a.pdf", 4], ["c.pdf", 2]], 3) == [("a.pdf", 4), ("b.pdf", 1), ("c.pdf", 2)]


def test_agent_run_survives_a_timed_out_question(tmp_path):
    from lab.agent import AgentSettings
    from lab.run_agent import answer_one

    def boom(*args):
        raise TimeoutError("timed out")

    row = answer_one({"question": "Q?", "id": "q"}, boom, boom, boom, tmp_path, set(), AgentSettings())
    assert row["failed"] is True and row["answer"] == []


def test_adjudicate_drops_yes_no_answers_to_which_questions(tmp_path):
    from lab.adjudicate import adjudicate_one

    rows = [
        {"id": "q", "question": "Which City pays over $75?", "answer": ["No"], "citations": []},
        {"id": "q", "question": "Which City pays over $75?", "answer": ["City of Haverhill"], "citations": []},
    ]

    def never(content):
        raise AssertionError("model should not run")

    assert adjudicate_one(rows, never, tmp_path, "", None)["answer"] == ["City of Haverhill"]


def test_sweep_parses_decisions_and_lists_documents_by_kind(tmp_path):
    from lab.sweep import documents_for, parse_json

    assert parse_json('```json\n{"many": true, "kind_search": "restaurant menu"}\n```')["kind_search"] == "restaurant menu"
    assert parse_json("no json") is None
    rows = [
        {"document": "menu1.pdf", "page": 1, "text": "restaurant menu appetizers calories 300"},
        {"document": "menu2.pdf", "page": 1, "text": "restaurant menu kids menu"},
        {"document": "report.pdf", "page": 1, "text": "annual report calories"},
    ]
    index = PageIndex(tmp_path / "i.sqlite", rows)
    docs = documents_for(index, "restaurant menu", "Which restaurants list calories?", 10)
    assert set(docs[:2]) == {"menu1.pdf", "menu2.pdf"} and "report.pdf" in docs


def test_score_pick_keeps_the_candidate_the_checker_believes(tmp_path):
    from lab.score_pick import pick

    _adj_pdf(tmp_path)
    rows = [
        {"id": "q", "question": "Q?", "answer": ["50"], "citations": [{"file": "a.pdf", "page": 1}]},
        {"id": "q", "question": "Q?", "answer": ["50"], "citations": [{"file": "a.pdf", "page": 1}]},
        {"id": "q", "question": "Q?", "answer": ["6"], "citations": [{"file": "a.pdf", "page": 1}]},
    ]
    score = lambda content: 0.9 if '"6"' in content[-1]["text"] else 0.2
    out = pick(rows, score, tmp_path, None)
    assert out["answer"] == ["6"] and out["scored"] is True


def test_docread_parses_items_merges_and_finds_pages_in_reading_order(tmp_path):
    from lab.docread import matching_pages, merge, parse_items

    assert parse_items('{"items": [{"item": "Laura", "page": 2}, {"item": " ", "page": 1}]}') == [("Laura", 2)]
    assert parse_items("nothing") == []
    assert merge(["ARACELI"], ["Araceli", "Laura"]) == ["ARACELI", "Laura"]
    rows = [{"document": "h.pdf", "page": p, "text": f"Humans of HUD profile {p}"} for p in (31, 10, 17)]
    rows.append({"document": "other.pdf", "page": 1, "text": "Humans of HUD"})
    index = PageIndex(tmp_path / "i.sqlite", rows)
    assert matching_pages(index, "Humans of HUD", ["h.pdf"], 27) == [("h.pdf", 10), ("h.pdf", 17), ("h.pdf", 31)]


def test_docread_reads_short_documents_in_full(tmp_path):
    from lab.docread import matching_pages

    rows = [{"document": "h.pdf", "page": 1, "text": "Humans of HUD"}, {"document": "h.pdf", "page": 2, "text": "image only"}]
    index = PageIndex(tmp_path / "i.sqlite", rows)
    assert matching_pages(index, "Humans of HUD", ["h.pdf"], 40, lambda d: 3, 40) == [("h.pdf", 1), ("h.pdf", 2), ("h.pdf", 3)]
    assert matching_pages(index, "Humans of HUD", ["h.pdf"], 40, lambda d: 300, 40) == [("h.pdf", 1)]
