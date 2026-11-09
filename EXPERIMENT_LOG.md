# MADQA experiment log

Read this before starting anything, so nothing gets repeated. Add a row after every experiment.
Raw results are in `data/results/`: `ledger.jsonl` (retrieval), `reader_ledger.jsonl` (correct
pages), `answer_ledger.jsonl` (end to end), `judged_ledger.jsonl` (judged scores).

Last updated: 2026-11-03

## How to read the numbers

- **Judged:** MADQA's own prompt and rules, with `deepseek/deepseek-v4.1-flash` standing in for its
  Gemini judge and no bias correction. Comparable between our runs, and only an estimate of a
  leaderboard score.
- **Text:** ANLS* >= 0.5, with no judge. It undercounts correct answers and isn't comparable to the
  leaderboard.
- **Correct pages:** the reader is handed the labelled pages, so this tests reading alone.
- **End to end:** search plus reader, which is what the leaderboard measures.
- **Noise:** about +/-5 points on 200 questions. Small single results may be chance.
- Ignore the 50.5 row in `judged_ledger.jsonl` (judge calls failed) and the 57.5 thinking row in
  `reader_ledger.jsonl` (parser bug, 81.5 after the fix).

## Current status

| | |
|---|---|
| Best dev, judged, end to end | **91.0** Qwen only (adjudicator over verify, workspace, agent; units). 92.0 with Gemma, dropped by the user's choice of a Qwen-only system |
| Best estimate on untuned questions | **92.0** Qwen only (92.5 with Gemma, not used) on train mix200 (test mix, untuned): full stack (verify, workspace, agent, adjudicator, units). By type 96.58 / 82.35 / 75.0; Page F1 87.97 |
| Targets on the leaderboard | 94.4 best open-weight (ARRAY), 96.9 overall (Mixedbread), 99.4 human with correct pages |
| ColQwen visual page retrieval (T18) | Adopted 2026-11-06: dev 187 to 189 of 200 (judged >= 0.5), train 142 changed questions 87 to 114 right (+30/-3, about +1.7 points over 1,550). Expected: dev ~92 to 93, test mix ~93 to 94 |
| Official leaderboard score | None. Nothing has been submitted |

**Current best stack:** BM25 over OCR text with document context, Qwen3-Reranker-0.6B on the top
100, LLM page selection from the top 20, top 3 pages plus neighbours (up to 9 images; plus ColQwen2.5's top page when the text ranking missed it, `thin_ocr_visual`), reader
`RadixArk/Qwen3.8-27B-NVFP4` on vLLM, images at 2,048 pixels, document labels, prompt v3.

**Where points are lost (train check, test mix):** single-page about 6.6, cross-page about 4.5,
cross-document about 3.1.

## Kept

| Change | Before | After | Metric | Cost |
|---|---:|---:|---|---|
| OCR on near-empty pages (<20 chars) | 47.08 | 63.48 | Dev page recall@5, 676 PDFs | One run, 0.5 s/page on GPU |
| Reranker, Qwen3-Reranker-0.6B, top 100 | 63.48 | 70.98 | Dev page recall@5, 676 PDFs | 1.3 s/question uncached |
| Recovering 123 PDFs through Chrome (799 of 800) | 51.5 | 61.0 | End to end, text, 3 pages, no reranker | One time |
| Reranker, end to end | 61.5 | 69.0 | End to end, text, 5 pages | |
| Reading 5 pages instead of 3 | 67.0 | 69.0 | End to end, text; within noise | About 40% slower |
| Qwen 3.8 NVFP4 instead of Qwen3.5-9B | 76.0 | 79.0 | End to end, judged | About the same speed |
| Neighbouring pages (top 3 plus neighbours) | 79.0 | 83.0 | End to end, judged | About half the speed |
| Neighbouring pages, correct-pages test | 83.5 | 86.5 | Correct pages, judged | |
| OCR on thin pages (<200 chars) | 83.0 | 84.0 | End to end, judged; Page F1 80.92 to 83.44 | One run, 0.34 s/page |
| Document labels on page images | 84.0 | 84.5 | End to end, judged; text 75.0 to 78.5 | None |
| Prompt v3 | 84.0 | 84.0 | End to end, judged; average 82.0 to 83.5 | None |
| Images at 2,048 pixels | 83.5 | 86.0 | Correct pages, judged; 84.0 to 84.5 end to end | Half the speed |
| Labels + v3 + 2,048 together | 84.0 | **86.5** | End to end, judged | |
| LLM page selection, top 20 | 84.17 | 86.16 | Page F1; judged unchanged at 86.5; recall@3 76.01 to 81.31 | 2.4 s/question |
| Agent loop, cross-document questions (135 train) | 69.6 | 77.8 | Judged; 17 gained, 6 lost, p = 0.035; Page F1 71.0 to 73.7 | Extra planning and reading calls |
| Adjudicator over verify, workspace and agent (T30) | 87.5 | **90.5** | Judged, all 200: gained 6 (57, 69, 85, 104, 127, 178), lost 0, p = 0.031; Page F1 86.16 to 87.41; by type 94.48 / 72.22 / 73.68; 39 of 200 adjudicated | Three full reads plus one adjudication on disagreement; agent is slow (0.027 q/s) |
| Gemma 4 31B as a 4th adjudication candidate (T10) | Dev 91.0 / train 92.0 | **92.0 / 92.5** | Dev gained 112, 119, lost 0; train gained 353, 833, 1058, lost 539, 1386. Gemma alone: dev 84.5, train 86.0. Page F1 88.08 to 87.17 (dev). Disagreements 40 to 53 | Gemma read ~18 min per 200 + model swap ~8 min |
| Units step: number answers re-read on their cited pages for unit and scale, digits must stay identical (`lab/units.py`) | Train verify 87.0 / dev 91.0 | **88.0 / 91.0** | Train gained 479 (MHz), 1115 (millions of CAD), lost 0; dev changed 9 ("$", "acres"), none flipped | About 0.3 s/question (67 s per 200) |
| Adjudicator retry + salvage of cut-off replies | 90.5 | **91.0** | Judged, all 200: dev/87 recovered. The quote repeated dot leaders ("Accounting . . .") until the token limit; now retries answer-first and reads answer/pages from unclosed JSON | One extra call on failure only |
| Agent searches in parallel (thread-local SQLite, locked reranker; no answer logic changed) | 0.028 q/s | **0.078 q/s** | 16 never-read train questions, cold, 8 workers: 2.8x faster. dev/87, 104, 127, 178 (agent-only wins) still all right | None |
| Agent run survives timeouts (any request error now marks one question failed) | Crash | Fixed | A single timed-out request killed a 33-minute train run | None |
| Verify and fix, without blend (T2) | 86.5 | **87.5** | Judged, all 200: changed 6, gained 3, lost 1. Keeps draft citations unless the answer changes | Doubles reading time |
| Verify and fix (on top of blend) | 85.0 | **87.0** | Judged, all 200: changed 9, fixed 4, lost 0; hard set 6 of 26 with blend | Doubles reading time |
| Document context added to search text | 78.24 | 83.88 | BM25 dev recall@5; train 80.84 to 83.15; judged end to end tied at 79.0 | Tiny |

## Tried and not adopted

Don't retest these unless something relevant changes.

| What | Result | Why dropped | Worth retrying if |
|---|---|---|---|
| OCR text only, no images | 64.60 vs 78.88, text, correct pages | Clearly worse | No |
| Images + OCR text together | 78.26 vs 78.88, text, correct pages | No gain, 22% slower | Not judged; could return as a check on misread characters |
| Prompt v2 ("value only") | 76.0 to 75.0 judged | Likely stripped units | Replaced by v3 |
| Thinking on Qwen3.5-9B, all questions | 70.5 to 71.0 text; Page F1 77.22 to 71.90; 3x slower | Little gain, weaker citations | Only for uncertain questions |
| SAG2-style expansion, simple version | Recall@5 86.53 to 85.35; 2x slower | No gain | With judging after each round and earlier evidence visible, as SAG2 does |
| Neighbours for retrieval only, 676 PDFs | Recall@10 49.12 to 51.96, inconclusive | Superseded | Kept later for the reader |
| Reranker on enriched text | Recall@5 86.34 vs 86.53 | No gain, blurs pages | Use plain text to rerank |
| Claude Haiku 5.5 reader | Correct pages 86.5 judged vs Qwen 83.5; end to end 74.0 vs 73.5 text; p = 0.56 on text | A statistical tie, and closed | User chose Qwen and stopped using Haiku |
| Correcting labels locally | **Done 2026-11-03 at the user's request** for dev/3, 12, 171, 198: corrected answers added as extra accepted variants, originals kept (`data/questions/dev.original.jsonl`, `experiments/label_corrections.json`) | Affects local dev scoring only, not the leaderboard | — |

| Exact-name seed placed first | Dev recall@5 86.63 to 62.36; recall@100 97.00 to 97.50 | Floods the top with repeated document text | Only as extra candidates for a later ranker |

| Agent loop on cross-page questions | 73.5 to 74.5 judged (102 train), Page F1 66.4 to 60.6 | No gain, weaker citations | Only with a read-more-of-the-document step |

| Maths in code | Fixed 1 of 26 hard questions | Model picks the wrong operation (lists became counts) and misreads values | Only if restricted to clear sum questions |
| Adjudicator told which page each candidate cites (`--sources`) | 91.0 to 90.5, 0 gained, 1 lost (dev/143); Page F1 88.08 to 87.24; 79 s | No gain | No |
| Majority vote instead of the adjudicator (offline, saved answers) | Right on dev: adjudicator 182, majority else adjudicator 178, majority else verify 175 | The adjudicator sometimes correctly overrules two agreeing methods | No |
| Skip the agent when verify and workspace agree (offline) | Skips the agent on 174 of 200; right 181 vs 182 (loses dev/104) | Costs accuracy; accuracy comes first | Fine for quick experiments only |
| Read, verify, workspace back to back per question (`lab/chain.py`) | Same answers 24 of 24 for all three; 0.056 vs 0.060 q/s; 7 min (over the 5-minute limit) | No speed gain: with 8 in flight, KV room for about 4 evicts the pages before verify | With fp8 KV cache or fewer workers |
| fp8 KV cache (`--kv-cache-dtype fp8`) | KV size unchanged at 102,673 tokens | The NVFP4 checkpoint already stores KV in fp8 | No |
| `--gpu-memory-utilization 0.90 --max-num-batched-tokens 16384` | KV only 102,673 to 105,949 tokens; 24 reads ran past 400 s (normally about 2 min), throughput often 0 with 31.9 of 32.6 GB used | Likely Windows spilling GPU memory to system RAM; reverted to the original server | Try `--max-num-batched-tokens 8192` alone at 0.85 |
| Adjudicator quotes evidence for each candidate before choosing (`--each`) | Dev 91.0 vs 91.0, train both-wrong 8 vs 8 of 21; 0 gained, 0 lost; 129 s | No effect | No |
| Same-document pages in the 9 slots (offline recall) | Train recall@9 92.0 to 92.25, cross-page 80.9 to 81.8 (2 slots); dev 93.58 to 93.83 | Too small to test end to end | No |
| Second hop inside the shown documents (T32, `lab/hop.py`) | 14 cross-page misses (9 train, 5 dev): hop wanted on 6, re-read 3, fixed 0 | The second page is often in a document not shown (1157, 1190), and re-reads gave other wrong answers | With search across all documents (agent-style) |
| Sweep: check every document of a kind for collection questions (T33, `lab/sweep.py`) | 9 cross-doc misses: triggered on 3, fixed 0. With 25 docs x 3 pages: train/760 found all 3 gold menus but per-document yes/no checks missed Pueblo Viejo and added 2 false ones | Per-document checks too unreliable (46 documents match "restaurant menu") | With a stronger checker, or menus pre-tagged with features |
| Sampled Qwen reads (temperature 0.7, 3 samples) as extra candidates, 31 hard only | Right in some sample: dev/3, 112, 119, 155, train/353 (dev/3 and 155 never right before); adjudicated 3 of 31 vs 2 | Kept for the 31-question work | — |
| Score-and-pick (T34, `lab/score_pick.py`): each candidate scored alone by P("yes") to "is this correct?" | **4 of 31** (adds dev/119) vs adjudicator 3 of 31, same 6 candidates; 1 min | Better chooser on the hard set; 200 check later | — |
| Document deep read for list answers (T35, `lab/docread.py`) on the 31 | 10 -> **12 of 31**: dev/23 (7 appendices), dev/56 (9 names), train/1384 (11 products) now complete; lost dev/117 (adds a stamp date), dev/18 varies between runs | Kept for the 31 work. Key settings: one page per request (with 4+ pages the reader lists one item per batch), short documents (<= 40 pages) read in full (search missed 4 of 8 profile pages), text-only tidy pass (an image pass dropped items whose page wasn't shown) | ~2.5 min for the 31 (159 + 128 one-page reads) |
| Deep-read gate (T35b, `wants_set` in `lab/docread.py`): skip comparisons ("lower than") and single facts ("on what date") | 12 -> **13 of 31**: dev/117 back (no stamp date); dev/15 no longer gets every year (still wrong). Gate skips only dev/15 and dev/117 of the 16 list questions; scored on the saved run, no GPU | Kept | Fewer reads |
| Deep read extras (T35c): case-number filter + rebuild (train/353), find mode for "..." answers (dev/36), per-item yes/no check on the item's pages (`--check`) | 3 runs of the 31: 12, 13 (check 0.5), 12 (check 0.5 + item's own pages); check 0.1 on saved scores **13 of 31**. Filter works: train/353 now exactly 9 jurors, 1 misread (64 vs 66). Find on dev/36 answers from the wrong section. Check drops junk (RFeyeSPI 0.006, Irv Dennis) but at 0.5 also real items (RFeye View 0.13, an appendix 0.47). **Run-to-run variance of ±1-2 on the 31** (reader extracts different extras; judge flips on dev/23 "Acronyms List" with only item order changed) | Kept, check at 0.1. Single runs can't separate changes of 1 question | +~0.3 min |
| Score-and-pick on all 200 dev (T34 check, 3 candidates: verify, workspace, agent; `dev_pick3.jsonl`) | Adjudicator 187 right (judged >= 0.5) vs pick 181; gained 0, lost 6 (dev/10, 20, 69, 85, 143, 166) | Worse than the adjudicator on the full set despite 4 vs 3 on the hard 31; 39 of 200 scored, 68 s | Not adopted. Hard-set wins don't carry to all 200 | No |
| Deep read (T35) on all 200 dev, applied to the final answers (`dev_final_units_docread.jsonl`) | 187 to 175 right (judged >= 0.5): gained 3 (dev/18, 23, 56), lost 13 (dev/1, 22, 26, 29, 35, 40, 65, 66, 77, 85, 116, 134, 135) | It deep-read 61 list answers and changed 24; it adds extra items to lists that were already complete. Keeping only changes that contain every original item still gives 177 (+1, -11). 12.5 min | Not adopted on the full set. The hard-set gain (13 of 31) doesn't hold on all 200 | Only with a gate that tells an incomplete list from a complete one |
| Gating the deep read by its per-item check score (offline, from `check_scores` in `dev_final_units_docread.jsonl`) | No threshold separates gains from losses | Items that wreck complete lists score high (dev/1 states 0.94-1.0, dev/22 steps 0.96-1.0, dev/65 0.76-0.96) because the check asks "is this a member of the kind" and ignores the question's qualifier ("favorites", "former officers", "reported changes", "term ending"); real gains score low or mixed (dev/23 0.47, dev/56 Irv Dennis 0.04) | Not adopted | A qualifier-aware check, or a gate that detects whole-document collection questions only |
| ColQwen2.5 visual retrieval (T18, `lab/visual.py`, image `madqa-colpali`): add ColQwen's single best page to the reader's 9 only when the text ranking doesn't already show it | Dev recall@9 93.58 to **95.83**; ColQwen alone @3 81.02, @9 90.12, @50 94.33; text top 9 + its top 3 gives 96.46 | **Plain reader, the 20 dev questions whose pages changed: 12 to 17 right, gained dev/104, 120, 127, 155, 178, lost 0** (judged >= 0.5). Larger additions changed too many questions (top 5, 2 slots: 163 of 200 changed, recall 95.56) | One-time download (vidore/colqwen2.5-v0.2, ~7 GB, 27 min at 2.6 MB/s) + embedding 18,619 pages (40 min, 8 pages/s, 3.4 GB in `data/visual/`). Query: scores all pages | Kept pending train check and full stack |
| ColQwen check on train (T18): same setting, not tuned on train | Train recall@9 92.00 to **95.60** (ColQwen alone @9 94.76, @3 88.72, @50 97.67); top 9 changed for 142 of 1,550 | **Plain reader on those 142: 70 to 108 right, gained 39, lost 1 (train/18)**, judged >= 0.5, 0 judge errors. Over all 1,550 that is about +38 net, +2.5 points | Same as above; recall run took 40 min for 1,550 queries | **Kept for the plain reader.** Full stack (verify, workspace, agent, adjudicator) not yet tested with it | |
| ColQwen in the full stack, dev (T18): verify and workspace re-read with the ColQwen page on the 20 changed questions, old agent candidate kept, adjudicator + units; other 180 carried over (`dev_final_visual_units.jsonl`) | 187 right (93.5, judged >= 0.5; mean 93.25) | **189 right (94.5; mean 94.25)**: gained dev/120 and dev/155, lost 0; 0 judge errors | About 25 min for the 20 questions | Kept; matches the prediction (dev/104, 127, 178 were already fixed by the agent). Needs the train full-stack check and the agent re-run with ColQwen | |
| ColQwen in the full stack, train (T18): verify, workspace and agent (agent run on all 142, same candidate both sides), adjudicator + units on the 142 questions whose top 9 changed | Old pages: 87 right of 142 | **New pages: 114 right of 142: gained 30, lost 3 (train/267, 337, 1355), net +27, about +1.7 points over all 1,550**; 0 judge errors | Agent on 97 questions took about 25 min; the whole check about 2 h | **Adopted.** ColQwen top-1 page (2 reader slots swapped only when text ranking missed it) is now part of the stack: `thin_ocr_visual`. Dev +2/-0, train +30/-3 | Not yet re-run with ColQwen inside the agent's own search; test split not run |
| ColQwen on train mix200 (no new run: 15 of the 200 had changed pages, taken from the 142-question run above) | 185 right (92.5; the log's 92.0 uses a different count) | 185 right: gained train/318, lost train/1355 | None | No change on this 200: too few changed questions to see +1.7 points. The all-1,550 train number (+30/-3) is the better estimate | |
| ColQwen inside the agent's search (`lab.run_agent --visual`, query side on CPU because the GPU is full; `CpuSearcher` in `lab/visual.py`): ColQwen top page replaces the 3rd page of each search when the text ranking missed it | The 11 dev questions still wrong: 0 fixed | 1 of 11 fixed (dev/98), 10 unchanged | **About 60 s per search on CPU** (14 searches, 845 s) against a few seconds; agent 0.048 q/s vs 0.078 | **Not adopted:** too slow, gain within noise; the main pipeline already gets ColQwen's pages | Could work on GPU if the reader and ColQwen could share it |
| ColQwen setting sweep, recall only (take 1-2 pages, depth 1-3) | take 1 depth 1: train 95.60 (142 changed), dev 95.83 (20 changed) | Deeper: train up to 96.42 (take 2 depth 3) but 961 of 1,550 questions change; dev gets worse for depth 2 (95.42) and only 95.90 for take 2 depth 3 | None (offline) | Kept take 1 depth 1: +0.5 recall is not worth re-reading 600-960 questions, each a chance to break a right answer | |
| Deep read with a qualifier-aware item check (`--qualified --check 0.5`, `CHECK_QUALIFIED` in `lab/docread.py`) on all 200 dev (`dev_final_visual_units_docread.jsonl`) | 189 right | 183 right: gained dev/18, 23, 56; lost dev/1, 16, 22, 26, 28, 30, 40, 66, 85 (9; the plain check lost 13) | 62 list answers read, 16 changed, 11.6 min | Not adopted | The check still scores wrong extras 0.99 (dev/22, 26, 85: gold list is narrower than the page, or a near-match); no threshold separates them. A gate on whether the first list is incomplete is what's missing |
| Three sampled reads (temperature 0.7) as extra adjudicator candidates, all 200 dev (`dev_vote_base_units.jsonl` vs `dev_vote_samples_units.jsonl`; same verify, workspace and agent candidates) | 188 right (a fresh rerun of the 189; adjudicator noise about 1) | 188 right: gained dev/112, 130; lost dev/26, 95 | 3 sampled reads took about 45 min, adjudication about 20 min | Not adopted | No gain over the same stack without them; hard-set gain (3 vs 2 of 31) does not carry to all 200 |
| Entity guard (`--entity` in `lab/docread.py`): answer cites only documents that never mention the named person/place while the top-ranked document does, so re-read the top document | Flags dev/36 and train/175 only (0 of ~390 right answers flagged: dev 189 + train mix200 181 with names) | Neither fixed. dev/36: re-read on the right document gets page 7, but the gold ($96,680) is the SEM row alone while the page lists two spouse-salary rows ($143,305 total), so the label is ambiguous. train/175: the top document is itself wrong (retrieval miss) | Seconds | Not adopted (flag off by default) | Fires on about 0.5% of questions, fixes none; dev/36 is a label ambiguity |
| Full stack with ColQwen on 300 random train questions outside mix200 (`experiments/train_extra300.json`; `train_extra300_adj_units.jsonl`) | Not run before | **285 of 300 right (95.0%, judged >= 0.5)**; candidates alone: workspace 282, verify 277, agent 271; 2 judge misses | Reads ~2.5 h (agent 5 h wall with the GPU shared), judging ~10 min | Fresh, untuned estimate: random sample, so more single-page than the test mix (the test-mix mix200 was 92.5) | 15 misses: 6 had a right candidate that the adjudicator overruled (train/10, 125, 476, 1051, 1219, 1489), 4 retrieval misses (0 of the gold page shown), 2 misspelled names copied from the image (228, 1335), 2 lists/answers incomplete, 1 unit/format (637 '134,300' vs '$134.3 million') |
| Majority override of the adjudicator (when 2 of the 3 candidates agree and the adjudicator picks something else), offline on dev, extra300, mix200 | Final right: dev 188, extra300 285, mix200 185 | Dev 187 (+1/-2), extra300 284 (+2/-3), mix200 179 (+2/-8) | None | Not adopted | The adjudicator overruling the majority is usually correct |
| Snap answer words to near-identical words on the cited pages (offline, edit ratio >= 0.84), dev, extra300, mix200 | Final right as above | 18 answers changed, 0 gained, 0 lost | None | Not adopted | Does not fix the misspelled names (the correct spelling is not in the text layer); corrupts others (Freedlom, acres to acre) |
| Deep-read answer as a 4th adjudicator candidate (`--check 0.5 --qualified`, then `lab.adjudicate` with verify, workspace, agent, deep read), dev then untuned train | Dev 188 (rebuilt base), extra300 285, mix200 185 | **Dev 192** (gained dev/18, 23, 56, 130; lost 0). **extra300 285** (deep read changed 1 of 300, nothing gained or lost). **mix200 184** (gained train/726; lost train/645, 1386) | Deep read 12-15 min per 200-300, adjudication 10 min | **Not adopted as the headline.** Gain is on dev, which the deep-read settings were tuned on; untuned train shows no gain (pooled train +1/-2 over 500) | Few lists to fix in train samples; a test on a list-heavy train sample would settle it |
| Same deep-read candidate on 60 untouched list-type train questions (`experiments/train_lists60.json`, chosen by wording; full stack run first) | 57 of 60 right (95.0) | 57 of 60: the deep read changed 0 answers | Full stack ~1.5 h | Neutral on untuned data: no gain, no harm. Pooled over all untuned train (extra300 + mix200 + lists60 = 560): +1/-2; dev +4/-0 | Adopt as an optional last stage only; not part of the headline estimate. Dev list failures (5 of 200) are rarer in train (3 of 60 list-type) |
| Higher-resolution candidate: reader on the top 4 pages at 2,560 px (the 32,768-token limit rules out 9 images above 2,048) as a 4th adjudicator candidate, dev and untuned train extra300 | Dev 188, extra300 285 | Dev 189 (gained dev/112, 130; lost dev/95); extra300 285 (gained train/476; lost train/1550) | Reads about 10 min per 200, adjudication 10 min | Not adopted | Net +1 on dev, 0 on train: within adjudicator noise |
| Reciprocal rank fusion of the text ranking and ColQwen (k = 10, 30, 60; and text top 3 + fusion), recall of the 9 reader pages, offline on train 1,550 and dev 200 | Swap-1 (current): train 95.60 (142 changed), dev 95.83 (20 changed) | RRF k=10: train 96.75, dev 95.90; k=30: 94.97 / 94.50; k=60: 94.42 / 94.17; keep top 3 + fusion: 95.13 / 95.08. All change the pages for every question | None (offline) | Not adopted | Only k=10 beats swap-1 on train (+1.2) and it ties on dev; it reorders every question's pages, so each answer is re-exposed to risk. Not worth it without a reader run |
| Learned chooser (logistic regression over agreement, method, length, not-found words, adjudicator flag) picking among verify, workspace, agent and the adjudicator's answer; trained on the 560 untuned train questions, tested on dev; also leave-one-set-out | Adjudicator: dev 188, extra300 285, lists60 57, mix200 185 | Chooser: dev 186, extra300 284, lists60 56, mix200 183. Oracle (any candidate right): dev 189, train 538 of 560 | Offline, seconds | Not adopted | The adjudicator, which reads the pages, beats a chooser that only sees the answers. Dev has only 1 question where a right candidate is not picked |
| Error analysis of all 45 wrong final answers across dev 200, extra300, lists60, mix200 (760 questions) | 715 right (94.1%) | By cause: partial lists about 14 (train/10, 293, 318, 703, 760, 1058, 1307, 1384, dev/15, 18, 23, 56, 74...), extra items or extra words 4 (train/1051, 1210, 1363, 1058), misspelled names 2 (1335, 228), retrieval miss (0 gold pages shown) about 8, wrong value or wrong document the rest | Offline | Analysis only | No single cause is more than about 2% of questions, so no one general fix is worth more than about +0.5 points |
| Thinking in the adjudicator | 91.0 to 89.0 judged, 0 gained, 4 lost (10, 85, 87, 130); 374 s vs 133 s | Worse and 3x slower | No |
| Thinking when first and verified answers disagree | Fixed 2 of 26, both also fixed by verify; 3 of 8 unusable | Adds nothing, slow | No |

| List completion: re-read cited pages ±2 for list questions (T14, `lab/complete.py`) | Hard set: 9 list questions, changed 0 | Reader keeps the same list even with the next page shown; dev/3's extra gold items are a different section | Only with a stronger reader |
| BM25 route as a fourth adjudication candidate (`experiments/thin_ocr_bm25_route.json`) | Hard set: alone fixed 1 (dev/112), after adjudication 0 net (adjudicator didn't choose it); dev/36 still missed | No gain | If the adjudicator gets better at choosing |
| 12 pages at 2,048 pixels | HTTP 400: over the 32,768-token context (9 pages use about 28.7k) | Use 9 pages, or smaller images for more | |

| BM25 blend | 86.5 to 85.0 judged, 1 gained, 4 lost, despite recall@5 89.05 to 90.05 | Offline recall gain didn't reach answers | No |

| Staged read, escalating on self-reported confidence (T1) | 85.0 to 81.5 judged, Page F1 85.94 to 83.38; reader time about 40 s to 10.7 s; 10.5% escalated | Overconfident: 85.5 vs 89.4 on the questions it kept | Retry with question-type routing or disagreement between two reads |

## Failed to run

| What | Problem | Status |
|---|---|---|
| `RedHatAI/Qwen3.5-9B-quantized.w8a8` as INT8 | vLLM: `Int8 not supported on SM120` on RTX 5090 | Works unpacked to 16-bit |
| Official `Qwen/Qwen3.8-27B-FP8` | 30.4 GB of weights is too tight for 32 GB | Not attempted. Use NVFP4 (21.9 GB) |
| Images at 3,000 pixels | `400 Bad Request`, likely past the 32,768-token limit | Could try about 2,500 |
| Qwen 3.8 with thinking | Connection reset on long replies | Streaming fix is in. **Not yet rerun** |
| `23696095.pdf` | 403 AccessDenied from the source, also reported on MADQA issue #2 | One PDF missing, 4 train questions affected |

## Findings to remember

- **Reader is the main limit:** end-to-end (86.5) nearly equals correct pages (86.0 at 2,048 pixels), measured on different setups.
- **Benchmark labels have errors:** about 1-2% of checkable questions have the answer only on a
  neighbouring page (dev/118, dev/198), and dev/198's accepted answer looks like a typo. We now score dev with 4 corrected labels
  added as extra variants (see `experiments/label_corrections.json`); earlier dev numbers used the original labels.
- **Small samples mislead:** dev has 18 cross-page questions. Cross-page read 55.56 on dev and 73.5
  on train's 102. Use train for multi-step questions.
- **Hard-20 misleads:** Haiku looked 18 vs 13 better there, but only 80.5 vs 78.5 on all 200.
- **Leaderboard pages need Chrome:** plain fetches only see an empty Streamlit shell, and the saved
  results in the Space repo are stale.

- **Empty answers mean a search miss:** dev/120, dev/127 and dev/178 returned empty because the correct
  page wasn't shown. Empty or "not found" now triggers a different search in the agent.
- **BM25 pool is already strong:** on enriched text it reaches recall@100 of 97.00 on dev and 95.16 on
  train, so most misses are ranking problems rather than pages that can't be found.
- **Later ranking can bury good BM25 hits:** dev/127 is rank 1 in BM25 and rank 10 after reranking and LLM selection.
- **Blending BM25 back in looks strong (offline, not yet end to end):** train recall@3 79.56 to 84.62,
  @5 86.62 to 89.14, @9 90.72 to 92.45. Dev @3 81.31 to 86.44, @5 89.05 to 91.70, @9 93.58 to 94.33.
  `blend_bm25` stage built, c=10, weight=2. Dev's blend was over LLM selection; train's over reranker and neighbours.
- **Alone, workspace and agent don't beat verify on 200:** workspace 88.0 (gained 3, lost 2), agent 85.0 (gained 6, lost 11).
  Together under the adjudicator: 90.5, 0 lost. Their value is in disagreeing, not in replacing.
- **Runaway dot leaders:** tables with dot leaders can make Qwen repeat " . " until max tokens. Seen in the adjudicator (dev/87).
- **Not fully deterministic at temperature 0:** rerunning the same adjudication flipped dev/119 between two wrong answers (100 and 50). Expect about ±0.5 noise per run.
- **Remaining 18 dev misses (hard set `experiments/hard_adjudicated.json`, 19 before the fix):** likely label issues 12, 171, 198
  (and maybe 23, 56); all methods agree and are wrong: 3 (list incomplete: 5 of 9), 117 (extra date), 155, 74 (third name
  on an unretrieved page), 96, 36 and 120 (retrieval); adjudicator chose wrong although a candidate was right: 15, 98, 119;
  list errors 18, 44, 23.
- **OCR runs words together:** dev/120's page reads "TeddDrucK", so word BM25 can't match "Tedd" or "Druck". A trigram
  (substring) FTS5 index finds it. Offline pool recall, word top 100 plus trigram top 20: dev 97.0 to 98.25 (dev/56, 104, 120),
  train 95.16 to 95.69 (10 questions). Trigram alone, top 20: dev 92.57. 0.024 s/question, 127 MB index, 5 s to build.
  Stage `trigram` built; config `experiments/thin_ocr_trigram_llm_select.json` (T31).
- **Why the agent is slow (dev profile):** search averages 153 s but the pipeline only 30 s. All 8 workers queue on one
  searcher thread, and each planned search is a keyword rewrite of the question (only 20 of 200 equal it), so the reranker
  scores 100 uncached pages (about 23 s). Reusing the saved ranking would change the agent's behaviour, which is part of
  why it finds dev/87 and dev/104. Fixed: searches now run in parallel (2.8x). Remaining cost: reranker turns on the GPU (about 15 s per search under load) and LLM selection.
- **Idle-GPU timing, one fresh question (train/10):** read 6.3 s, verify 1.5 s, workspace 2.6 s (vLLM reuses the cached
  pages), agent 33.9 s (search 22 s: reranker 14 s cold, LLM selection 7.8 s), adjudicate 1.8 s. About 46 s in all, 12 s without the agent.
  Warm reranker: 100 pages in 1.8 s at batch 8 (batch 32 and 64 are slower).
- **The reader server is memory-bound:** vLLM runs only about 4 requests at once (9 pages at 2,048 px is about 28k tokens,
  and about 5 GB is left for KV cache), with up to 6 waiting. Prompt throughput is about 2-3k tokens/s and the prefix cache
  hit rate is 1.3%. fp8 KV was already on; memory 0.90 made it far slower. The GPU memory is the hard limit.
- **Train hard set (`experiments/train_hard.json`):** 21 both-wrong + 9 one-wrong of train mix200. After the agent and
  adjudicator, 13 of the 21 remain: missing units (479, 1115, now fixed), adjudicator overruling a right agent (353, 1384),
  misreads (89, 56), retrieval (175, 1190, 318, 760), lists (293, 1307), arguable label (153).
- **Cross-page misses:** train 9 (6 with only 1 of 2 gold pages shown, 2 with none, 1 with all), dev 5 (2 all shown, 3 partly).
  Two-step questions inside a document: the second page matches the bridge fact, not the question.
- **Don't run tests while a long GPU job runs:** the zoom test (3 pages at 2,800 px) alongside the train agent run took 24 min for
  18 questions, with 7 requests timing out. Results discarded; rerun with the GPU free.
- **GPU overflow was the real cause of slow agent runs:** the reranker's PyTorch memory crept up until the card hit 31.85/31.84 GB
  and Windows spilled 0.77 GB to system RAM (144 W, throughput 0). With caps (reranker 0.05 of the card, batches of 4, empty_cache
  after each search) the same run did 8 questions/min (97 in 12 min) at 570 W. Watchdog `tools/gpu_watchdog.ps1` kills GPU jobs if
  shared GPU memory exceeds 0.4 GB (idle baseline 0.17). Agent runs now save each answer and resume.
- **Oracle on the 31 hardest (dev 18 + train 13):** only 10 have ever been answered right by any saved run; 21 never.
  Label suspects among the 21: dev/12, 171, 198, likely 3, 23, 153. The adjudicator misses right candidates on dev/112 and 119.
- **Wrong-kind filter:** the adjudicator now drops a bare yes/no answer to a which/what/who question (dev/112 answered "No" to
  "Which City"). Only 1 of 800 dev+train answers was affected.
- **Cross-document misses (train 5, dev 4):** mostly collection questions ("which restaurants...", "whose cases...", "earliest
  fine by any..."): train/318, 760, dev/74, 96. Others: arithmetic over two documents (train/181: 31 vs 30), bridge (726),
  comparison (1242, dev/155), table years (dev/15).
- **Qwen only (user decision, 2026-11-03):** Gemma 4 helped as a 4th candidate (+1 dev, +0.5 train) but the system stays on one
  model: simpler, no model swaps. Gemma weights remain in the HF cache (23.3 GB) if this is revisited.
- **Second overflow, 16:15 (vLLM at 0.82):** the desktop compositor (dwm) grew to 9.06 GB reported on the 5090 and shared memory
  reached 1.29 GiB during score-and-pick on train (378 s instead of ~30). vLLM now at **0.78** (KV 52,428 tokens, about 2 full
  requests at once); GPU 25.6 GB used, ~6 GB free. Watchdog now also stops host `python -m lab.*` jobs. Docker VM file cache dropped
  (RAM 48.6 to 38.8 GB). Permanent fix: monitor on the motherboard port.
- **Deeper reads (ranking positions 10-27) on the 31:** train/1190 right for the first time, but score-and-pick prefers the wrong
  "Raymond W. Kelly" (0.55 vs 0.009). With 8 candidates score-and-pick got 3 of 31 (lost dev/119 vs 4 with 6 candidates).
- **Score-and-pick variants on the 31 (8 candidates):** shared pages 3, own cited pages (`--own`) 4 (dev/119, 155 new, train/479,
  1115; loses dev/112), mean of both 2, max 3. The pickers disagree on near-ties; the right answer is among candidates for about 9.
- **Label check of the 31, by eye (2026-11-03):** label errors (unfixable, labels unchanged): dev/171 (page "rnorton@camken.com",
  label "rmorton"), dev/12 (11 states over two lines, label kept only the wrapped "Tennessee"), dev/198 (label "$3,417,108" drops
  the leading 1 of $13,417,108), dev/3 (label adds the 4 "Good Retail Practices" sections from page 2 to the 5 risk-factor
  categories). dev/155 label is right (MWI $7,500,000 vs Indianapolis $1,241,500). **27 of the 31 are fixable.**
- **Root causes of the ~21 fixable hard failures (2026-11-05):** A) lists spread over many pages of one document, 10
  (dev/15, 18, 23, 56, 74, train/318, 353, 760, 1307, 1384); B) wrong page or document, 7 (dev/36, 96, 112, 120, train/175,
  1190, 293); C) fine print misread, 2 (train/89 1911 newspaper, train/56); D) two valid sources, 1 (dev/98). Earlier tables cut
  gold lists at 3 items, which hid that many "wrong" answers were correct but incomplete.
- **Workflow:** test on the current hard set first (`experiments/hard_current.json`, 26 dev questions).
  Run all 200 only once satisfied, then check on train.

- **Agent gains come from splitting, not retries:** 213 of 237 questions finished in one round. The run
  predates the retry on empty answers and BM25 blending. Route to it when the plan has 2+ searches.

- **Verify and fix looks strong:** fixed 5 of 26 hard questions (dev/10, 69, 85, 130, 143), and changed 3 others
  that stayed wrong. Hard questions can't show harm, so it still needs the 200 test.

- **Retrieval gains don't guarantee answer gains:** the blend improved recall yet lost 3 net answers. Always check end to end.

- **Multi-agent alone isn't enough:** MDocAgent, five fixed agents, scores 63.8 on the leaderboard. Keep only agents that fix hard questions.
- **Unavailable:** `agentic-document-ai/pdfs-to-markdown-mistral-ocr`, referenced by MADQA's RLM baseline, returns 404.

## Environment notes

- GPU work runs in Docker. The Windows-side proxy at 127.0.0.1:3456 drops connections, so downloads and
  API calls go through Docker.
- DocumentCloud blocks fast downloads. Go slowly.
- Command Code's firewall rejects Python's default User-Agent, so set a named one.
- Plan allows `deepseek/deepseek-v4.1-flash`; Gemini models are not on it.

## Test tracker

Move each item to Kept or Tried and not adopted when it finishes. **Status:** Running, Queued, Next,
Planned, Idea. **Hard set:** `experiments/hard_current.json`.

**Rules:** every test finishes in 5 minutes or less (user, 2026-11-03): hard set, or only the questions a change touches (`lab.incremental`), reusing saved candidates; judge with the `madqa-judge` image (23 s). One change at a time. Hard set first, then all 200 once satisfied, then train. Keep a change
only if it fixes questions without breaking others.

### Now

| ID | Test | Targets | Pass if | Needs | Status |
|---|---|---|---|---|---|
| T3 | Confirm the adjudicated stack on 200 train questions at test mix (`experiments/train_mix200.json`) | Overfitting | Adjudicated beats verify on train too | T30 passes | **Done: 92.0** vs verify+units 88.0 (10 gained, 2 lost, p = 0.039). Agent alone 88.5; adjudicated without units 91.0. Dev 91.0, so no overfitting |

### Reading: the biggest limit

| ID | Test | Targets | Pass if | Needs | Status |
|---|---|---|---|---|---|
| T4 | Specialized agents by question type | Format and reasoning errors | Fixes several of the 20 hard leftovers | Nothing | Next |
| T5 | Evidence workspace: fact table with sources, then answer | Wrong rows, comparisons | Fixes hard table and comparison questions | Nothing | Next |
| T6 | Qwen 3.8 with thinking when unsure, streaming | Hard reasoning | Fixes what verify misses | Nothing | Planned |
| T7 | Zoom: crop and re-read regions (AgenticOCR's tool design) | Wrong rows, misread characters | Fixes character and row errors | Nothing, unless using AgenticOCR's weights | Planned |
| T8 | Evidence-first: judge pages alone, return nothing if irrelevant | Distractor errors, speed | Better answers or fewer images | Nothing | Planned |
| T9 | Images at about 2,500 pixels | Small text | Beats 2,048 | Nothing | Idea |
| T10 | Vote across open models | Random errors | Beats the best single model | Downloads, user OK | Idea |
| T11 | Train Qwen on the 1,550 train questions | Reading overall | Beats base model on unseen questions | Larger effort | Idea |

### Search and evidence

| ID | Test | Targets | Pass if | Needs | Status |
|---|---|---|---|---|---|
| T12 | Agent with retry on empty answers, hard set | 5 hard questions with no correct page shown | Recovers several | Nothing | **Already in the agent** (`different_search`, `lab/agent.py`); hard set 5 of 26 fixed (see results table below) |
| T13 | Route to the agent only with 2+ planned searches | Cross-document | Keeps 77.8, single-page unharmed | Nothing | **Not adopted** (offline, saved dev answers, judge cache, 2026-11-06): 33 of 200 plan 2+ searches. Multi-search: agent 22, verify 27, final 30 right. Single-search (167): agent 153, verify 153, final 157. Routing saves time but the agent's single-search answers still feed the adjudicator's +4; speed-only option |
| T14 | Read more of the document for lists and counts | 7 partly-shown questions, cross-page | Raises cross-page above 73.5 | Nothing | Planned |
| T15 | Adaptive page count per question (ViDoRAG) | Long lists, speed | Same accuracy or better, fewer images | Nothing | Planned |
| T16 | Better OCR: GLM-OCR, on hard-question pages | Search text | More evidence found | Download approved 2026-11-03 | Planned |
| T17 | Layout parsing: MinerU, on hard-question pages | Table structure | Better extraction than current | Download approved 2026-11-03 | Planned |
| T18 | Visual search: ColQwen alongside BM25 | Pages keywords miss | Finds new evidence, and answers improve | Download, **user OK** | **Done, adopted** (dev +2/-0, train +30/-3) |
| T19 | Doc2query: generated questions in search text | Missing evidence | Recovers hard questions | Nothing, one long run if it passes | Planned |
| T20 | Page events and entities with links (SAG 1) | Multi-step search | Recovers hard questions | Nothing, one long run if it passes | Planned |
| T21 | Proper SAG2 expansion: judge each round, earlier evidence visible | Multi-step search | Beats the simple version | Nothing | Planned |
| T22 | Document structure navigation (PageIndex) | Right document, wrong page | Fixes those misses | Nothing | Idea |
| T23 | Kev-4B 16-bit vs current reranker | Ranking | Better answers, not just recall | Stage not written | Planned |
| T24 | Check whether document context helps the final stack | Simplify | Removed if no effect | Nothing | Idea |

### Automation, speed and submission

| ID | Test | Targets | Pass if | Needs | Status |
|---|---|---|---|---|---|
| T25 | LLM auto-tuning of agents on train samples | All | Beats hand-written agents on unseen questions | T4 first | Planned |
| T26 | Generated questions for scarce multi-step types | Tuning data | Helps T25 | T25 first | Idea |
| T27 | Trim blank page margins | Speed | Same accuracy, fewer tokens | Nothing | Idea |
| T28 | Full stack together: all winners, on all 200 then train | Overlap between parts | Beats best single setup | After T2 to T14 | Planned |
| T29 | Test-set run and leaderboard submission | Real score | Valid file, official number | **User approval** | Planned |

### Hard-set results so far, 26 questions

| Method | Fixed | Which |
|---|---:|---|
| Agent with retry | 5 | 69, 87, 104, 127, 143 |
| Agent + verify | 6 | 57, 87, 104, 127, 130, 143 |
| Verify | 5 | 10, 69, 85, 130, 143 |
| Evidence workspace | 4 | 57, 69, 85, 130 |
| Workspace + verify | 4 | 57, 69, 85, 130 |
| Router + workspace | 3 | 10, 69, 143 |
| Specialist router | 2 | 57, 69 |

Together they fix 9 different questions, which is why T30 adjudicates between them.

| ID | Test | Targets | Pass if | Needs | Status |
|---|---|---|---|---|---|
| T30 | Adjudicate: read every method's cited pages when they disagree | Different methods fix different questions | Fixes more than any single method, then holds on 200 | Nothing | **Kept**: hard set 9 of 26; all 200 87.5 to 90.5, 0 lost |

| T31 | Trigram substring search added to the pool before reranking | Merged OCR words (dev/120) | Ranking recall up on dev and train, then answers on changed questions | Nothing | **Tie, not adopted**: recall@9 93.58 to 94.35, @3 81.31 to 80.98; 29 of 200 changed pages; base read on those 23 vs 23 (fixed 104, 120; broke 26, 51), Page F1 81.65 to 80.34 |

| T7 | Zoom: cited pages (up to 3) re-read at 2,800 px as an extra candidate (`lab/zoom.py`) | Misreads | Fixes hard questions, then no harm on 200 | Nothing | **Not adopted**: hard sets 2 of 31 alone, but on all 200 dev as a 4th candidate 91.0 vs 91.0, nothing changed; zoom alone 86.0; 12 min |
| T10 | Gemma 4 31B NVFP4 (`RedHatAI/gemma-4-31B-it-NVFP4`, 23.3 GB, Apache-2.0) as a second reader | Different mistakes from Qwen | Fixes hard questions as a candidate | GPU swap (`--max-model-len 16384`: 32k needs 7.1 GiB KV, only 6.5 free; 1,120 tokens per image) | **Worked (dev 92.0, train 92.5) but not used: user chose Qwen only (2026-11-03).** Hard sets: alone 4 of 31 (dev/112, 119, train/353, 1115); 4-way adjudication +1 train (353), +0 dev. 0.178 q/s |

### Decided against

Don't add these without new evidence: calibrated abstention (refusals score as wrong), maths in code as
built, GraphRAG, extra benchmarks (MMDocIR, UniDoc-Bench), correcting labels, Haiku, ColBERTv2 before
visual search.

| Dynamic few-shot (4 similar train Q/A in reader prompt, `lab/fewshot.py`, `--rules v3_fs`) | Not adopted | 300 fresh train, single read: 276 -> 274 (+2/-4), 31 answers changed. Examples did not fix format errors and cost some right answers. |

| Qwen3.8-Max (API) as reader | Cancelled by user | Rule: local Qwen3.8-27B only. Hit the API 5-hour limit before any result. |
| Qwen3.8-27B thinking mode as the reader (local, `--thinking`, 10,000 tokens, k9, v3) | Dev read alone 181 vs 185 for the plain verified read (+5/-9) | As a fourth adjudicator candidate: dev 189 -> 189 (6 answers changed, +0/-0). It reads dev/18 and dev/119 right, but the adjudicator does not pick them | ~35 min for dev | Not adopted | Thinking makes the reader worse alone and adds nothing as a candidate |
| Deep read also for list questions with a comparison (`lab.docread --allow-compare`) | Only 5 such questions exist across dev, extra300, lists60 and mix200 (4 train, 1 dev) | Train 3 -> 4 right (train/10 fixed, nothing lost) | minutes | Flag kept, off by default | Too few questions to measure; not the cause of the remaining list failures |
| Collect-scan deep read (`lab.spread` flags questions needing every matching item, 33 of 760; `lab.docread --collect-only --scan 30`: also reads the top 30 ranked pages one by one across documents) | Base on the 33 flagged: dev 13 of 16, lists60 7 of 8, mix200 4 of 8, extra300 0 of 1 | Plain check: +3/-10 (dev +2/-4, lists60 0/-2, mix200 0/-2, extra300 +1/0). Qualifier-aware check at 0.4 (`--qualified --check 0.4`): +3/-4 (dev +2/-4, all train sets 0 lost, extra300 +1) | ~20 min per run | Not adopted | Recovers missing items (dev/23, dev/56, train/10) but adds items from unrelated pages to answers that were already complete (dev/1 got 7 more states, train/38 got Labor Day). Trigger by ranking shape alone fires on 40 to 70% of questions, so the classifier is used instead |
| Collect-scan with two-pass agreement (`lab.docread --collect-only --scan 30 --agree --qualified --check 0.4`: a second pass over the same pages, reversed, batches of 2; an added item stays only if both passes find it) | Same 33 flagged questions: base dev 13 of 16, lists60 7 of 8, mix200 4 of 8, extra300 0 of 1 | Dev +2/-0 (dev/23, dev/56; 189 -> 191), lists60 0/0, mix200 +1/-1 (train/1384 gained, train/38 lost), extra300 +1/0 (train/10). Pooled +4/-1 | ~25 min | Optional last stage; not in the headline | Untuned train is only +2/-1 over 17 flagged questions, too few to call solid. The second pass removes the plain scan's over-collection. Needs a fresh, larger list sample before it goes into the headline |
| Fresh validation of the collect-scan with agreement: 50 train questions never used before (`experiments/train_lists50.json`, 2+ gold items), full stack then `--collect-only --scan 30 --agree --qualified --check 0.4` | Full stack 47 of 50 right (94.0) | Scan changed 2 answers, both lost (train/670, train/1208): +0/-2, 45 of 50 | ~1.5 h | **Not adopted** | Pooled over all untuned train: +2/-3; dev +2/-0. The dev gain does not carry over. Headline stays dev 189 |
| Answer-type rule in the reader prompt (`--rules v3_id`: "the number of / assigned to / what is the number" + a named item asks for its identifier, not a quantity), single read on every question with that wording: dev 8, train 74 | v3 read: dev 7 of 8, train 71 of 74 | Dev 8 of 8 (+1: dev/119, the targeted miss), train 71 of 74 (+1 train/1256, -1 train/1077), 3 answers changed | minutes | Not adopted | Fixes the dev miss it was written for; neutral on 74 untuned train questions. Dev gain is the tuned-on kind |
| Train-gold page prior (pages that other train questions used as evidence for the same document, as a retrieval or tiebreak signal; leave-one-out on train) | 34% of dev gold pages are also train-gold pages (91 of 268), but a document has a median of 1 such page | Swapping one prior page of the top-ranked document into the reader's 9 pages: page recall up for 1 and down for 5 dev questions (68 changed); up 6, down 14 on 610 untuned train questions (241 changed). As a tiebreak it does not separate: dev/98's wrong cited page is in the prior, the gold page is not. Only 4 of the 25 gold pages missing from the top 9 are in the prior | Offline, seconds | Not adopted | The prior is informative about which pages get asked about, but not specific enough to beat the text and ColQwen ranking |
| Hybrid read (`lab/hybrid.py`): top 3 pages as images plus OCR text of about 11 more pages of the same documents (50,000 characters) in one request, so the reader sees a whole spread of pages at once | 16 dev questions the collect classifier flags: final stack 13 of 16 | Hybrid alone 11 of 16: +0, -2 (dev/57, dev/77); did not fix dev/23 or dev/56; dev/15 hit the context limit (HTTP 400) | ~3 min | Not adopted; not run on train | More pages in view does not make the reader list every item; extraction page by page was the only thing that recovered them, and it adds wrong items |
| Quote grounding on top of the two-pass scan (`lab.docread --ground`: before an added item is kept, the model must quote words from its page that show it meets every condition, and the quote must appear in the page text) | Same flagged questions plus the 50 fresh train questions: stack base dev 13 of 16, lists60 7 of 8, mix200 4 of 8, extra300 0 of 1, lists50 47 of 50 | Dev +1/-2 (gained dev/23; lost dev/22, dev/26; dev/56 no longer recovered), lists60 0/0, mix200 0/-1 (train/38), extra300 0/0, lists50 0/-2 (train/670, train/1208). Pooled +1/-5, worse than agreement alone (+4/-3) | ~40 min | Not adopted | Grounding drops real items (dev/56) and still lets the wrong ones through: the model can always quote words from the page that look like support. Same wrong items as before were added |

## 2026-11-09 Retrieval ceiling for multi-page questions (no model judgement involved)
- Share of questions whose gold pages are ALL inside the reader's pages: single-page train 97.6% (1282/1313) at 9, multi-page train 72% (171/237) at 9, 86% at 40; dev multi 27/37 at 9, 32/37 at 40.
- Re-ordering by document (best document first, or best document's pages first) hurts: multi 171 -> 152 / 97 on train, single 1282 -> 1241 / 924. Gold pages of multi questions are often in different documents; the existing ranking (LLM select + ColQwen swap) is already better than these rules.
- Conclusion: not adopted. Any further retrieval gain needs a learned ranker (train gold as labels), not a hand rule.

## 2026-11-09 Slot filling (lab/slots.py): look up named parts the answer lacks
- Text step lists the values the answer must contain (one per named thing); a slot is covered if its words are in the answer/quote; only uncovered slots get a targeted search (3 pages) and a compose step.
- First prompt (any named entity = slot): dev 189 -> 188, lost dev/101. Cause: yes/no and "which of A or B" questions name entities that are not parts of the answer. Fixed in the prompt (slots only when the answer needs one value per thing).
- Fixed prompt: dev 189 -> 189 (4 questions get slots, 0 changed); train mix200 0 slots, extra300 5 slots 0 changed, lists60 4 slots 1 changed (lost train/1179), lists50 2 slots 0 changed. Pooled +0/-1.
- Result: not adopted. Only ~2% of questions name countable slots, and the misses we have (dev/15, 18, 23, 56) are not among them: open lists and conditions have no slots to enumerate.

## 2026-11-09 Page notes (lab/notes.py): pages 6-30 read alone into text notes, one final read
- Top 5 pages as images + notes (quoted facts, no answer) from the next 25 pages, one final read with RULES_V3. Run on the 37 dev questions with several gold pages (selected by gold, so favourable to the method). 2065 s.
- Result: answer changed on 11; right 29 -> 24: gained dev/112, lost dev/10, 69, 85, 139, 143, 155 (+1/-6). Not adopted, no train run needed.
- Likely cause: the final read drops the verify/adjudicate/units steps, and 25 pages of notes (most of them on-topic but not the answer) pull it toward wrong items. Same pattern as every other recall fix: more material in, more wrong answers out.

## 2026-11-09 Learned page ranker (lab/pagerank.py): choose the reader's 9 pages
- HistGradientBoosting on rank/doc-context features (text rank, ColQwen rank, document's best rank, same-document, distance to top page...), labels = gold pages of train. Train scores are out-of-fold (5-fold by question).
- All gold pages inside the 9: train single 1282 -> 1297 of 1313, train multi 171 -> 195 of 237 (5-fold); dev single 160 -> 159, dev multi 27 -> 30.
- Read only (rules v3, no later steps), dev: 183 -> 184 (+4/-3).
- Read only, train, 207 questions (57 whose gold coverage changed + 150 random of the 1458 whose 9 differ but coverage is the same): 168 -> 185 (+24/-7). Coverage-changed group +20/-4; same-coverage sample +4/-3 (noise).
- Status: promising; needs the full pipeline (verify, workspace, agent, adjudicate, units) before adopting. experiments/thin_ocr_ranker.json, runs ranker_dev / ranker_train.

## 2026-11-09 Learned page ranker, full pipeline (read, verify, workspace, adjudicate, units; agent answers shared)
- Dev, 9 questions where the ranker changes gold coverage of the 9 pages: right 3 -> 4 (gained dev/74, lost none). Dev would be 190/200.
- Untuned train, 52 questions where the ranker changes gold coverage (out-of-fold ranker scores): right 34 -> 41 (+11 / -4). Gained train/337, 414, 969, 1219, 1223, 1324, 1335, 1373, 1374, 1419, 1523; lost train/499, 656, 670, 726.
- The 1,458 train questions whose 9 pages differ but gold coverage is the same were neutral in the plain-read sample (+4/-3 of 150).
- Expected gain: about +7 net per 52 coverage-changed questions, i.e. roughly +0.5 point over the full question set. Kept: holds on untuned train.
- Not yet wired for test: the ranker needs the text and ColQwen rankings for the test split (lab.pagerank exports dev and train only). Nothing run on test.
