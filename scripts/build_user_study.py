"""Build the blind user study: 30 cross-reference cases, three models' outputs in random order.

    python -m scripts.build_user_study --models gan flow controlnet

Writes to results/user_study/:
    study.html   self-contained form for raters (images embedded; no model names anywhere in it).
                 Raters rank the three options on each criterion and click "Download answers".
    key.csv      case -> option -> model. Keep it away from raters.
Collect the downloaded CSVs into results/user_study/responses/ and run scripts.analyze_user_study.
"""

import argparse
import base64
import csv
import html
import io
import json
import random
from pathlib import Path

from PIL import Image

from src.metrics.files import index_by_id, index_outputs
from src.utils.seeding import SEEDS

CRITERIA = {
    "color_match": "Which coloring best matches the reference's colors?",
    "overall": "Which result looks best overall (clean, on the lines, no artifacts)?",
}


def data_uri(path: Path) -> str:
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Colorization Study</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto; padding: 16px; background: #fafafa; color: #222; }}
 .case {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 16px 0; }}
 .row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
 figure {{ margin: 0; text-align: center; }}
 figure img {{ width: 180px; height: 180px; image-rendering: auto; border: 1px solid #ccc; }}
 .given figcaption {{ color: #666; }}
 table {{ border-collapse: collapse; margin-top: 8px; }}
 td, th {{ padding: 4px 10px; text-align: center; }}
 .bad {{ outline: 2px solid #d33; }}
 #bar {{ position: sticky; top: 0; background: #fafafa; padding: 8px 0; border-bottom: 1px solid #ddd; }}
</style></head><body>
<div id="bar">Rater name: <input id="rater" placeholder="your name or alias">
<button onclick="download()">Download answers</button> <span id="msg"></span></div>
<h1>Anime line-art colorization study</h1>
<p>Each case shows a <b>sketch</b>, a <b>color reference</b> of the same character, and three colorings.
For each question rank the three options: 1 = best, 3 = worst. Each rank is used once per question.</p>
{cases}
<script>
const CASES = {case_ids}, CRITERIA = {criteria}, KEY = "colorization-study-v1";
const rater = document.getElementById("rater"), msg = document.getElementById("msg");
function selects() {{ return [...document.querySelectorAll("select")]; }}
function save() {{ try {{ const s = {{rater: rater.value}}; selects().forEach(e => s[e.name] = e.value);
  localStorage.setItem(KEY, JSON.stringify(s)); }} catch (e) {{}} }}
function load() {{ try {{ const s = JSON.parse(localStorage.getItem(KEY) || "{{}}"); if (s.rater) rater.value = s.rater;
  selects().forEach(e => {{ if (s[e.name]) e.value = s[e.name]; }}); }} catch (e) {{}} }}
function download() {{
  let ok = true, rows = ["rater,case,criterion,option,rank"], name = rater.value.replace(/[",\\n\\r]+/g, " ").trim();
  if (!name) {{ msg.textContent = "Please enter a rater name."; return; }}
  for (const c of CASES) for (const k of Object.keys(CRITERIA)) {{
    const sel = [1, 2, 3].map(o => document.querySelector(`select[name="${{c}}|${{k}}|${{o}}"]`));
    const ranks = sel.map(s => s.value);
    const valid = ranks.every(r => r) && new Set(ranks).size === 3;
    sel.forEach(s => s.classList.toggle("bad", !valid));
    if (!valid) ok = false; else sel.forEach((s, i) => rows.push([name, c, k, i + 1, ranks[i]].join(",")));
  }}
  if (!ok) {{ msg.textContent = "Some cases are incomplete or reuse a rank (outlined in red)."; return; }}
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\\n") + "\\n"], {{type: "text/csv"}}));
  a.download = `responses_${{name.replace(/[^\\w-]+/g, "_")}}.csv`; a.click();
  msg.textContent = "Downloaded. Please send this file back.";
}}
document.addEventListener("change", save); rater.addEventListener("input", save); load();
</script></body></html>
"""


def case_html(case: int, sketch: Path, reference: Path, options: list[Path]) -> str:
    figs = [f'<figure class="given"><img src="{data_uri(sketch)}" alt="sketch"><figcaption>sketch</figcaption></figure>',
            f'<figure class="given"><img src="{data_uri(reference)}" alt="reference"><figcaption>reference</figcaption></figure>']
    figs += [f'<figure><img src="{data_uri(p)}" alt="option {i}"><figcaption>Option {i}</figcaption></figure>'
             for i, p in enumerate(options, 1)]
    head = "".join(f"<th>Option {i}</th>" for i in range(1, len(options) + 1))
    rows = ""
    for key, question in CRITERIA.items():
        cells = "".join(
            f'<td><select name="{case}|{key}|{i}"><option value=""></option>'
            + "".join(f'<option value="{r}">{r}</option>' for r in range(1, len(options) + 1))
            + "</select></td>"
            for i in range(1, len(options) + 1)
        )
        rows += f"<tr><td style='text-align:left'>{html.escape(question)}</td>{cells}</tr>"
    return (f'<div class="case"><h3>Case {case}</h3><div class="row">{"".join(figs)}</div>'
            f"<table><tr><th></th>{head}</tr>{rows}</table></div>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--n-cases", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="cross_ref")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("results/user_study"))
    args = parser.parse_args()

    sample_seed = SEEDS[0]
    sketches = index_by_id(args.eval_dir / "sketch")
    refs = index_by_id(args.eval_dir / f"ref_{args.mode}")
    outputs = {m: index_outputs(args.outputs_dir / m / args.mode) for m in args.models}
    candidates = sorted(i for i in set(sketches) & set(refs)
                        if all(sample_seed in outputs[m].get(i, {}) for m in args.models))
    if len(candidates) < args.n_cases:
        raise SystemExit(f"only {len(candidates)} ids have outputs from every model; need {args.n_cases}")
    rng = random.Random(args.seed)
    chosen = rng.sample(candidates, args.n_cases)

    args.out.mkdir(parents=True, exist_ok=True)
    blocks, key_rows = [], []
    for case, image_id in enumerate(chosen, 1):
        order = list(args.models)
        rng.shuffle(order)
        blocks.append(case_html(case, sketches[image_id], refs[image_id], [outputs[m][image_id][sample_seed] for m in order]))
        key_rows += [{"case": case, "image_id": image_id, "option": i, "model": m} for i, m in enumerate(order, 1)]

    page = PAGE.format(cases="\n".join(blocks), case_ids=json.dumps(list(range(1, len(chosen) + 1))),
                       criteria=json.dumps(CRITERIA))
    (args.out / "study.html").write_text(page, encoding="utf-8")
    with (args.out / "key.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "image_id", "option", "model"])
        writer.writeheader()
        writer.writerows(key_rows)
    (args.out / "responses").mkdir(exist_ok=True)
    print(f"wrote {args.out / 'study.html'} ({len(chosen)} cases) and {args.out / 'key.csv'}")


if __name__ == "__main__":
    main()
