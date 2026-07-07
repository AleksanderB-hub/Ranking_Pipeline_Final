On the few_shot_5_3_gold_2_zero_data_even vs original split (the same few shot composition)

This is a question the three-variant design was built to answer, and your numbers decompose very cleanly. The direct answer first: within the NDCG/MAP metrics, no — the gate has no benefit beyond correct silence on no-gold sentences, and it cannot have one by construction. But two important nuances: that benefit is bigger than "free 1.0s" suggests, and the gate has a second, separate benefit that lives in the P/R/F1 metrics, not the ranking ones.

## Why the gate can never help e2e or given_cand
The LLM's predictions are identical with and without the gate — gating only masks rejected sentences to empty lists. On a gold-bearing sentence, an empty list scores NDCG = 0, and any non-empty list scores ≥ 0. So on the population e2e and given_cand are computed over (gold-bearing sentences only), the gate can only do two things: leave a sentence unchanged (true accept) or zero it out (false reject). It is structurally impossible for it to improve those variants.

Your numbers confirm this: gated e2e ≈ no-gate e2e × gate recall — 0.5113 × 0.8432 = 0.4311, and you measured 0.4381. The small surplus just means the ~148 gold sentences the gate falsely rejected were ones the LLM ranked slightly worse than average anyway. So yes: any gate benefit is invisible in e2e/given_cand; only its cost (false rejections) shows there.

## The deploy benefit is real, not merely mechanical
"Just getting 1.0 for correctly identified irrelevant sentences" undersells what's happening, because the counterfactual isn't 1.0 — it's what the ungated LLM actually does on no-gold sentences, which is poor. You can back this out of your own numbers. Deploy is a mixture: gold-bearing sentences contribute their e2e score, no-gold sentences contribute their silence rate (1.0 if nothing predicted, 0.0 otherwise). Solving your even-split numbers:

No gate: (944 × 0.5113 + 944 × s) / 1888 = 0.3732 → silence rate s ≈ 0.235. The ungated LLM stays correctly silent on only ~23% of irrelevant sentences — consistent with its avg_preds of 13.4.
Gated: (944 × 0.4381 + 944 × s′) / 1888 = 0.6793 → s′ ≈ 0.920 — which matches the gate's confusion matrix exactly (it rejects ~866 of 944 no-gold sentences, 0.917).
So the gate lifts correct-silence on irrelevant content from ~0.23 to ~0.92, at the cost of zeroing ~16% of gold sentences. That trade is the entire gate story in ranking terms, and deploy is the only variant that prices both sides of it.

The 0.7483 → 0.6793 drop is composition, not degradation
Run the same decomposition on your original-split gated run: (944 × 0.4379 + 1697 × s) / 2641 = 0.7483 → s ≈ 0.921. Identical per-class behaviour to the even split (e2e 0.4379 vs 0.4381, silence 0.921 vs 0.920). Nothing got worse — the even split just shifted the mixture from 64% no-gold (where the gated pipeline scores ~0.92) to 50% no-gold, giving more weight to the harder gold-bearing class (~0.44). This is exactly the class-prior sensitivity the --data-even caveat warned about, and it's actually a nice validation that your pipeline's per-class performance is stable across priors.

## The gate's second benefit: FP suppression in the set metrics
Beyond ranking, the gate does something the deploy variant doesn't capture in the same terms: it deletes false-positive skill predictions. That's why all-regime precision jumps 0.0879 → 0.1428 and F1 0.1494 → 0.2148 while avg_preds halves (13.40 → 7.17). This isn't a per-sentence 1.0 bonus — it's removing ~866 no-gold sentences' worth of spurious skills from the prediction pool. The cost side appears as recall 0.4979 → 0.4331 (≈ × the gate's 0.8432 recall). Meanwhile skill_bearing F1 tells the same story as e2e: 0.2325 → 0.2292, slightly worse gated, because on gold sentences the gate can only remove.

One last practical benefit that no evaluation metric here captures: in deployment you wouldn't run the LLM on gate-rejected sentences at all (this eval runs it on everything so it can derive both pipelines by masking) — so the gate would also cut ~46% of LLM calls on your even split, and ~65% on the natural distribution.