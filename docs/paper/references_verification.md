# Bibliography verification log

Every entry in `neurips_benchmark/references.bib` was checked by web search
before it was allowed into the file. This log records what was checked, where
it was confirmed, and what was corrected. Checked 2026-09-19.

**The rule applied throughout:** a field that could not be confirmed against
arXiv, the publisher, the proceedings, or the project's own citation metadata
was **removed**, not guessed. Four entries in the previous draft carried
`% VERIFY` flags; all four are resolved below and the flags are gone.

| key | verdict | confirmed against | change made |
|---|---|---|---|
| `jimenez2024swebench` | confirmed | arxiv.org/abs/2310.06770 — ICLR 2024, author list matches | none |
| `chan2024mlebench` | **flag resolved** | arxiv.org/abs/2410.07095 — "published as a conference paper at ICLR 2025" on the PDF itself | `% VERIFY` removed; ICLR 2025 confirmed, not inferred |
| `huang2024mlagentbench` | confirmed + enriched | proceedings.mlr.press/v235/huang24y.html; dblp `conf/icml/HuangVLL24` | added series PMLR, vol. 235, pp. 20271–20309 |
| `jing2024dsbench` | **upgraded** | github.com/LiqiangJing/DSBench (ICLR 2025 tag); arXiv 2409.07703 | `@article` preprint → `@inproceedings`, ICLR 2025 |
| `chen2025scienceagentbench` | confirmed + enriched | arXiv PDF header "Published as a conference paper at ICLR 2025"; OSU publication record | added pp. 12138–12194 |
| `wijk2024rebench` | confirmed as preprint | arxiv.org/abs/2411.15114; metr.org blog 2024-11-22 | none (no venue found — stays a preprint) |
| `siegel2024corebench` | **upgraded** | openreview.net/forum?id=BsMMc4MEGS; Princeton record gives *TMLR*, January 2025 | preprint → TMLR 2025 |
| `starace2025paperbench` | confirmed as preprint | arxiv.org/abs/2504.01848 | none |
| `mitchener2025bixbench` | confirmed as preprint | arxiv.org/abs/2503.00096; github.com/Future-House/BixBench | none |
| `majumder2025discoverybench` | confirmed | proceedings.iclr.cc paper file for 2025; arXiv 2407.01725 | none |
| `liu2024agentbench` | confirmed | arxiv.org/abs/2308.03688; github.com/THUDM/AgentBench ("ICLR'24") | none |
| `bragg2025astabench` | **new**, confirmed as preprint | arxiv.org/abs/2510.21652; allenai.org/asta/bench | added for the 2026 positioning re-check. Only the first two authors were confirmed, so the entry uses `and others` rather than inventing a list |
| `zhang2026harness` | **new**, confirmed as preprint | arxiv.org/abs/2605.23950 — full six-author list and abstract read | added; it is the position paper that argues the harness, not the model, binds agent performance — directly relevant to our model-controlled design |
| `hansen2024tokamaker` | confirmed | arxiv.org/abs/2311.07719; sciencedirect S0010465524000341; OSTI 2311941 | none — CPC 298:109111, DOI 10.1016/j.cpc.2024.109111 all match |
| `grad1958hydromagnetic` | confirmed | IAEA proceedings scan (`Paper25_Vol31.pdf`); OSTI/ETDEWEB 21068316 | none — Geneva 1958, vol. 31 *Theoretical and Experimental Aspects of Controlled Nuclear Fusion*, pp. 190–197 |
| `shafranov1958magnetohydrodynamical` | confirmed | jetp.ras.ru scan `e_006_03_0545.pdf`; ADS 1958JETP....6..545S | none — Sov. Phys. JETP 6:545–554, 1958 |
| `akiba2019optuna` | confirmed | Semantic Scholar record; KDD 2019 | none — pp. 2623–2631 |
| `khattab2024dspy` | confirmed | proceedings.iclr.cc 2024 paper file; arXiv 2310.03714 | none |
| `agrawal2025gepa` | **flag resolved** | arXiv PDF cover page: "Accepted at ICLR 2026 (Oral)" | `@article` preprint → `@inproceedings` ICLR 2026, Oral noted; `% VERIFY` removed |
| `langchain2022` | **flag resolved** | github.com/langchain-ai/langchain `CITATION.cff`, author Harrison Chase, dated 2022-10-17 | kept as `@misc` with the CITATION.cff provenance stated in the note. **No LangGraph paper exists**; the text now says "LangGraph, part of the LangChain project" rather than implying a separate publication |
| `ursa2025` | **flag resolved** | arxiv.org/abs/2506.22653 — v1 submitted 27 June 2025, 13 authors, LANL | year 2025 confirmed from the v1 submission date; `% VERIFY` removed |
| `mckay1979lhs` | **new**, confirmed | Technometrics 21(2):239–245; the LHS origin paper | added for the space-filling design vocabulary |
| `settles2009active` | **new**, confirmed | minds.wisconsin.edu/handle/1793/60660 — CS Tech. Report 1648, UW–Madison, 2009 | added for the active-learning positioning paragraph |
| `chaloner1995bayesian` | **new**, confirmed | projecteuclid.org, Statist. Sci. 10(3):273–304, DOI 10.1214/ss/1177009939 | added for Bayesian experimental design |
| `wilson1927probable` | confirmed | tandfonline 10.1080/01621459.1927.10502953; JSTOR 2276774 | none — JASA 22(158):209–212 |
| `vanelteren1960combination` | **new**, confirmed | SAS KB 25022 and the Stata `vanelteren` module doc, both citing van Elteren, P.H. (1960), *Bull. Int. Statist. Inst.* 37:351–361 | added — this is the primary test of the paper and previously had no citation |
| `efron1979bootstrap` | **new**, confirmed | projecteuclid 10.1214/aos/1176344552 — Ann. Statist. 7(1):1–26 | added for the percentile bootstrap |

## Deliberately not cited

- **Burger et al., "A mobile robotic chemist" (Nature, 2020)** — a natural
  citation for the self-driving-laboratory sentence in §2, but the full author
  list could not be read from an accessible source (the Nature page is behind
  an authentication redirect and the repository PDF would not convert). Rather
  than ship a partial author list, the sentence cites
  [`settles2009active`] and [`chaloner1995bayesian`] instead, which carry the
  same point — that choosing where to measure next is a long-studied problem —
  and are fully verified.
- **A separate LangGraph publication** — searched for and not found. The
  LangChain repository's `CITATION.cff` is the only citable metadata.

## Positioning re-check (the §2 FILLBOX)

The "to our knowledge no existing agent benchmark requires the agent to drive
an expensive solver, allocate a hard budget across adaptive rounds, and be
scored on a frozen set it never sees" claim was re-checked against 2026
releases. The closest new work found:

- **AstaBench** [`bragg2025astabench`] — 2400+ problems across the scientific
  discovery process, and notably motivated by the same complaint about
  confounders (model cost, tool access) that motivates our model control. It
  supplies tools and data; it does not make the acquisition policy the scored
  object.
- **Stop Comparing LLM Agents Without Disclosing the Harness**
  [`zhang2026harness`] — argues the execution harness is often a stronger
  determinant of performance than the model. This is a position paper, not a
  benchmark, and our campaign is a direct empirical test of it: one model, four
  harnesses, measured spread.

Neither displaces the claim, which is retained and narrowed in the text.

## Added for the journal version (checked 2026-09-20)

Two citations were added to §2 so that the motivation for a Grad–Shafranov
surrogate rests on the live literature rather than on assertion. Both were
verified by fetching the arXiv abstract page directly, not from a search
summary.

| key | verdict | confirmed against | fields |
|---|---|---|---|
| `krastev2026millisecond` | confirmed as preprint | arxiv.org/abs/2608.05555 | *Millisecond-Scale Neural Operator Surrogates for Double-Null Free-Boundary Grad–Shafranov Equilibria*; sole author Plamen G. Krastev (Harvard); submitted 6 August 2026 |
| `grandin2026experimental` | confirmed as preprint | arxiv.org/abs/2606.09487 | *Experimental validation of a fast control-oriented, physics-informed surrogate model for plasma equilibrium reconstruction in the TCV tokamak*; M. Grandin, A. Mele, C. Galperti, D. Gonzales Castineiras, C. Heiß, A. Cenedese, with the TCV team and the EUROfusion Tokamak Exploitation team; submitted 8 June 2026 |

The full author lists were read from the abstract pages rather than inferred;
the two collaboration authorships are recorded in the `note` field rather than
dropped, because omitting them would misstate the authorship.

Also verified in passing while writing §2, and used without citation because
they are textbook material rather than results: the Grad–Shafranov operator
$\Delta^{*} = \partial_{RR} - R^{-1}\partial_R + \partial_{ZZ}$, and that
contours of $\psi$ are the magnetic flux surfaces with the outermost closed
one the plasma boundary. The equation itself is cited to Grad & Rubin (1958)
and Shafranov (1958), both already verified above.
