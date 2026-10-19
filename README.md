[README.md](https://github.com/user-attachments/files/33665077/README.md)
# referential-choice-and-common-ground
A computational investigation of how event predictability and shared knowledge influence referential expression choice, using probabilistic modeling and statistical analysis.
# When Are Pronouns Too Risky?

### A computational study of event predictability, common ground, and referential choice

In conversation, speakers often choose between a short referring expression (such as a pronoun) and a more explicit description. This project explores how that choice might change when events are predictable or surprising, and when a listener does or does not share the speaker's expectations.

This is a **simulation-based study**, not a human-participant experiment. It explores the behavior of a deliberately simple probabilistic speaker-listener model; its outputs should not be treated as observations of human linguistic behavior.

## Research question

**Under what conditions does event predictability make explicit reference more useful, and how does this depend on shared knowledge between interlocutors?**

## Design and approach

A 2 × 2 within-agent simulation crosses two factors:

| Factor | Conditions |
| --- | --- |
| Event status | Predictable / surprising |
| Listener knowledge | Shared expectations / unshared expectations |

- **Dataset:** 2,304 synthetic trials (48 virtual agents × 12 stimulus IDs × 4 conditions).
- **Response:** Explicit noun phrase (1) versus reduced expression (0).
- **Speaker-listener model:** Listener inference is updated through a noisy identifying signal; choice is influenced by communicative success and an utterance-cost parameter.
- **Analyses:** Condition summaries, adjusted logistic regression, participant-block bootstrap intervals, and a noise-sensitivity check.

## Illustrative outputs

In the simulated data, the model selected explicit noun phrases in **74.8%** of shared-knowledge / surprising-event trials, compared with **30.7%** of shared-knowledge / predictable-event trials. When knowledge was not shared, both rates were close to **50%**. These values reflect the assumptions programmed into the model rather than empirical evidence for a new linguistic effect.

![Model-generated reference choices](figures/condition_means.png)

## Repository contents

```text
.
├── README.md
├── requirements.txt             # Python dependencies
├── .gitignore
├── Independent_Research_Report.pdf
├── code/
│   ├── build_simulation.py       # Generates synthetic observations and Python analysis
│   ├── analyze_in_R.R            # R analysis of the supplied simulated dataset
│   └── make_report.py           # Optional PDF/figure regeneration (see note below)
├── data/
│   └── synthetic_reference_trials.csv
├── results/                      # Outputs from the executed Python analysis
└── figures/                      # Figures used in the report
```

## Reproducibility

The included Python script generated the dataset and reported numerical outputs. From the repository root, use:

```bash
python code/build_simulation.py
```

Install the Python dependencies with `pip install -r requirements.txt`. The main simulation uses `numpy`, `pandas`, `scipy`, and `statsmodels`.

To regenerate the PDF and figures in a Linux environment with Lato fonts and the optional Python packages `matplotlib`, `Pillow`, and `reportlab` installed, run `python code/make_report.py`. The report is already included, so this step is not necessary to inspect the results.

An equivalent R analysis workflow is also provided:

```r
source("code/analyze_in_R.R")
```

The principal R analysis uses base R. Optional packages `sandwich` and `lme4` enable additional robustness checks. **The R script has not yet been executed or validated in R**; reported figures and estimates come from the executed Python analysis. Small differences in resampling intervals and standard-error conventions may arise if the R implementation is run.

## Scope and limitations

- All observations are **synthetic**; no participant recruitment or speech collection took place.
- The model uses fixed illustrative parameters rather than values estimated from linguistic data.
- There are only two expression categories, and the unshared-knowledge case imposes a neutral listener prior.
- Statistical significance within generated samples is **not evidence** that actual speakers behave this way.

See the [research report](Independent_Research_Report.pdf) for the model equations, analysis, limitations, and detailed figures.

## Selected reading

- Achimova, A., Stegemann-Philipps, C., Winkler, S., & Butz, M. V. (2022). *Anaphoric reference in descriptions of surprising events*. Proceedings of Sinn und Bedeutung 26, 20–34. [Paper](https://asya-achimova.github.io/files/2022_Achimova_et_al_Anaphoric_reference.pdf).
- Arnold, J. E. (2001). The effect of thematic roles on pronoun use and frequency of reference continuation. *Discourse Processes, 31*(2), 137–162. [DOI](https://doi.org/10.1207/S15326950DP3102_02).
- Clark, H. H., & Brennan, S. E. (1991). Grounding in communication. In *Perspectives on Socially Shared Cognition*. [DOI](https://doi.org/10.1037/10096-006).
- Frank, M. C., & Goodman, N. D. (2012). Predicting pragmatic reasoning in language games. *Science, 336*(6084), 998. [DOI](https://doi.org/10.1126/science.1218633).

**Development transparency:** This initial project package, documentation, and code were prepared with AI assistance. The included Python implementation was executed; independent review, adaptation, and R validation remain important before claiming it as fully self-conducted research.
