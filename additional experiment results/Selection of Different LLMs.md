### Selection of Different LLMs


In $HBTGen$, LLMs are used solely to instantiate structurally abstracted test cases; they do not participate in oracle design, execution comparison, or post-hoc analysis. We therefore examined whether different LLMs influence the validity and defect-triggering utility of the generated test cases.

Specifically, we compared a locally deployed QWQ-32B and a remotely executed Claude under identical prompt settings and structural templates, each generating 200 test cases. As shown in Table below, both models produced scripts of comparable structural complexity and branch coverage, and each revealed one distinct defect.

Interestingly, despite their different model sizes and instruction-training objectives, the qualitative differences between the two models are modest in our setting. A plausible explanation is that the structural abstractions and history-derived templates already enforce sufficient semantic grounding for exploring compiler behaviors, reducing dependence on the LLM’s generic code-generation capabilities.

Overall, these results suggest that our framework $HBTGen$ exhibits stable behavior under different LLM backends, and its effectiveness does not hinge on specific model capabilities or scale; once structural abstractions are fixed, the choice of generation backend has limited influence on the downstream testing outcome.

**Table: Comparison between QWQ-32B (local) and Claude (remote) over 200 test programs per model**

| Model | #Bugsᵃ | Stability (%) | Def LOC | Def Total API | Def Unique API | Inv LOC | Inv Total API | Inv Unique API |
|------|--------|---------------|---------|---------------|----------------|--------|---------------|----------------|
| QWQ-32B (local) | 1 | 92.87 | 17.86 | 4.10 | 3.26 | 20.29 | 4.98 | 4.34 |
| Claude (remote) | 1 | 93.16 | 17.92 | 4.02 | 3.18 | 21.01 | 5.12 | 4.78 |

ᵃ Bugs found by the two models are distinct (not duplicates).
