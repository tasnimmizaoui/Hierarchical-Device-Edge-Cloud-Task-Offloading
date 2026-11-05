# Mandatory article outline — professor's template

Follow this exact section order. The article is not a project diary: explain the technical choices, formulate the model, present evidence, and critically interpret results.

## Title, Authors & Abstract

- Precise title describing hierarchical device-edge-cloud task offloading and the learning approach.
- One abstract paragraph, 150–250 words maximum: context/problem, technical challenge, proposed architecture, evaluation approach, and actual quantitative findings.
- 4–6 technical keywords.
- Do not write expected or invented results as if measured; fill quantitative claims only after final experiments.

## 1. Introduction

- MEC/edge-cloud context and why task placement matters.
- Concrete problem: heterogeneous compute/network, device energy, deadlines, and local information constraints.
- Why purely centralized offloading may create information/communication bottlenecks.
- Brief positioning and article structure.

## 2. Related Work

- Critical synthesis of PureEdgeSim/EdgeCloudSim/iFogSim and relevant hierarchical or multi-agent RL offloading work.
- Compare assumptions, action spaces, state observability, reward terms, workload types and evaluation metrics.
- Use the professor-provided references and verified additional IEEE/arXiv sources.

## 3. System Model / Proposed Approach

- Three-tier architecture diagram and task lifecycle.
- Device, edge and cloud agent states/actions/rewards.
- Compute and network models; queue disciplines; result-return path.
- Formal latency, energy, deadline and coordination-cost definitions.
- State which data fields come from the trace and which are synthetically generated.

## 4. Implementation & Performance Evaluation

- Simulator/version, Python and library versions, dataset subset and preprocessing.
- Scenario parameter table with units and provenance.
- Baseline definitions: Random, Greedy/Nearest, EDF; centralized RL; hierarchical RL.
- Training protocol, reward weights, hyperparameters, seeds and number of repeats.
- Figures/tables comparing latency, 95th percentile latency, energy, deadline success, throughput/utilization and communication overhead.
- Every figure must be referenced and interpreted in the text. Explain why trends occur and identify when the proposed method loses.

## 5. Conclusion & Future Work

- Summarize actual contributions and evidence.
- State limitations of the simulation and learning setup.
- Give realistic future work (federated RL only if implemented; otherwise list it as future work).

## References

- Numbered IEEE style, in order of citation.
- Cite simulator papers and dataset documentation, not only software landing pages.
- Every factual claim from literature should be supported by a citation.

> The professor's provided template numbers the Abstract/Authors block before Introduction and then asks for Introduction, Related Work, Architecture, Implementation/Evaluation, Conclusion, References. Use the section names/order exactly as the course handout and confirm whether the article counts the title/abstract as an unnumbered front matter section.
