# AgentRE_redux: Benchmarking Supply Chain Relation Extraction

> **Dataset Construction, AgentRE Reproducibility, and the Limits of Agentic Prompting**

This repository builds upon the foundational work presented in the paper "**AgentRE: An Agent-Based Framework for Navigating Complex Information Landscapes in Relation Extraction**". For a comprehensive overview of the original methodology, please refer to the [AgentRE Original Repository](https://github.com/Lightblues/AgentRE/).

As part of this research, we have extended AgentRE to evaluate and improve its performance specifically on **supply chain relation extraction tasks**. A key focus of this work has been to address and resolve discrepancies between the original paper's claims and its provided codebase.

---

## 🛠️ Key Improvements (Addressing Original Discrepancies)

Prior to our domain-specific experiments, the following architectural and logical discrepancies between the AgentRE paper and its codebase were addressed:

1. **Reflexion Memory Integration:** The original reflexion memory mechanism did not actively contribute to the response generation process. We have addressed this by adding a dedicated reflection step in the ReAct prompt-set, forcing the model to focus on the reflection when a previous response is incorrect.
2. **Relevant Information Retrieval:** The retrieval mechanism (which extracts entity information to provide context) was disconnected from the ReAct prompt. We fixed this by introducing an extraction step prior to response generation, significantly enhancing the Chain-of-Thought (CoT) process.
3. **Correct/Incorrect Memory Definition:** The paper stated that correct and incorrect memories contain correctly and incorrectly classified triples, respectively. The original implementation merely stored the full example and golden label set regardless of the F1 score. We have corrected this by properly distinguishing and enhancing correct and incorrect memory stores.
4. **Test-time Reflexion:** Reflexions generated during training were not utilized during the testing phase. We have successfully integrated the reflexion memory mechanism into the test pipeline.

---

## 📂 Repository Structure

Below is a brief overview of the key directories in this project:

- `src/`: Core source code including models, tools, data utilities, and the main execution scripts (`main.py`, `configurator.py`).
- `data/` & `src/data/`: Contains the datasets used for evaluation.
- `ablation_study/`: Scripts and results for various ablation setups on AgentRE, AgentRE_redux, and SCAgent (CRA).
- `error_analysis/`: Results from our error analysis studies to inform architectural decisions.
- `out/`: Default output directory for predictions and audit reports.
- `llm_input_trace_...`: JSON traces logging the exact inputs sent to the LLMs.
- `log/` & `logs/`: Execution logs categorized by chosen configurations.

---

## 📊 Datasets

### Standard Benchmarks
- **DuIE2.0:** Sourced from [HuggingFace (Viscacha-Chinese-IE)](https://huggingface.co/datasets/hccngu/Viscacha-Chinese-IE/tree/main/DuIE2.0).
- **SciERC:** Sourced from [Kaggle](https://www.kaggle.com/datasets/hectorrabago/scierc-processed). *(Note: The small size of the SciERC test set resulted in lower overall performances).*

### SC-RED (Wikidata Supply Chain Dataset)
To analyze domain-specific performance in relation extraction, we prepared a new dataset derived from the Wikidata knowledge graph, focusing on the supply chain domain. 
- Relations were generated using a prompt rooted in **SCOR-DS** and **Enterprise Ontology**.
- We manually pruned and semantically mapped overlapping relations, resulting in **15 unique relations** and **5 entity types**.
- The dataset creation pipeline is located under `wikidata_v3` (see below for generation instructions).

---

## 🔬 Error and Ablation Studies

Both AgentRE and AgentRE_redux were subjected to ablation setups using the DuIE2.0 dataset. These analyses are located in the `ablation_study` directory.

We also developed **SCAgent** (also referred to as **CRA** or Constrained ReAct Architecture) and subjected it to similar analyses to compare new changes against the baseline on our SC-RED dataset. 
- You can find the `ReAct_FSL` results on SC-RED and SCAgent results in `ablation_study/wikidata_results`.

**Findings:** Error analysis informed the design of CRA. However, further analysis revealed that `ReAct_FSL` actually outperformed CRA. We suspect this discrepancy is due to the limitations of using F1 scores for generative comparison, and we propose **GenRES** as an alternative evaluation metric.

---

## 🚀 How to Run the Code

1. **Environment Setup:** Ensure your environment is up-to-date by installing dependencies from `agentre_requirements.txt`.
2. **API Keys:** Define your LLM API keys inside `src/new_client_langchain.py`. This file updates the original AgentRE client to use newer LangChain integrations.
3. **Configuration:** 
   - Choose your configuration in `src/configurator.py`. 
   - Alternatively, use `run.sh` to pass system arguments directly (recommended).
   - Whichever configuration you choose, carefully verify the corresponding `.yml` file in the configuration directory.
4. **Execution:** Run the main pipeline:
   ```bash
   python src/main.py
   ```
5. **Outputs:** Results will be saved in the `out/` folder (predictions, audit reports). Traces are saved as JSON files in the root folder, and runtime logs can be found in `log/` under your specific config's subfolder.

---

## 🌐 Generating the Wikidata Dataset (SC-RED)

To regenerate the SC-RED dataset:
1. Navigate to the `wikidata_v3` directory and run the 3 scripts in sequential order. *(Refer to the README in that folder for stage-specific details).*
2. Once complete, `std_train.json` and `std_test.json` will be generated in `src/data/wikidata_v3/`.
3. You can run `visualizer.py` inside the `wikidata_v3` folder to generate an HTML map of the knowledge graph nodes and connections.

---
