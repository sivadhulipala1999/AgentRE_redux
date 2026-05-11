# Benchmarking Supply Chain Relation Extraction: Dataset Construction, AgentRE Reproducibility, and the Limits of Agentic Prompting 

This repository is based on the paper "**AgentRE: An Agent-Based Framework for Navigating Complex Information Landscapes in Relation Extraction**". Go to <a href="https://github.com/Lightblues/AgentRE/"> AgentRE's original repo </a> for a better overview of what was done. 

As a part of my master thesis, my aim is to improve upon the original AgentRE architecture for extracting supply chain information from raw text. Additional experiments will be done to see what changes contribute the most to the model's increase in performance, as per the GenRES framework. 

Prior to this, the following discrepancies between the code and the paper will be addressed. 
<li>The reflexion memory mechanism does not contribute in any way to the response generation process. This will be changed by adding a reflection step in the ReAct prompt-set and the model will be made to focus on the reflection in case of an incorrect response.</li>
<li>The relevant information retrieval mechanism which extracts information about the entities in the query to include as part of its context, does not actually work as part of the ReAct prompt. This will be fixed by including a step to do the extraction before proceeding with the response generation, thus enhancing the CoT process.</li>
<li>The paper mentions correct and incorrect memories contain correctly and incorrectly classified triples respectively. What was implemented instead was just the full example and the golden label set being stored regardless of the f1 score. We fix this by enhancing the correct and incorrect memories</li>
<li>Reflexion done at train time was not being used during test phase, rendering the reflexions worthless. We hence add the reflexion memory mechanism.</li>


## Dataset
DuIE2.0 obtained from <a href="https://huggingface.co/datasets/hccngu/Viscacha-Chinese-IE/tree/main/DuIE2.0">HuggingFace</a>
SciERC obtained from <a href="https://www.kaggle.com/datasets/hectorrabago/scierc-processed">Kaggle</a>
Note that the size of SciERC being small for tests, resulted in lower performances.


## Wikidata Dataset 
A new dataset was prepared from the wikidata knowledge graph pertaining to the supply chain domain to analyse domain specific performance of an agent in relation extraction. In this case, the relations were created using a prompt rooted in SCOR-DS and Enterprise Ontology. These relations were then subjected to manual pruning, if they were too unrelated or overlapping from a semantic perspective, we mapped them into a single label and ended up with 15 relations in total. We also had 5 entity types, based on the relations defined. The wikidata flow can be found in under the name wikidata_v3.


# Error and Ablation Studies 
AgentRE and AgentRE_redux were both subjected to ablation setups with respect to DuIE2.0. These analyses can be found under 'ablation_study'. 

SCAgent (referred to as CRA in the paper/thesis) was subjected to similar analyses as well to analyse how the new changes would compare to the original baseline on the wikidata dataset we prepared. They are also in the 'ablation_study' folder. More specifically, you can find the ReAct_FSL results on SC-RED as well as the SCAgent results in 'ablation_study/wikidata_results'. 

Error Analysis in on both datasets was also done, where we picked error examples to inform our decisions on what kind of architecture should we propose from ReAct_FSL, which ended up becoming CRA. Further analysis on CRA was also done to finally reveal the plot twist, ReAct_FSL actually performed better than CRA. This could be due to F1 scores being used for comparison and hence we propose GenRES as an alternative. 


# To run the code 
- Ensure your environment is up-to-date as per the agentre_requirements file. 
- choose your configuration in configurator.py. If you instead run the shell script its much easier because you pass system arguments, but I did it this way. 
- Make sure you define the API keys in the new_client_langchain.py file. This is the client the whole code is going to be using. Its an update from AgentRE's client file which uses older versions of the code. 
- Whatever configuration you have chosen, go to the corresponding yml file and check the configuration properly. 
- Then go to main.py and execute it to get the results. Your results would typically be, predictions and audit_reports in out folder, llm_input_trace JSON files in the llm_input_traces folder, and log files which you can access in the log folder and under your specific chosen config's subfolder. 

# To get the Wikidata dataset 
- You have to run 3 scripts defined under wikidata_v3. 
- What each stage does is explained the README file in that folder. But run the scripts in order. 
- Once done you will get std_test and std_train files in 'src/data/wikidata_v3/'
- 'visualizer.py' in the wikidata_v3 folder will give you an HTML file showing exactly how the nodes are connected giving you an overview of the knowledge graph 

# Issues 
- If something does not work, please raise an issue. I will try to get the notifications up and will try to respond to every single query. 