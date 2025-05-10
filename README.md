
# AgentRE - redux 

This repository is based on the paper "**AgentRE: An Agent-Based Framework for Navigating Complex Information Landscapes in Relation Extraction**". Go to <a href link="https://github.com/Lightblues/AgentRE/"> AgentRE's original repo </a> for a better overview of what was done. 

As a part of my master thesis, my aim is to improve upon the original AgentRE architecture for extracting supply chain information from raw text. Additional experiments will be done to see what changes contribute the most to the model's increase in performance, as per the GenRES framework. 

Prior to this, the following discrepancies between the code and the paper will be addressed. 
<li>The reflexion memory mechanism does not contribute in any way to the response generation process. This will be changed by adding a reflection step in the ReAct prompt-set and the model will be made to focus on the reflection in case of an incorrect response.</li>
<li>The relevant information retrieval mechanism which extracts information about the entities in the query to include as part of its context, does not actually work as part of the ReAct prompt. This will be fixed by including a step to do the extraction before proceeding with the response generation, thus enhancing the CoT process.</li>

