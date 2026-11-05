# Interactive Explanations for RL

## Links
Interactive version: https://explanationstruckfreightinteractive.streamlit.app/
Static version: https://explanationstruckfreightstatic.streamlit.app/

## Title
Interactive Explanations for Reinforcement Learning: Taking Advantage of Progressive Disclosure

## Keywords
Explainable AI, Reinforcement Learning, Interactive Explanations.

## TLDR Description
AI approach that takes advantage of interactive explanations to explain the decisions of a Reinforcement Learning (RL) agent. The context of the problem is the transportation of goods using vehicles; more specifically, the acceptance/rejection of orders and the assignment of freight to trucks. The explanations are based on the State2Explanation framework [1], in which concepts are attached to states to make them easily understandable by humans. To create AI explanations that are more aligned with how humans communicate with each other, we take advantage of interactivity, allowing users to engage in a dynamic exploration process. Notably, users can (a) view basic information about each truck-freight matchings, (b) select the matching they prefer, and (c) click to reveal various kinds of explanations on demand. For (c), these explanations include (1) why the AI model evaluated that matching as good/bad, (2) how the model was trained, (3) the ability to view similar training examples, and (d) the ability to verify how the evaluation and explanation would change if some values were different. Our solution employs progressive disclosure in all of its components, which is a paradigm that involves allowing the user to obtain explanations at their own pace to avoid overwhelming them. For the purpose of comparison, we built two versions of our XAI system, a static one and an interactive one.

1. Das, D., Chernova, S., Kim, B.: State2explanation: Concept-based explanations to benefit agent learning and user understanding. Advances in Neural Information Processing Systems 36, 67156–67182 (2023)

## Research Paper
The pdf is available in the folder named "Research Paper".

## Description
In recent years, Artificial Intelligence (AI) has come to the forefront of society, due to various advancements in the field of Machine Learning (ML), such as the rising power of Large Language Models (LLMs). Nowadays, people of all skill levels are in contact with AI, from novice users to experts. One of the three ML paradigms, Reinforcement Learnings (RL), has been responsible for much of this progress, playing a key role in the training of LLMs and surpassing humans in various tasks. RL involves an agent interacting with an environment and learning through trial-and-error. As the agent selects which action to take for each situation (state), it is rewarded accordingly to its choice, allowing it to improve its performance at the task at hand. However, the great capabilities of modern AI have brought along with them tremendous complexity; as a result, even AI experts cannot fully comprehend these approaches. Due to AI’s pivotal role in the future of mankind, the topic of trustworthy AI has gained a lot of attention recently. Trustworthy AI covers the key principles that need to be respected when working with this technology. One of such principles is transparency, which involves making AI more understandable to humans and is the main focus of Explainable Artificial Intelligence (XAI). In spite of the research community making strides in this area, various challenges remain. For instance, transparency by itself seems insufficient to attain human trust and for many years researchers tended to ignore non-expert users in their studies. Furthermore, research intersecting XAI and social sciences has argued that AI explanations should work similarly to a conversation, allowing for a dynamic back-and-forth between the system and the user. After all, such an interaction leads to a gradual and natural understanding of the inner workings of the AI. Despite positive reception of this theory, the vast majority of AI explanations remain static to this day. On top of that, works related to interactive explanations tend to neglect RL, with only a couple works on this intersection existing. Therefore, we propose an interactive XAI approach for RL that leverages concept-based explanations. Our solution takes advantage of progressive disclosure, a paradigm that consists of revealing information to the user on demand to meet their needs and avoid overwhelming them with unnecessary details. While prior work has used this principle to some extent, we make it a cornerstone of our contribution. Additionally, we take a completely different approach that follows a theoretical framework developed by Weld and Bansal. For the sake of comparison, we built two versions of our XAI system, a static one and an interactive one. We hope that our contribution paves the way towards more research that combines interactive explanations with RL.

Note: this research was not finished. Notably, we did not conduct the experiments using LLM-as-a-Judge and human participants.


## Project
Research conducted within the scope of the Nexus project for the green and digital transition of transportation and logistics. This project involved regularly meeting with people working for key companies related to the transportation and logistics industry. As such, we were able to take into account the real-world requirements for developing an AI solution for the transportation problem we worked on. https://nexuslab.pt/


