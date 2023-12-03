# ChatSim
ChatSim: Editable Scene Simulation for Autonomous Driving via LLM-Agent Collaboration

## Abstract
Scene simulation in autonomous driving has gained significant attention because of its huge potential for generating customized data. 
However, existing editable scene simulation approaches face limitations in terms of user interaction efficiency, 
multi-camera photo-realistic rendering and external digital assets integration. 
To address these challenges, this paper introduces <b><i>ChatSim</i></b>, the first system that enables editable photo-realistic 
3D driving scene simulations via natural language commands with external digital assets. To enable editing with 
high command flexibility, ChatSim leverages a large language model (LLM) agent collaboration framework. 
To generate photo-realistic outcomes, ChatSim employs a novel multi-camera neural radiance field method. 
Furthermore, to unleash the potential of extensive high-quality digital assets, ChatSim employs a novel 
multi-camera lighting estimation method to achieve scene-consistent assets' rendering. Our experiments 
on Waymo Open Dataset demonstrate that ChatSim can handle complex language commands and generate 
corresponding photo-realistic scene videos.

![teaser](./assets/teaser.jpg)

## LLM Collaboration Framework
To address complex or abstract user commands effectively, <b><i>ChatSim</i></b> adopts a large language model 
(LLM)-based multi-agent collaboration framework. The key idea is to exploit multiple LLM agents, 
each with a specialized role, to decouple an overall simulation demand into specific editing tasks, 
thereby mirroring the task division and execution typically founded in the workflow of a human-operated 
company. This workflow offers two key advantages for scene simulation. First, LLM agents' ability to 
process human language commands allows for intuitive and dynamic editing of complex driving scenes, 
enabling precise adjustments and feedback. Second, the collaboration framework enhances simulation 
efficiency and accuracy by distributing specific editing tasks, ensuring improved task completion rates.

![teaser](./assets/method.jpg)

## Todo
- [ ] Arxiv paper release 
- [ ] Code release
- [ ] Data and model release