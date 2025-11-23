# Orchestrating Dual-Boundaries: An Arithmetic Intensity Inspired Acceleration Framework for Diffusion Language Models

<div align="center">
  <img src="asset/1.png" alt="Inference with ODB-dLLM on GSM8K dataset" width="600">
</div>

We propose ODB-dLLM, an arithmetic intensity inspired framework for accelerating diffusion-based large language models. By analyzing the interleaved compute- and memory-bound phases in existing dLLM inference frameworks, ODB-dLLM introduces adaptive length prediction strategy and jump-share speculative decoding to optimize computation-memory characteristics on hardware platforms, thereby maximizing inference efficiency.

### Installation
1.Clone the repository:
```
git clone https://github.com/PKU-SEC-Lab/ODB-dLLM.git
cd ODB-dLLM
```
2.Install dependencies:
```
conda create --name ODB-dLLM python=3.10
conda activate ODB-dLLM
pip install -r requirements.txt
```

### Usage
For LLaDA-8B-Instruct model:
```
cd llada_instruct
./eval_llada_instruct.sh <GPU_ID> <Task_Name> 'GSAI-ML/LLaDA-8B-Instruct' <Settings>
```

For LLaDA-1.5 model:
```
cd llada_1_5
./eval_llada_1_5.sh <GPU_ID> <Task_Name> 'GSAI-ML/LLaDA-1.5' <Settings>
```
