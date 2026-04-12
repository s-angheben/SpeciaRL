<div align="center">
<h1>SpeciaRL: Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification</h1>

<a href="https://scholar.google.com/citations?user=6GPN8hQAAAAJ">Samuele Angheben</a><sup>1,2</sup>, <a href="https://scholar.google.com/citations?user=qMniWoYAAAAJ">Davide Berasi</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=EPImyCcAAAAJ">Alessandro Conti</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=xf1T870AAAAJ">Elisa Ricci</a><sup>1,2</sup>, <a href="https://scholar.google.com/citations?user=KBZ3zrEAAAAJ">Yiming Wang</a><sup>2</sup>

<sup>1</sup> University of Trento, <sup>2</sup> Fondazione Bruno Kessler

<a href="https://arxiv.org/abs/2603.03197"><img src='https://img.shields.io/badge/arXiv-2603.03197-red' alt='Paper PDF'></a> <a href="https://huggingface.co/collections/s-angheben/speciarl"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
</div>

> **Abstract.** *Classifying fine-grained visual concepts under open-world settings, i.e., without a predefined label set, demands models to be both accurate and specific. Recent reasoning Large Multimodal Models (LMMs) exhibit strong visual understanding capability but tend to produce overly generic predictions when performing fine-grained image classification. Our preliminary analysis reveals that models do possess the intrinsic fine-grained domain knowledge. However, promoting more specific predictions (specificity) without compromising correct ones (correctness) remains a non-trivial and understudied challenge. In this work, we investigate how to steer reasoning LMMs toward predictions that are both correct and specific. We propose a novel specificity-aware reinforcement learning framework, SpeciaRL, to fine-tune reasoning LMMs on fine-grained image classification under the open-world setting. SpeciaRL introduces a dynamic, verifier-based reward signal anchored to the best predictions within online rollouts, promoting specificity while respecting the model's capabilities to prevent incorrect predictions. Our out-of-domain experiments show that SpeciaRL delivers the best trade-off between correctness and specificity across extensive fine-grained benchmarks, surpassing existing methods and advancing open-world fine-grained image classification.*

## System overview

**Training.** Training datasets are built via `vlm_openworld_evaluator` (e.g. `vlm_openworld_evaluator/configs/cub_train/`). `verl` trains the policy model using RL algorithms (GRPO, DAPO, DrGRPO) with custom reward functions in `verl/ow_rf/`. The proposed reward function (`reward_best_prediction_dedup.py`) sends each (ground_truth, prediction) pair to `llm_caching_service` for LLM-based verification. The verifier categorizes each prediction as `specific`, `more_specific`, `less_specific`, `generic`, `wrong`, or `abstain`, and this categorization drives the reward signal. Additional reward variants for ablations include `reward_best_prediction_dedup_noisy.py` (noise injection) and `reward_function_print.py` (static scoring). Training configs are in `verl/configs/`.

**Evaluation.** `vlm_openworld_evaluator/run_pipeline_vllm.py` runs a three-stage pipeline. The `dataset` stage builds evaluation sets from HuggingFace sources, `predict` generates model predictions via vLLM, and `verify` scores predictions through `llm_caching_service`. The pipeline supports Best-of-N (BoN) evaluation by setting `num_predictions_per_sample` in the config (e.g. `bon64/` configs generate 64 predictions per sample). Each stage can be run independently via the `--stages` flag. The main evaluation configs are in `vlm_openworld_evaluator/configs/main_evaluation/`.

## Codebase structure

| Directory | Role |
|---|---|
| `containers/` | Apptainer/Singularity recipe (`Apptainer.sh`) for the image used by training and evaluation; based on the upstream `verlai/verl` Docker image plus Redis, MongoDB, and the caching service's Python deps. |
| `llm_caching_service/` | FastAPI service that caches LLM-based verification calls (vLLM) with Redis and tracks classification occurrences in MongoDB; prevents re-classifying identical (ground_truth, prediction) pairs. |
| `scripts/` | Reference SLURM + Apptainer launch scripts under `scripts/train/` and `scripts/eval/`, the bootstrap used on the HPC cluster, sanitised. |
| `verl/` | Snapshot of [verl](https://github.com/volcengine/verl) with the paper's custom reward functions in `verl/ow_rf/` and training configs under `verl/configs/`. |
| `vlm_openworld_evaluator/` | Three-stage pipeline (`run_pipeline_vllm.py`: dataset → predict → verify), YAMLs config present in `config/`. |

## Reproduction

The launch scripts below are the SLURM + Apptainer bootstrap we used on the HPC cluster. Adapt `REPO_ROOT`, `MODELS_ROOT`, and the SLURM account to your environment before use.

- `scripts/train/train_hetjob.sh`: hetjob launcher that brings up the caching service, vLLM, Redis and MongoDB on one node and runs `verl` training on the other.
- `scripts/eval/predict_only.sh`: SLURM array that runs the predict stage of `run_pipeline_vllm.py` over a directory of configs.
- `scripts/eval/verify_only.sh`: SLURM array that runs the verify stage, fronted by a per-task `llm_caching_service` + vLLM.
- `containers/Apptainer.sh`: Apptainer recipe for the image all components run inside.

## Citation

```bibtex
@misc{angheben2026specificityawarereinforcementlearningfinegrained,
      title={Specificity-aware reinforcement learning for fine-grained open-world classification}, 
      author={Samuele Angheben and Davide Berasi and Alessandro Conti and Elisa Ricci and Yiming Wang},
      year={2026},
      eprint={2603.03197},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.03197}, 
}
```

## Acknowledgements

This project builds on [verl](https://github.com/volcengine/verl) for RL training. We also used [lmms-owc](https://github.com/altndrr/lmms-owc) for additional open-world classification evaluation. We thank the authors of both projects.
