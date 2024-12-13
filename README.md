# tunain-page-worker

## Application Description

Tunain application allows users to co-write a book with an AI, generating both text and illustrations. Users provide prompts for extracts, and the AI completes paragraphs and generates illustrative images.

## Page Worker Overview

The text worker repository generates AI-created text scripts based on user prompts using a Llama 2 model. It listens to an SQS queue, processes messages and uses the backend REST api on completion.

## Features

- Generate coherent text completions for user-provided extracts.
- Communicate with the backend for task coordination.

## Technology Stack

- Model: Llama 2
- Environment: Python, PyTorch

## Commands

```
torchrun --nproc_per_node 1 worker.py
python3.8 -m torch.distributed.run --nproc_per_node 1 worker.py
```
