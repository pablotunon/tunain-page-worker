
import boto3
import fire
import json
import requests
import logging
import os
import torch.distributed as dist

from llama import Llama
from typing import Optional

logger = logging.getLogger(__name__)

# torchrun --nproc_per_node 1 worker.py
# python3.8 -m torch.distributed.run --nproc_per_node 1 worker.py

# Create SQS client
SQS_ENDPOINT = os.getenv('SQS_ENDPOINT', 'http://sqs.eu-west-1.localhost.localstack.cloud:4566')
PAGE_QUEUE_URL = os.getenv('SQS_PAGE_TASK_QUEUE_URL', 'http://sqs.eu-west-1.localhost.localstack.cloud:4566/000000000000/page-tasks')
MODEL_PATH = os.getenv('MODEL_PATH', '../models/llama/llama-2-7b-chat')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', '../models/llama/tokenizer.model')

# Create SQS client

sqs = boto3.client('sqs', endpoint_url=SQS_ENDPOINT)

backend_url = 'http://127.0.0.1:8000/write-page'

MAX_GENERATION_ATTEMPTS = 4

def send_results(page_id, results):
    print(results)
    requests.post(backend_url, {
        "page_id": page_id,
        "content": results
    })

def process_multi_prompt(chat, generator):
    text = generator.chat_completion([chat['text']],
        max_gen_len=1024,
        temperature=0.6,
        top_p=0.9
    )[0]['generation']['content']
    illustration_promt = [chat['illustration'] + [{"role": "user", "content": text}]]
    print(illustration_promt)
    illustration = generator.chat_completion(illustration_promt,
        max_gen_len=1024,
        temperature=0.6,
        top_p=0.9
    )[0]['generation']['content']
    return json.dumps({
        'text': text,
        'illustration': illustration
    })

def process_single_prompt(chat, generator):
    attempts = 0
    while attempts < MAX_GENERATION_ATTEMPTS:
        try:
            generated = generator.chat_completion([chat],
                max_gen_len=1024,
                temperature=0.6 + (attempts * 0.1),
                top_p=0.9
            )[0]['generation']['content']
            # load the string to verify it has the correct format
            generated_dict = json.loads(generated)
            generated_dict['text']
            generated_dict['illustration']
            return generated
        except json.decoder.JSONDecodeError:
            attempts += 1
            logger.warning(f"NOT JSON Generation - {attempts} attempts")
    raise Exception("MAX_GENERATION_ATTEMPTS reached without a JSON")


def process_message(message, generator):
    print(f"Received message: {message}")
    chat = json.loads(message['Body'])
    # book_id = message['MessageAttributes']['BookId']['StringValue']
    page_id = message['MessageAttributes']['PageId']['StringValue']

    # Check format
    if type(chat) == dict:
        results = process_multi_prompt(chat, generator)
    else:
        results = process_single_prompt(chat, generator)
    send_results(page_id, results)


def listen_for_messages(generator):
    while True:
        response = sqs.receive_message(
            QueueUrl=PAGE_QUEUE_URL,
            AttributeNames=['All'],
            MaxNumberOfMessages=1,
            MessageAttributeNames=['All'],
            VisibilityTimeout=0,
            WaitTimeSeconds=20  # Adjust the wait time as needed
        )

        if 'Messages' in response and response['Messages']:
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']
            
            process_message(message, generator)
            
            # Delete the received message from the queue
            sqs.delete_message(
                QueueUrl=PAGE_QUEUE_URL,
                ReceiptHandle=receipt_handle
            )

def main(
    ckpt_dir: str = MODEL_PATH,
    tokenizer_path: str = TOKENIZER_PATH,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 2048,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    port: int = 8000,
):
    # Create our Code Llama object.
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # With torchrun and distributed PyTorch, multiple copies of this code
    # can be run at once. We only want one of them (node 0) to consume SQS messages
    # and we will use it to control the rest.
    if dist.get_rank() == 0:
        listen_for_messages(generator)

    # Nodes which are not node 0 wait for tasks.
    else:
        # This infinite loop is the part that I find ugly, iterating actively
        # without a waiting statement
        while True:
            config = [None] * 4
            try:
                dist.broadcast_object_list(config)
                generator.chat_completion(
                    config[0], max_gen_len=config[1], temperature=config[2], top_p=config[3]
                )
            except:
                pass

if __name__ == "__main__":
    fire.Fire(main)
