
import boto3
import fire
import json
import requests
import torch.distributed as dist

from llama import Llama
from typing import Optional

# torchrun --nproc_per_node 1 worker.py

# Create SQS client
sqs = boto3.client('sqs', endpoint_url='http://sqs.eu-west-1.localhost.localstack.cloud:4566')
queue_url = 'http://sqs.eu-west-1.localhost.localstack.cloud:4566/000000000000/page_tasks'

backend_url = 'http://127.0.0.1:8000/write-page'


def send_results(page_id, results):
    print(results)
    # TODO: add checks for content matching the desired format
    requests.post(backend_url, {
        "page_id": page_id,
        "content": results[0]['generation']['content'],
    })

def process_message(message, generator):
    print(f"Received message: {message}")
    chat = json.loads(message['Body'])
    # book_id = message['MessageAttributes']['BookId']['StringValue']
    page_id = message['MessageAttributes']['PageId']['StringValue']
    results = generator.chat_completion([chat],
        max_gen_len=1024,
        temperature=0.6,
        top_p=0.9
    )
    send_results(page_id, results)


def listen_for_messages(generator):
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
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
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )

def main(
    ckpt_dir: str = "../models/llama/llama-2-7b-chat",
    tokenizer_path: str = "../models/llama/tokenizer.model",
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
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
                print(f"RANK {dist.get_rank()} NODE WORKING")
                generator.chat_completion(
                    config[0], max_gen_len=config[1], temperature=config[2], top_p=config[3]
                )
            except:
                print("EXCEPT")
                pass

if __name__ == "__main__":
    fire.Fire(main)
