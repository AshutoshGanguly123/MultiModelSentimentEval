# train_and_evaluate.py
import matplotlib.pyplot as plt
import numpy as np
from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from models import get_model_and_tokenizer  # import your get_model_and_tokenizer from models.py
import argparse

def fine_tune_model(model_type, dataset_path, output_dir):
    # Initialize the tokenizer and model using functions from models.py
    tokenizer, model = get_model_and_tokenizer(model_type)
    
    # Load the dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=128,
    )

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Define training arguments and set up Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Fine-tuning
    trainer.train()

    #save
    model.save_pretrained(output_dir)

    # Evaluating on test set
    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_dataset_path,
        block_size=128
    )

    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    
    return eval_results

if __name__ == "__main__":
    # You could add argparse here to take command-line arguments for dataset paths
    train_dataset_path = "path/to/train/dataset"
    test_dataset_path = "path/to/test/dataset"

    model_types = ["roberta", "textattack", "bert", "gpt2", "distilbert"]
    metrics = {}
    
    for model_type in model_types:
        print(f"Fine-tuning and evaluating {model_type}...")
        metrics[model_type] = fine_tune_and_evaluate(model_type, train_dataset_path, test_dataset_path)
    
    # Plotting metrics
    eval_losses = [metrics[model_type]['eval_loss'] for model_type in model_types]
    eval_accuracies = [metrics[model_type].get('eval_accuracy', 0) for model_type in model_types]

    x = np.arange(len(model_types))
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, eval_losses, 0.4, label='Evaluation Loss')
    plt.bar(x + 0.2, eval_accuracies, 0.4, label='Evaluation Accuracy')
    plt.xticks(x, model_types)
    plt.ylabel('Metric Value')
    plt.xlabel('Model Types')
    plt.title('Model Evaluation Metrics')
    plt.legend()
    plt.savefig("model_metrics.png")
    plt.show()
