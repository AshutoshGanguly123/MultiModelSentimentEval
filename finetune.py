import argparse
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from models import get_model_and_tokenizer  # import the method from models.py

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

    # Fine-tune the model
    trainer.train()

    # Save the model
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
    parser = argparse.ArgumentParser(description='Fine-tuning script')
    parser.add_argument('--model_type', type=str, default="roberta", choices=["roberta", "textattack", "bert", "gpt2", "distilbert"], help='Type of the model to fine-tune')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset for fine-tuning')
    parser.add_argument('--output_dir', type=str, default="./fine_tuned_model", help='Output directory to save fine-tuned model')

    args = parser.parse_args()

    metrics = fine_tune_model(args.model_type, args.dataset_path, args.output_dir)
    
    # Plotting metrics
    eval_losses = [metrics['eval_loss']]
    eval_accuracies = [metrics.get('eval_accuracy', 0)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, eval_losses, 0.4, label='Evaluation Loss')
    plt.bar(x + 0.2, eval_accuracies, 0.4, label='Evaluation Accuracy')
    plt.ylabel('Metric Value')
    plt.title('Model Evaluation Metrics')
    plt.legend()
    plt.savefig("model_metrics.png")
    plt.show()

