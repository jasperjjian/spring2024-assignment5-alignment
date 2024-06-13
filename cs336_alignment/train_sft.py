from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch
import wandb
from cs336_alignment import sft_loader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def train(model, train_loader, valid_loader, optim, scheduler, device, config, checkpoints=None):
    model.to(device)
    model.train()
    train_loss = 0.0

    for step, train_batch in enumerate(train_loader, start=1):
        
        input_ids = train_batch["input_ids"].to(device)
        labels = train_batch["labels"].to(device)
        logits = model(input_ids).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            optim.step()
            optim.zero_grad()
            scheduler.step()  # Step the scheduler after the optimizer step
            train_loss = loss.item()  # Update train_loss to current batch's loss

            wandb.log({
                'train_loss': train_loss, 
                'step': step, 
                'learning_rate': scheduler.get_last_lr()[0]  # Log the current learning rate
            })
        
        # check loss on sample of validation set
        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                # get 50 examples from the valid loader
                valid_loss = 0.0
                for _ in range(50):
                    valid_batch = next(iter(valid_loader))
                    input_ids = valid_batch["input_ids"].to(device)
                    labels = valid_batch["labels"].to(device)
                    logits = model(input_ids).logits
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                    valid_loss += loss.item()
                valid_loss /= 50
                wandb.log({'valid_loss': valid_loss, 'step': step})
            model.train()

    return train_loss

def main():
    tokenizer = AutoTokenizer.from_pretrained("/data/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "/data/Meta-Llama-3-8B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config_dict = {
        "context_length": 512,
        "learning_rate": 2e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "epochs": 1,
    }
    wandb.init(project="sft_llama", config=config_dict)
    config = wandb.config
    #train_data = sft_loader.SFTDataset(tokenizer, "/home/shared/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz", config.context_length, True)
    train_data = torch.load(open("/home/c-jjian/assignments/spring2024-assignment5-alignment/data/sft.train.update", "rb"))
    train_loader = sft_loader.get_batches_sft(train_data, config.batch_size, True)
    #valid_data = sft_loader.SFTDataset(tokenizer, "/home/shared/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz", config.context_length, False)
    valid_data = torch.load(open("/home/c-jjian/assignments/spring2024-assignment5-alignment/data/sft.valid", "rb"))
    valid_loader = sft_loader.get_batches_sft(valid_data, config.batch_size, False)
    

    optim = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_loader) * config.epochs)

    # Calculate number of warm-up steps (3% of total steps)
    warmup_steps = int(0.03 * len(train_loader) * config.epochs)

    # Create a combined scheduler with warm-up
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optim,
        schedulers=[
            LambdaLR(optimizer=optim, lr_lambda=lambda step: step / warmup_steps),
            CosineAnnealingLR(optimizer=optim, T_max=len(train_loader) * config.epochs - warmup_steps)
        ]
    )

    device = torch.device("cuda")

    for epoch in range(1, config.epochs + 1):
        train_loss = train(model, train_loader, valid_loader, optim, scheduler, device, config)
        wandb.log({'train_loss': train_loss, 'epoch': epoch})
        print(f"Epoch {epoch} completed. Train loss: {train_loss}")

    # save models
    model.save_pretrained("/home/c-jjian/assignments/spring2024-assignment5-alignment/models/sft")
    tokenizer.save_pretrained("/home/c-jjian/assignments/spring2024-assignment5-alignment/models/sft")

if __name__ == "__main__":
    main()
