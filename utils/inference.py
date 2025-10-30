import torch
from tqdm import tqdm


def batch_greedy_search(images, questions, model, tokenizer, max_length, device):
    """
    Performs greedy decoding to generate text answers for a batch of image-question pairs.

    Args:
        images (torch.Tensor): Batch of visual features or image tensors.
        questions (list[str]): List of text questions corresponding to each image.
        model (torch.nn.Module): Trained VQA model for generating answers.
        tokenizer: Tokenizer used for encoding/decoding text.
        max_length (int): Maximum sequence length for generation.
        device (torch.device): Device to run the computation on.

    Returns:
        list[str]: Generated text answers.
    """

    answers = []
    batch_size = len(questions)

    model.eval()
    with torch.no_grad():
        # Prepare the text prompts for the entire batch like "Question: <q> Answer:"
        prompt_texts = [f"Question: {q}\nAnswer:" for q in questions]

        # Tokenize the prompts with padding to handle varying lengths
        prompt_inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding='longest',
            add_special_tokens=False
        )

        # Prepare model inputs
        padded_input_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        padded_attention_mask = torch.zeros((batch_size, max_length), device=device)

        orig_length = prompt_inputs['input_ids'].size(1)
        padded_input_ids[:, :orig_length] = prompt_inputs['input_ids']
        padded_attention_mask[:, :orig_length] = prompt_inputs['attention_mask']

        images = images.to(device)

        # Initialize tensors to store generated tokens
        only_answer_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        # Track which sequences have finished generating
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Record each sample length (number of non-eos tokens)
        valid_lengths = padded_attention_mask.sum(dim=1).long()
        batch_indices = torch.arange(batch_size, device=device)

        for _ in range(max_length):
            max_valid_lengths = max(valid_lengths).item()
            if max_valid_lengths >= max_length:
                break  # Stop if any sequence reached max_length

            # Forward pass through the model
            logits = model(video=images, qa_inputs_ids=padded_input_ids[:, :max_valid_lengths], qa_att_mask=padded_attention_mask[:, :max_valid_lengths])

            # Get next token probabilities and entropy
            last_valid_logits = logits[batch_indices, valid_lengths - 1, :]

            # Get next token
            next_token_ids = torch.argmax(last_valid_logits, dim=-1)

            # Check EOS
            is_eos = (next_token_ids == tokenizer.eos_token_id)
            finished = finished | is_eos  # Update finished status

            padded_input_ids[batch_indices, valid_lengths] = next_token_ids
            padded_attention_mask[batch_indices, valid_lengths] = 1
            valid_lengths += 1

            # Append the selected tokens to the generated_ids
            only_answer_ids = torch.cat([only_answer_ids, next_token_ids.unsqueeze(1)], dim=1)

            # If all sequences have finished, exit early
            if finished.all():
                break

        # Decode the generated tokens into strings
        generated_ids_cpu = only_answer_ids.cpu().tolist()  # Move to CPU and convert to list for processing
        for i in range(batch_size):
            # Find the first occurrence of eos_token_id to truncate the answer
            try:
                eos_index = generated_ids_cpu[i].index(tokenizer.eos_token_id)
                answer_ids = generated_ids_cpu[i][:eos_index]
            except ValueError:
                # If eos_token_id is not found, use all generated tokens
                answer_ids = generated_ids_cpu[i]

            # Decode the token IDs to a string, skipping special tokens
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            answers.append(answer)

    return answers


def inference(val_loader, model, tokenizer, device, seq_length):
    """
    Runs inference over a dataset to generate answers and compute predictions.

    Args:
        val_loader (DataLoader): DataLoader providing validation batches.
        model (torch.nn.Module): Trained model for answer generation.
        tokenizer: Tokenizer for decoding text outputs.
        device (torch.device): Device used for inference.
        seq_length (int): Maximum generation length.

    Returns:
        tuple: (ground_truth_answers, predicted_answers, ground_truth_keywords)
    """
    
    references = []
    predictions = []
    keyword_references = []

    model.eval()
    with torch.no_grad():
        for _, (images, questions, answers, keywords) in enumerate(tqdm(val_loader), 0):
            images = images.to(device)
            generated_response = batch_greedy_search(
                images,
                questions,
                model,
                tokenizer,
                max_length=seq_length,
                device=device
            )

            references.extend(answers)
            predictions.extend(generated_response)
            keyword_references.extend(keywords)

    print("First 5 Labels:", references[:5])
    print("First 5 Predictions:", predictions[:5])
    print("First 5 Keywords:", keyword_references[:5])

    return references, predictions, keyword_references