gpt_bigcode_generate <- function(model, tokenizer, prompt, ..., config = list(), verbose = TRUE) {
  config$max_new_tokens <- config$max_new_tokens %||% 64
  config$do_sample <- config$do_sample %||% TRUE
  config$bos_token_id <- config$bos_token_id %||% 0
  config$eos_token_id <- config$eos_token_id %||% 0
  config$top_k <- config$top_k %||% 50
  config$temperature <- config$temperature %||% 1

  new_tokens <- list()
  new_text <- list()
  device <- model$gpt_neox$embed_in$weight$device

  model$eval() # model should be in eval mode
  for (i in seq_len(config$max_new_tokens)) {
    token <- get_next_token(model, tokenizer, prompt, config, device)
    new <- tokenizer$decode(token)

    new_tokens[[length(new_tokens) + 1]] <- token
    new_text[[length(new_text) + 1]] <- new

    if (token == config$eos_token_id) {
      break
    }

    if (verbose) {
      if (i == 1) cat(prompt)
      cat(new)
    }

    prompt <- paste0(prompt, new)
  }

  if (verbose) cat("\n")

  list(
    prompt = prompt,
    new_tokens = new_tokens,
    new_text = new_text
  )
}

gpt_bigcode_generate_fill <- function(model, tokenizer, prefix, suffix, ..., config = list(), verbose = TRUE) {
  config$max_new_tokens <- config$max_new_tokens %||% 64
  config$do_sample <- config$do_sample %||% TRUE
  config$bos_token_id <- config$bos_token_id %||% 0
  config$eos_token_id <- config$eos_token_id %||% 0
  config$top_k <- config$top_k %||% 50
  config$temperature <- config$temperature %||% 1

  new_tokens <- list()
  new_text <- list()
  device <- model$gpt_neox$embed_in$weight$device
  prompt <- glue::glue("<fim-prefix>{prefix}<fim-suffix>{suffix}<fim-middle>")

  for (i in seq_len(config$max_new_tokens)) {
    token <- get_next_token(model, tokenizer, prompt, config, device)
    new <- tokenizer$decode(token)

    new_tokens[[length(new_tokens) + 1]] <- token
    new_text[[length(new_text) + 1]] <- new

    if (token == config$eos_token_id) {
      break
    }

    if (verbose) {
      if (i == 1) cat(prefix)
      cat(new)
    }

    prompt <- paste0(prompt, new)
  }

  cat(suffix, "\n")

  list(
    prompt = prompt,
    new_tokens = new_tokens,
    new_text = new_text
  )
}


get_next_token <- function(model, tokenizer, prompt, config, device) {
  encoding <- tokenizer$encode(prompt)

  inputs <- torch_tensor(encoding$ids + 1L, device = device)$unsqueeze(1)
  mask <- torch_tensor(encoding$attention_mask, device = device)$unsqueeze(1)

  with_no_grad({
    out <- model(inputs, attention_mask = mask)
  })

  logits <- out$logits[,-1,]

  if (config$do_sample) {
    logits <- logits/config$temperature
    logits <- logits$topk(config$top_k)

    selected <- torch_multinomial(nnf_softmax(logits[[1]], dim = -1), num_samples = 1)
    token <- logits[[2]][,selected$item()]$item()
  } else {
    token <- as.integer(logits$argmax(dim = -1))
  }

  as.integer(token - 1L)
}
