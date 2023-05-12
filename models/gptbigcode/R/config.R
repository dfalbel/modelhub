#' @importFrom rlang %||%
gpt_bigcode_config <- function(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=NULL,
    activation_function="gelu_pytorch_tanh",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    scale_attn_weights=TRUE,
    use_cache=TRUE,
    bos_token_id=50256,
    eos_token_id=50256,
    attention_softmax_in_fp32=TRUE,
    scale_attention_softmax_in_fp32=TRUE,
    multi_query=TRUE,
    ...
) {
  self <- list(...)
  self$vocab_size <- vocab_size
  self$n_positions <- n_positions
  self$n_embd <- n_embd
  self$n_layer <- n_layer
  self$n_head <- n_head
  self$n_inner <- n_inner
  self$activation_function <- activation_function
  self$resid_pdrop <- resid_pdrop
  self$embd_pdrop <- embd_pdrop
  self$attn_pdrop <- attn_pdrop
  self$layer_norm_epsilon <- layer_norm_epsilon
  self$initializer_range <- initializer_range
  self$scale_attn_weights <- scale_attn_weights
  self$use_cache <- use_cache
  self$bos_token_id <- bos_token_id
  self$eos_token_id <- eos_token_id
  self$attention_softmax_in_fp32 <- attention_softmax_in_fp32
  self$scale_attention_softmax_in_fp32 <- scale_attention_softmax_in_fp32
  self$multi_query <- multi_query

  # attr map
  self$hidden_size <- self$n_embd
  self$max_position_embeddings <- self$n_positions
  self$num_attention_heads <- self$n_head
  self$num_hidden_layers <- self$n_layer

  # defaults from inherited class
  self$add_cross_attention <- self$add_cross_attention %||% FALSE
  self$output_attentions <- self$output_attentions %||% FALSE
  self$output_hidden_states <- self$output_hidden_states %||% FALSE

  self
}

#' @importFrom hfhub hub_download
gpt_bigcode_config_from_pretrained <- function(identifier, revision = "main") {
  path <- hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)
  do.call(gpt_bigcode_config, config)
}

#' @importFrom hfhub WEIGHTS_NAME WEIGHTS_INDEX_NAME
gpt_bigcode_from_pretrained <- function(identifier, ..., revision = "main") {
  config <- gpt_bigcode_config_from_pretrained(identifier, revision)

  # now we will get the weights for that model
  # first look for a file called `WEIGHTS_NAME()`
  # if this is not found, it means that the weights are sharded, thus we get the
  # index and then load the sharded files.
  weights_path <- try(hub_download(identifier, WEIGHTS_NAME(), revision = revision), silent = TRUE)
  if (inherits(weights_path, "try-error")) {
    index_path <- hub_download(identifier, WEIGHTS_INDEX_NAME(), revision = revision)
    filenames <- unique(unlist(jsonlite::fromJSON(index_path)$weight_map))
    weights_path <- sapply(filenames, function(fname) {
      hub_download(identifier, fname, revision = revision)
    })
    names(weights_path) <- NULL
  }
  with_device(device = "meta", {
    model <- do.call(config$architectures, list(config = config))
  })
  weights <- do.call("c", lapply(weights_path, torch::load_state_dict))
  model$load_state_dict(weights, .refer_to_state_dict = TRUE)
  model
}
