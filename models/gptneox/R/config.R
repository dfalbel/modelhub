#' Defines a GPTNeoX model configuration
#'
#' @param vocab_size (int, optional, defaults to 50432) — Vocabulary size of the GPTNeoX model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling GPTNeoXModel.
#' @param hidden_size (int, optional, defaults to 6144) — Dimension of the encoder layers and the pooler layer.
#' @param num_hidden_layers (int, optional, defaults to 44) — Number of hidden layers in the Transformer encoder.
#' @param num_attention_heads (int, optional, defaults to 64) — Number of attention heads for each attention layer in the Transformer encoder.
#' @param intermediate_size (int, optional, defaults to 24576) — Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
#' @param hidden_act (str or function, optional, defaults to "gelu") — The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu", "selu" and "gelu_new" are supported.
#' @param rotary_pct (float, optional, defaults to 0.25) — percentage of hidden dimensions to allocate to rotary embeddings
#' @param rotary_emb_base (int, optional, defaults to 10000) — base for computing rotary embeddings frequency
#' @param max_position_embeddings (int, optional, defaults to 2048) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
#' @param initializer_range (float, optional, defaults to 1e-5) — The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
#' @param layer_norm_eps (float, optional, defaults to 1e-12) — The epsilon used by the layer normalization layers.
#' @param use_cache (bool, optional, defaults to True) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if config.is_decoder=True.
#' @param use_parallel_residual (bool, optional, defaults to True) — Whether to use a “parallel” formulation in each Transformer layer, which can provide a slight training speedup at large scales (e.g. 20B). Example —
#' @param ... Additional configuration options.
#'
#' @describeIn gpt_neox_config Defines the configuration of the tokenizer.
#'
#' @export
gpt_neox_config <- function(
  vocab_size=50432,
  hidden_size=6144,
  num_hidden_layers=44,
  num_attention_heads=64,
  intermediate_size=24576,
  hidden_act="gelu",
  rotary_pct=0.25,
  rotary_emb_base=10000,
  max_position_embeddings=2048,
  initializer_range=0.02,
  layer_norm_eps=1e-5,
  use_cache=TRUE,
  bos_token_id=0,
  eos_token_id=2,
  tie_word_embeddings=FALSE,
  use_parallel_residual=TRUE,
  ...
  ) {
  self <- list(...)
  self$vocab_size = vocab_size
  self$max_position_embeddings = max_position_embeddings
  self$hidden_size = hidden_size
  self$num_hidden_layers = num_hidden_layers
  self$num_attention_heads = num_attention_heads
  self$intermediate_size = intermediate_size
  self$hidden_act = hidden_act
  self$rotary_pct = rotary_pct
  self$rotary_emb_base = rotary_emb_base
  self$initializer_range = initializer_range
  self$layer_norm_eps = layer_norm_eps
  self$use_cache = use_cache
  self$tie_word_embeddings = tie_word_embeddings
  self$use_parallel_residual = use_parallel_residual
  self
}

#' @describeIn gpt_neox_config Uses a configuration defined a HF hub repository.
gpt_neox_config_from_pretrained <- function(identifier, revision = "main") {
  path <- hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)
  do.call(gpt_neox_config, config)
}

#' GPTNeoX model
#'
#' Creates a GPTNeoX Model.
#'
#' @param identifier The HuggingFace repository to download the model from.
#' @param revision The repository revision to download from. Either a branch name
#'   or a commit hash.
#' @param config A model configuration created by [gpt_neox_config()] or obtained
#'   from [gpt_neox_config_from_pretrained()]
#' @param ... Not currently used.
#'
#' @describeIn gpt_neox_from_pretrained Creates from a configuration from a HF repository.
#'
#' @export
gpt_neox_from_pretrained <- function(identifier, ..., revision = "main") {
  config <- gpt_neox_config_from_pretrained(identifier, revision)

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

#' @describeIn gpt_neox_from_pretrained Creates a GPTNeoX model from a configuration list.
#' @export
gpt_neox_from_config <- function(config) {
  arch <- if (!is.null(config$architectures)) config$architectures else "GPTNeoXForCausalLM"
  model <- do.call(arch, list(config = config))
}

#' Creates a GPTNeoX tokenizer from a pre-trained tokenizer
#'
#' @inheritParams gpt_neox_from_pretrained
#' @importFrom hfhub hub_download
#'
#' @export
gpt_neox_tokenizer_from_pretrained <- function(identifier, ..., revision = "main") {
  tok_file <- hub_download(identifier, "tokenizer.json", revision = revision)
  tok::tokenizer$from_file(tok_file)
}
