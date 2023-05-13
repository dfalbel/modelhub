#' GPTBigCode neural net modules
#'
#' @import torch
#' @importFrom zeallot %<-%
#' @param config Configuration options as created by [gpt_bigcode_config()].
#' @param is_cross_attention Is cross attention?
#' @param layer_idx The layers idx (used to automatically define its size).
#' @export
GPTBigCodeAttention <- nn_module(
    "GPTBigCodeAttention",
    initialize = function(config, is_cross_attention=FALSE, layer_idx=NULL) {
        self$mask_value <- NULL

        self$multi_query <- config$multi_query
        self$embed_dim <- config$hidden_size
        self$num_heads <- config$num_attention_heads
        self$head_dim <- trunc(self$embed_dim / self$num_heads)
        self$kv_heads <- if (self$multi_query) 1 else self$num_heads
        self$kv_dim <- self$kv_heads * self$head_dim
        self$split_size <- self$embed_dim
        if (self$head_dim * self$num_heads != self$embed_dim) {
            cli::cli_abort(c(
                x = "{.val embed_dim} must be divisible by num_heads.",
                i = "got {.val embed_dim}: {.val {self$embed_dim}}",
                i = "got {.val num_heads}: {.val {self$num_heads}}"
            ))
        }
        self$scale_attn_weights <- config$scale_attn_weights
        self$is_cross_attention <- is_cross_attention

        self$layer_idx <- layer_idx
        self$attention_softmax_in_fp32 <- config$attention_softmax_in_fp32
        self$scale_attention_softmax_in_fp32 <-
            config$scale_attention_softmax_in_fp32 &&
                config$attention_softmax_in_fp32

        if (self$is_cross_attention) {
            if (sef$multi_query) {
                cli::cli_abort(c(
                    "Multi-Query Attention not supported for cross_attention"
                ))
            }
            self$c_attn <- nn_linear(self$embed_dim, 2 * self$embed_dim)
            self$q_attn <- nn_linear(self$embed_dim, self$embed_dim)
        } else {
            self$c_attn <- nn_linear(
                self$embed_dim, self$embed_dim + 2 * self$kv_dim)
        }

        self$c_proj <- nn_linear(self$embed_dim, self$embed_dim)

        self$attn_dropout <- nn_dropout(config$attn_pdrop)
        self$resid_dropout <- nn_dropout(config$resid_pdrop)
    },
    .get_mask_value = function(device, dtype) {
        if (is.null(self$mask_value) ||
            (self$mask_value$dtype != dtype) ||
            (!(self$mask_value$device == device))) {
            self$mask_value <- torch_scalar_tensor(
                torch_finfo(dtype)$min,
                dtype=dtype,
                device=device
            )
        }
        self$mask_value
    },
    .attn = function(query, key, value, attention_mask = NULL,
        head_mask = NULL) {

        dtype <- query$dtype
        softmax_dtype <- if (self$attention_softmax_in_fp32)
            torch_float32() else dtype
        upcast <- dtype != softmax_dtype

        unscale <- if (self$scale_attention_softmax_in_fp32 && upcast)
            self$layer_idx + 1 else 1
        scale_factor <- unscale ^ -1
        if (self$scale_attn_weights)
            scale_factor <- scale_factor / (self$head_dim ^ 0.5)

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape <- query$shape
        batch_size <- query_shape[1]
        key_length <- key$size(-1)
        if (self$multi_query) {
            # (batch_size, query_length, num_heads, head_dim) x (batch_size, head_dim, key_length)
            # -> (batch_size, query_length, num_heads, key_length)
            query_length <- query_shape[2]
            attn_shape <- list(batch_size, query_length, self$num_heads, key_length)
            attn_view <- list(batch_size, query_length * self$num_heads, key_length)
            # No copy needed for MQA 2, or when layer_past is provided.
            query <- query$reshape(list(batch_size, query_length * self$num_heads, self$head_dim))
        } else {
            query_length <- query_shape[3]
            attn_shape <- list(batch_size, self$num_heads, query_length, key_length)
            attn_view <- list(batch_size * self$num_heads, query_length, key_length)
            # No copy needed for MQA 2, or when layer_past is provided.
            query <- query$reshape(list(batch_size * self$num_heads, query_length, self$head_dim))
            # No copy when layer_past is provided.
            key <- key$reshape(list(batch_size * self.num_heads, self$head_dim, key_length))
        }

        attn_weights <- torch_empty(attn_view, device=query$device, dtype=query$dtype)
        if (query$device$type == "cpu") {
            attn_weights$zero_()
            beta <- 1
        } else {
           beta <- 0
        }
        attn_weights <- torch_baddbmm(attn_weights, query, key, beta=beta, alpha=scale_factor)$
            view(attn_shape)

        if (upcast) {
            # Use a fused kernel to prevent a large overhead from casting and scaling.
            # Sub-optimal for fp16 since it uses fp32 compute.
            if (is.null(attention_mask)) {
                attn_weights <- upcast_softmax(attn_weights, unscale, softmax_dtype)
            } else {
                mask_value <- self$.get_mask_value(attn_weights$device, softmax_dtype)
                attn_weights <- upcast_masked_softmax(attn_weights, attention_mask, mask_value, unscale, softmax_dtype)
            }
        } else {
            if (!is.null(attention_mask)) {
                mask_value <- self$.get_mask_value(attn_weights$device, softmax_dtype)

                # The fused kernel is very slow when the key length is not a multiple of 8, so we skip fusion.
                attn_weights <- torch_where(attention_mask, attn_weights, mask_value)
            }
            attn_weights <- nnf_softmax(attn_weights, dim=-1)
        }

        attn_weights <- self$attn_dropout(attn_weights)

        # Mask heads if we want to
        if (!is.null(head_mask)) {
            if (self$multi_query) {
                head_mask <- head_mask$transpose(2, 3)
            }
            attn_weights <- attn_weights * head_mask
        }

        if (self$multi_query) {
            attn_output <- torch_bmm(attn_weights$view(attn_view), value)$view(query_shape)
        } else {
            attn_output <- torch_matmul(attn_weights, value)
        }

        list(attn_output, attn_weights)
    },
    forward = function(hidden_states, layer_past = NULL, attention_mask = NULL, head_mask = NULL,
                       encoder_hidden_states = NULL, encoder_attention_mask = NULL, use_cache = FALSE,
                       output_attentions = FALSE) {
        if (!is.null(encoder_hidden_states)) {
            if (is.null(self$q_attn) || !self$is_cross_attention) {
                cli::cli_abort("If class is used as cross attention, the weights `q_attn` have to be defined.")
            }
            query <- self$q_attn(hidden_states)
            key_value <- self$c_attn(encoder_hidden_states)
            attention_mask <- encoder_attention_mask
        } else if (self$multi_query) {
            c(query, key_value) %<-% self$c_attn(hidden_states)$split(list(self$embed_dim, 2 * self$kv_dim), dim=3)
        } else {
            # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
            # i.e., the memory layout is not the same as GPT2.
            # This makes the concatenation with past_key_value more efficient.
            c(query, key_value) %<-% (
                self$c_attn(hidden_states)$
                    view(c(hidden_states$shape[1:2], self$num_heads, 3 * self$head_dim))$
                    transpose(2, 3)$
                    split(list(self$head_dim, 2 * self$head_dim), dim=4)
            )
        }

        if (!is.null(layer_past)) {
            key_value <- torch_cat(list(layer_past, key_value), dim = -2)
        }
        present <- if (use_cache) key_value else NULL

        c(key, value) %<-% key_value$split(list(self$head_dim, self$head_dim), dim=-1)

        c(attn_output, attn_weights) %<-% self$.attn(query, key$transpose(-1, -2), value, attention_mask, head_mask)

        if (!self$multi_query) {
            attn_output <- attn_output$transpose(2, 3)$reshape(hidden_states$shape)
        }

        attn_output <- self$c_proj(attn_output)
        attn_output <- self$resid_dropout(attn_output)

        outputs <- list(attn_output, present)
        if (output_attentions) {
            if (self$multi_query) {
                # Transpose to return weights in the usual format (batch_size, num_heads, query_length, key_length)
                attn_weights <- attn_output$transpose(2, 3)
            }
            outputs[[length(outputs) + 1]] <- attn_weights
        }
        outputs
    }
)

#' @describeIn GPTBigCodeAttention MLP module
#' @param intermediate_size the intermediate size
#' @export
GPTBigCodeMLP <- nn_module(
    "GPTBigCodeMLP",
    initialize = function(intermediate_size, config) {
        embed_dim <- config$hidden_size
        self$c_fc <- nn_linear(embed_dim, intermediate_size)
        self$c_proj <- nn_linear(intermediate_size, embed_dim)
        self$act <- ACT2FN[[config$activation_function]]
        self$dropout <- nn_dropout(config$resid_pdrop)
    },
    forward = function(hidden_states) {
        hidden_states <- self$c_fc(hidden_states)
        hidden_states <- self$act(hidden_states)
        hidden_states <- self$c_proj(hidden_states)
        hidden_states <- self$dropout(hidden_states)
        hidden_states
    }
)

GPTBigCodeBlock <- nn_module(
    "GPTBigCodeBlock",
    initialize = function(config, layer_idx = NULL) {
        hidden_size <- config$hidden_size
        self$inner_dim <- if (!is.null(config$n_inner)) config$n_inner else 4 * hidden_size

        self$ln_1 <- nn_layer_norm(hidden_size, eps=config$layer_norm_epsilon)
        self$attn <- GPTBigCodeAttention(config, layer_idx=layer_idx)
        self$ln_2 <- nn_layer_norm(hidden_size, eps=config$layer_norm_epsilon)

        if (config$add_cross_attention) {
            if (config$multi_query) {
                cli::cli_abort("Cross-attention not implemented for MQA")
            }
            self$crossattention <- GPTBigCodeAttention(config, is_cross_attention=TRUE, layer_idx=layer_idx)
            self$ln_cross_attn <- nn_layer_norm(hidden_size, eps=config$layer_norm_epsilon)
        }

        self$mlp <- GPTBigCodeMLP(self$inner_dim, config)
    },
    forward = function(hidden_states, layer_past = NULL, attention_mask = NULL, head_mask = NULL,
                       encoder_hidden_states = NULL, encoder_attention_mask = NULL, use_cache = FALSE,
                       output_attentions = FALSE) {
        residual <- hidden_states
        hidden_states <- self$ln_1(hidden_states)
        attn_outputs <- self$attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        attn_output <- attn_outputs[[1]]
        outputs <- attn_outputs[-1]

        # residual connection
        hidden_states = attn_output + residual

        if (!is.null(encoder_hidden_states)) {
            # add one self-attention block for cross-attention
            if (is.null(self$crossattention)) {
                cli::cli_abort("If `encoder_hidden_states` are passed, {.val self} has to be instantiated with cross-attention layers by setting `config$add_cross_attention=TRUE`")
            }
            residual <- hidden_states
            hidden_states <- self$ln_cross_attn(hidden_states)
            cross_attn_outputs <- self$crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
            attn_output <- cross_attn_outputs[[1]]
            # residual connections
            hidden_states <- residual + attn_output
            outputs <- c(outputs, cross_attn_outputs[-c(1, 2)])  # add cross attentions
        }

        residual <- hidden_states
        hidden_states <- self$ln_2(hidden_states)
        feed_forward_hidden_states <- self$mlp(hidden_states)
        # residual connection
        hidden_states <- residual + feed_forward_hidden_states

        if (use_cache) {
            outputs <- c(hidden_states, outputs)
        } else {
            outputs <- c(hidden_states, outputs[-1])
        }

        outputs
    }
)

#' @include model-utils.R
NULL

#' @describeIn GPTBigCodeAttention The pretrained model class
#' @export
GPTBigCodePreTrainedModel <- nn_module(
    "GPTBigCodePreTrainedModel",
    inherit = module_utils_mixins,
    initialize = function(config) {
        self$config <- config
    }
)

#' @describeIn GPTBigCodeAttention The model class
#' @export
GPTBigCodeModel <- nn_module(
    "GPTBigCodeModel",
    inherit = GPTBigCodePreTrainedModel,
    initialize = function(config) {
        super$initialize(config)
        self$multi_query <- config$multi_query
        self$embed_dim <- config$hidden_size

        self$wte <- nn_embedding(config$vocab_size, self$embed_dim)
        self$wpe <- nn_embedding(config$max_position_embeddings, self$embed_dim)

        self$drop <- nn_dropout(config$embd_pdrop)
        self$h <- nn_module_list(lapply(seq_len(config$num_hidden_layers), function(x) {
            GPTBigCodeBlock(config, layer_idx=x-1)
        }))

        self$max_positions <- config$max_position_embeddings
        self$ln_f <- nn_layer_norm(self$embed_dim, eps=config$layer_norm_epsilon)

        self$register_bias()
        self$gradient_checkpointing = FALSE

        # Initialize weights and apply final processing
        #self.post_init()
    },
    register_bias = function() {
      max_positions <- self$max_positions
      if (is.null(max_positions)) return()
      self$register_buffer(
        "bias", torch_tril(torch_ones(max_positions, max_positions, dtype=torch_bool())), persistent=FALSE
      )
    },
    .load_from_state_dict = function(...) {
      super$.load_from_state_dict(...)
      self$register_bias()
    },
    get_input_embeddings = function() {
        self$wte
    },
    set_input_embeddings = function(new_embeddings) {
        self$wte <- new_embeddings
    },
    forward = function(
        input_ids = NULL,
        past_key_values = NULL,
        attention_mask = NULL,
        token_type_ids = NULL,
        position_ids = NULL,
        head_mask = NULL,
        inputs_embeds = NULL,
        encoder_hidden_states = NULL,
        encoder_attention_mask = NULL,
        use_cache = NULL,
        output_attentions = NULL,
        output_hidden_states = NULL,
        return_dict = NULL
    ) {
        output_attentions <- if (!is.null(output_attentions)) output_attentions else self$config$output_attentions
        output_hidden_states <- if (!is.null(output_hidden_states)) output_hidden_states else self$config$output_hidden_states
        use_cache <- if (!is.null(use_cache)) use_cache else self$config$use_cache
        return_dict <- if (!is.null(return_dict)) return_dict else self$config$use_return_dict

        if (!is.null(input_ids) && !is.null(inputs_embeds)) {
            cli::cli_abort("You cannot specify both `input_ids` and `inputs_embeds` at the same time")
        } else if (!is.null(input_ids)) {
            input_shape <- input_ids$size()
            input_ids <- input_ids$reshape(c(-1, input_ids$size(-1)))
            batch_size <- input_ids$shape[1]
        } else if (!is.null(inputs_embeds)) {
            input_shape <- inputs_embeds$size()
            batch_size <- input_shape[1]
        } else {
            cli::cli_abort("You have to specify either `input_ids` or `inputs_embeds`")
        }

        device <- if (!is.null(input_ids)) input_ids$device else inputs_embeds$device

        if (!is.null(token_type_ids)) {
            token_type_ids <- token_type_ids$view(c(-1, tail(input_shape, 1)))
        }

        if (!is.null(position_ids)) {
            position_ids <- position_ids$view(c(-1, tail(input_shape, 1)))
        }

        if (is.null(past_key_values)) {
            past_length <- 1
            past_key_values <- rep(list(NULL), length(self$h))
        } else {
            past_length <- past_key_values[[1]]$size(-2)
        }

        if (!is.null(attention_mask) && (length(attention_mask$shape) == 2) && is.null(position_ids)) {
            position_ids <- attention_mask$to(dtype = torch_long())$cumsum(-1)
            position_ids$masked_fill_(attention_mask == 0, 2L)
            if (past_length > 1) {
                position_ids <- position_ids[, past_length:(tail(input_shape, 1) + past_length)]
            }
        } else if (is.null(position_ids)) {
            position_ids <- torch_arange(past_length, tail(input_shape, 1) + past_length, dtype=torch_long(), device=device)
            position_ids <- position_ids$unsqueeze(1)$view(c(-1, tail(input_shape, 1)))
        }

        # Self-attention mask.
        query_length <- tail(input_shape, 1)
        key_length <- past_length + query_length
        self_attention_mask <- self$bias[NULL, (key_length - query_length):(key_length-1L), 1:(key_length-1L)]

        if (is.null(attention_mask)) {
            self_attention_mask <- self_attention_mask * attention_mask$view(c(batch_size, 1, -1))$to(
                dtype=torch_bool(), device=self_attention_mask$device)
        }

        # MQA models: (batch_size, query_length, n_heads, key_length)
        # MHA models: (batch_size, n_heads, query_length, key_length)
        attention_mask <- self_attention_mask$unsqueeze(if (self$multi_query) 3 else 2)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if (
            self$config$add_cross_attention &&
            !is.null(encoder_hidden_states) &&
            !is.null(encoder_attention_mask)
        ) {
            if (encoder_attention_mask$dim() == 2) {
                encoder_attention_mask$unsqueeze(2)
            }
            encoder_attention_mask <- encoder_attention_mask$bool()$unsqueeze(if (self$multi_query) 3 else 2)
        } else {
            encoder_attention_mask <- NULL
        }

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask <- self$get_head_mask(head_mask, self$config$n_layer)

        if (is.null(inputs_embeds)) {
          inputs_embeds <- self$wte(input_ids)
        }

        position_embeds <- self$wpe(position_ids)
        hidden_states <- inputs_embeds + position_embeds

        if (!is.null(token_type_ids)) {
          token_type_embeds <- self$wte(token_type_ids)
          hidden_states <- hidden_states + token_type_embeds
        }

        hidden_states <- self$drop(hidden_states)
        output_shape <- c(input_shape, hidden_states$size(-1))

        presents <- if (use_cache) list() else NULL
        all_self_attentions <- if (output_attentions) list() else NULL
        all_cross_attentions <- if (output_attentions) list() else NULL
        all_hidden_states <- if (output_hidden_states) list() else NULL

        for (i in seq_along(self$h)) {
          block <- self$h[[i]]
          layer_past <- past_key_values[[i]]

          if (output_hidden_states) {
            all_hidden_states[[length(all_hidden_states) + 1]] <- hidden_states
          }

          if (self$gradient_checkpointing && self$training) {
            # TODO
          } else {
            outputs <- block(
              hidden_states,
              layer_past=layer_past,
              attention_mask=attention_mask,
              head_mask=head_mask[[i]],
              encoder_hidden_states=encoder_hidden_states,
              encoder_attention_mask=encoder_attention_mask,
              use_cache=use_cache,
              output_attentions=output_attentions
            )
          }

          hidden_states <- outputs[[1]]
          if (use_cache) {
            presents[[length(presents) + 1]] <- outputs[[2]]
          }

          if (output_attentions) {
            all_self_attentions[[length(all_self_attentions) + 1]] <- outputs[[use_cache + 2]]
            if (self$config$add_cross_attention) {
              all_cross_attentions[[length(all_cross_attentions) + 1]] <- outputs[[use_cache + 3]]
            }
          }

        }

        hidden_states <- self$ln_f(hidden_states)
        hidden_states <- hidden_states$view(output_shape)

        if (output_hidden_states) {
          all_hidden_states[[length(all_hidden_states) + 1]] <- hidden_states
        }

        list(
          last_hidden_state=hidden_states,
          past_key_values=presents,
          hidden_states=all_hidden_states,
          attentions=all_self_attentions,
          cross_attentions=all_cross_attentions
        )
    }
)

#' @describeIn GPTBigCodeAttention The text generation class
#' @export
GPTBigCodeForCausalLM <- nn_module(
  "GPTBigCodeForCausalLM",
  inherit = GPTBigCodeModel,
  initialize = function(config) {
    self$transformer <- GPTBigCodeModel(config)
    self$lm_head <- nn_linear(config$n_embd, config$vocab_size, bias=FALSE)
  },
  forward = function(
    input_ids = NULL,
    past_key_values = NULL,
    attention_mask = NULL,
    token_type_ids = NULL,
    position_ids = NULL,
    head_mask = NULL,
    inputs_embeds = NULL,
    encoder_hidden_states = NULL,
    encoder_attention_mask = NULL,
    labels = NULL,
    use_cache = NULL,
    output_attentions = NULL,
    output_hidden_states = NULL,
    return_dict = NULL
    ) {
    return_dict <- if(!is.null(return_dict)) return_dict else self$config$return_dict

    transformer_outputs = self$transformer(
      input_ids,
      past_key_values=past_key_values,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      encoder_hidden_states=encoder_hidden_states,
      encoder_attention_mask=encoder_attention_mask,
      use_cache=use_cache,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict
    )

    hidden_states <- transformer_outputs[[1]]
    lm_logits <- self$lm_head(hidden_states)

    loss <- NULL
    if (!is.null(labels)) {
      # Shift so that tokens < n predict n
      shift_logits <- lm_logits[.., 1:(-1), ]$contiguous()
      shift_labels <- labels[.., 1:N]$contiguous()$to(shift_logits$device)
      # Flatten the tokens
      loss_fct = nn_cross_entropy_loss()
      loss <- loss_fct(shift_logits$view(c(-1, shift_logits.size(-1))), shift_labels$view(-1))
    }

    list(
      loss=loss,
      logits=lm_logits,
      past_key_values=transformer_outputs$past_key_values,
      hidden_states=transformer_outputs$hidden_states,
      attentions=transformer_outputs$attentions,
      cross_attentions=transformer_outputs$cross_attentions
    )
  }

)

gelu_pytorch_tanh <- nn_module(
  initialize = function() {

  },
  forward = function(input) {
    torch:::torch_gelu(input, approximate = "tanh")
  }
)

ACT2FN <- list(
  "relu" = nn_relu(),
  "gelu" = nn_gelu(),
  "sigmoid" = nn_sigmoid(),
  "gelu_pytorch_tanh" = gelu_pytorch_tanh()
)


