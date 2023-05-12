test_that("Model initialization", {

  config <- gpt_bigcode_config()
  model <- GPTBigCodeForCausalLM(config)

  n_parameters <- sum(sapply(model$parameters, function(x) x$numel()))
  expect_equal(n_parameters, 150044160)

})

test_that("config from pre-trained", {
  repo <- "bigcode/gpt_bigcode-santacoder"

  config <- gpt_bigcode_config_from_pretrained(repo)
  model <- gpt_bigcode_from_pretrained(repo)
  model$to(dtype=torch_float())
  tok <- tok::tokenizer$from_pretrained(repo)

  prefix <- "def print_hello_world():\n\""
  suffix <- "print('Hello world')"
  prompt <- glue::glue("<fim-prefix>{prefix}<fim-suffix>{suffix}<fim-middle>")

  expect_error(
    out <- gpt_bigcode_generate(model, tok, prompt, config = config, verbose = FALSE),
    regexp = NA
  )

  expect_error(
    out <- gpt_bigcode_generate_fill(model, tok, prefix, suffix, config = config),
    regexp = NA
  )

})
