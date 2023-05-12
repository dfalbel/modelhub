
# gptbigcode

<!-- badges: start -->
<!-- badges: end -->

## Installation

You can install the development version of gptbigcode from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("dfalbel/modelhub/models/gptbigcode")
```

## Example

Initializing a model:

```r
config <- gpt_bigcode_config_from_pretrained(repo)
model <- gpt_bigcode_from_pretrained(repo)
# weights are in Half, but we get errors for some kernels not implemented
model$to(dtype=torch_float())
tok <- tok::tokenizer$from_pretrained(repo)
```

You can use gptbigcode for standard prompt completion:

```r
prompt <- "def print_hello_world():"
out <- gpt_bigcode_generate(model, tok, prompt, config = config)
```

Or for completing in the middle:

``` r
library(gptbigcode)
prefix <- "def print_hello_world():\n\""
suffix <- "print('Hello world')"
out <- gpt_bigcode_generate_fill(model, tok, prefix, suffix, config = config)
```


