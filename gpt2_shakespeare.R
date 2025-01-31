library(tidyverse)
library(torch)
library(luz)
library(zeallot)

url <- "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
source("gpt2.R")

char_dataset <- torch::dataset(
  initialize = function(data, block_size = 128) {
    self$block_size <- block_size
    self$data <- stringr::str_split_1(data, "")
    
    self$data_size <- length(self$data)
    self$vocab <- unique(self$data)
    self$vocab_size <- length(self$vocab)
  },
  .getitem = function(i) {
    chunk <- self$data[i + seq_len(self$block_size + 1)]
    idx <- match(chunk, self$vocab)
    list(
      x = head(idx, self$block_size),
      y = tail(idx, self$block_size)
    )
  },
  .length = function() {
    self$data_size - self$block_size - 1L # this is to account the last value
  }
)

dataset <- char_dataset(readr::read_file(url))
dataset[1] # this allows us to see an element of the dataset

model <- torch::nn_module(
  initialize = function(vocab_size) {
    self$gpt <- gpt2(
      vocab_size = vocab_size,
      n_layer = 6,
      n_head = 6,
      n_embd = 192
    )
  },
  forward = function(x) {
    # we have to transpose to make the vocabulary the last dimension
    self$gpt(x)$transpose(2,3)
  },
  generate = function(x, temperature = 1, iter = 50, top_k = 10) {
    # samples from the model givn a context vector.
    for (i in seq_len(iter)) {
      logits <- self$forward(x)[,,-1]
      logits <- logits/temperature
      c(prob, ind) %<-% logits$topk(top_k)
      logits <- torch_full_like(logits, -Inf)$scatter_(-1, ind, prob)
      logits <- nnf_softmax(logits, dim = -1)
      id_next <- torch_multinomial(logits, num_samples = 1)
      x <- torch_cat(list(x, id_next), dim = 2)
    }
    x
  }
)

# samples from the model using the context.
generate <- function(model, vocab, context, ...) {
  local_no_grad() # disables gradient for sampling
  x <- match(stringr::str_split_1(context, ""), vocab)
  x <- torch_tensor(x)[NULL,]$to(device = model$device)
  content <- as.integer(model$generate(x, ...)$cpu())
  paste0(vocab[content], collapse = "")
}

display_cb <- luz_callback(
  initialize = function(iter = 500) {
    self$iter <- iter # print every 500 iterations
  },
  on_train_batch_end = function() {
    if (!(ctx$iter %% self$iter == 0))
      return()
    
    ctx$model$eval()
    with_no_grad({
      # sample from the model...
      context <- "O God, O God!"
      text <- generate(ctx$model, dataset$vocab, context, iter = 100)
      cli::cli_h3(paste0("Iter ", ctx$iter))
      cli::cli_text(text)
    })
    
  }
)

fitted <- model |>
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam
  ) |>
  set_opt_hparams(lr = 5e-4) |>
  set_hparams(vocab_size = dataset$vocab_size) |>
  fit(
    dataset,
    dataloader_options = list(batch_size = 128, shuffle = TRUE),
    epochs = 1,
    callbacks = list(
      display_cb(iter = 500),
      luz_callback_gradient_clip(max_norm = 1)
    )
  )

save.image("gpt2_shakespeare.RData")

context <- "O God, O God!"
text <- generate(fitted$model, dataset$vocab, context, iter = 100)
cat(text)