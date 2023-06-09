#' Download models from HF Hub
#' @export
hub_download <- function(repo_id, filename, ..., revision = "main", repo_type = "model", local_files_only = FALSE, force_download = FALSE) {
  cache_dir <- HUGGINGFACE_HUB_CACHE()
  storage_folder <- fs::path(cache_dir, repo_folder_name(repo_id, repo_type))

  # revision is a commit hash and file exists in the cache, quicly return it.
  if (grepl(REGEX_COMMIT_HASH(), revision)) {
    pointer_path <- get_pointer_path(storage_folder, revision, filename)
    if (fs::file_exists(pointer_path)) {
      return(pointer_path)
    }
  }

  url <- hub_url(repo_id, filename, revision = revision, repo_type = repo_type)

  etag <- NULL
  commit_hash <- NULL
  expected_size <- NULL

  if (!local_files_only) {
    tryCatch({
      metadata <- get_file_metadata(url)

      commit_hash <- metadata$commit_hash
      if (is.null(commit_hash)) {
        cli::cli_abort("Distant resource does not seem to be on huggingface.co (missing commit header).")
      }

      etag <- metadata$etag
      if (is.null(etag)) {
        cli::cli_abort("Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility.")
      }

      # Expected (uncompressed) size
      expected_size <- metadata$size
    })
  }

  # etag is NULL == we don't have a connection or we passed local_files_only.
  # try to get the last downloaded one from the specified revision.
  # If the specified revision is a commit hash, look inside "snapshots".
  # If the specified revision is a branch or tag, look inside "refs".
  if (is.null(etag)) {
    # Try to get "commit_hash" from "revision"
    commit_hash <- NULL
    if (grepl(REGEX_COMMIT_HASH(), revision)) {
      commit_hash <- revision
    } else {
      ref_path <- fs::path(storage_folder, "refs", revision)
      if (fs::file_exists(ref_path)) {
        commit_hash <- readLines(ref_path)
      }
    }

    # Return pointer file if exists
    if (!is.null(commit_hash)) {
      pointer_path <- get_pointer_path(storage_folder, commit_hash, filename)
      if (fs::file_exists(pointer_path)) {
        return(pointer_path)
      }
    }

    if (local_files_only) {
      cli::cli_abort(paste0(
        "Cannot find the requested files in the disk cache and",
        " outgoing traffic has been disabled. To enable hf.co look-ups",
        " and downloads online, set 'local_files_only' to False."
      ))
    } else {
      cli::cli_abort(paste0(
        "Connection error, and we cannot find the requested files in",
        " the disk cache. Please try again or make sure your Internet",
        " connection is on."
      ))
    }
  }

  if (is.null(etag)) cli::cli_abort("etag must have been retrieved from server")
  if (is.null(commit_hash)) cli::cli_abort("commit_hash must have been retrieved from server")

  blob_path <- fs::path(storage_folder, "blobs", etag)
  pointer_path <- get_pointer_path(storage_folder, commit_hash, filename)

  fs::dir_create(fs::path_dir(blob_path))
  fs::dir_create(fs::path_dir(pointer_path))

  # if passed revision is not identical to commit_hash
  # then revision has to be a branch name or tag name.
  # In that case store a ref.
  # we write an alias between revision and commit-hash
  if (revision != commit_hash) {
    ref_path <- fs::path(storage_folder, "refs", revision)
    fs::dir_create(fs::path_dir(ref_path))
    fs::file_create(ref_path)
    writeLines(commit_hash, ref_path)
  }

  if (fs::file_exists(pointer_path) && !force_download) {
    return(pointer_path)
  }

  if (fs::file_exists(blob_path) && !force_download) {
    fs::link_create(blob_path, pointer_path)
    return(pointer_path)
  }

  withr::with_tempfile("tmp", {
    lock <- filelock::lock(paste0(blob_path, ".lock"))
    on.exit({filelock::unlock(lock)})
    curl::curl_download(url, tmp, quiet = !interactive())
    fs::file_move(tmp, blob_path)
    fs::link_create(blob_path, pointer_path)
  })

  pointer_path
}

hub_url <- function(repo_id, filename, ..., revision = "main", repo_type = "model") {
  glue::glue("https://huggingface.co/{repo_id}/resolve/{revision}/{filename}")
}

get_pointer_path <- function(storage_folder, revision, relative_filename) {
  snapshot_path <- fs::path(storage_folder, "snapshots")
  pointer_path <- fs::path(snapshot_path, revision, relative_filename)
  pointer_path
}

repo_folder_name <- function(repo_id, repo_type = "model") {
  repo_id <- gsub(pattern = "/", x = repo_id, replacement = REPO_ID_SEPARATOR())
  glue::glue("{repo_type}s{REPO_ID_SEPARATOR()}{repo_id}")
}

get_file_metadata <- function(url) {
  req <- httr::HEAD(
    url = url,
    httr::add_headers("Accept-Encoding" = "identity")
  )
  list(
    commit_hash = grab_from_headers(req$all_headers, "x-repo-commit"),
    etag = normalize_etag(grab_from_headers(req$all_headers, "etag")),
    size = as.integer(grab_from_headers(req$all_headers, "content-length"))
  )
}

grab_from_headers <- function(headers, nm) {
  for(h in headers) {
    header <- h$headers
    if (!is.null(header[[nm]]))
      return(header[[nm]])
  }
  NULL
}

normalize_etag <- function(etag) {
  if (is.null(etag)) return(NULL)
  etag <- gsub(pattern = '"', x = etag, replacement = "")
  etag <- gsub(pattern = "W/", x = etag, replacement = "")
  etag
}

REPO_ID_SEPARATOR <- function() {
  "--"
}
HUGGINGFACE_HUB_CACHE <- function() {
  path <- Sys.getenv("HUGGINGFACE_HUB_CACHE", "~/.cache/huggingface/hub")
  path.expand(path)
}
REGEX_COMMIT_HASH <- function() {
  "^[0-9a-f]{40}$"
}

#' Weight file names in HUB
#' 
#' @describeIn WEIGHTS_NAME Name of weights file
#' @export
WEIGHTS_NAME <- function() "pytorch_model.bin"
#' @export
#' @describeIn WEIGHTS_NAME Name of weights index file
WEIGHTS_INDEX_NAME <- function() "pytorch_model.bin.index.json"
