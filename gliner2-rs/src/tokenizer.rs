use std::path::Path;

use anyhow::Context;
use ndarray::Array2;
use tokenizers::{EncodeInput, tokenizer::Tokenizer};

use crate::Result;

/// Light wrapper over HF tokenizers to prepare encoder inputs.
pub struct RuntimeTokenizer {
    inner: Tokenizer,
    unk_id: Option<u32>,
}

impl RuntimeTokenizer {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let tokenizer_path = dir.as_ref().join("tokenizer.json");
        let inner = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        let unk_id = inner.get_vocab(true).get("[UNK]").copied();
        Ok(Self { inner, unk_id })
    }

    /// Encode raw text with add_special_tokens=false and return 1xL tensors.
    pub fn encode_text(&self, text: &str) -> Result<(Array2<i64>, Array2<i64>)> {
        let encoding = self
            .inner
            .encode(EncodeInput::Single(text.into()), false)
            .map_err(anyhow::Error::msg)?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let len = ids.len();
        let input_ids = Array2::from_shape_vec((1, len), ids)?;
        let attention_mask = Array2::from_elem((1, len), 1i64);
        Ok((input_ids, attention_mask))
    }

    /// Convert a list of pre-tokenized subwords (as strings) to ids and mask.
    pub fn convert_subwords<'a>(
        &self,
        subwords: impl IntoIterator<Item = &'a str>,
    ) -> Result<(Array2<i64>, Array2<i64>)> {
        let ids: Vec<i64> = subwords
            .into_iter()
            .map(|tok| {
                self.inner
                    .token_to_id(tok)
                    .or(self.unk_id)
                    .map(|id| id as i64)
                    .context("unknown token and no [UNK] id")
            })
            .collect::<Result<_>>()?;
        let len = ids.len();
        let input_ids = Array2::from_shape_vec((1, len), ids)?;
        let attention_mask = Array2::from_elem((1, len), 1i64);
        Ok((input_ids, attention_mask))
    }

    /// Look up a token id.
    pub fn id_for(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Tokenize a single token string into subword pieces.
    pub fn tokenize_token(&self, token: &str) -> Result<Vec<String>> {
        let encoding = self
            .inner
            .encode(EncodeInput::Single(token.into()), false)
            .map_err(anyhow::Error::msg)?;
        Ok(encoding.get_tokens().to_vec())
    }
}
