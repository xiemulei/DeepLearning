// Increase recursion limit to handle complex generic types in Burn framework
#![recursion_limit = "512"]

use crate::llm_train::train_llm_main;

mod llm;
mod llm_train;

fn main() {
    let text_path = "src/assets/sub_wiki_0_99.txt";
    let tokenizer_path = "src/assets/tokenizer.json";
    let artifact_dir = "../tmp/llm";
    train_llm_main(text_path, tokenizer_path, artifact_dir);
}
