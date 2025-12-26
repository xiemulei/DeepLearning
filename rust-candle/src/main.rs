use candle_core::Result;

mod chapter2;
mod utils;

fn main() -> Result<()> {
    let _ = crate::chapter2::train()?;

    Ok(())
}
