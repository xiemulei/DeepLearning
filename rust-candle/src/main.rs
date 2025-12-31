use candle_core::Result;

use crate::chapter3::train_main;

mod chapter2;
mod chapter3;
mod utils;

// 主函数：演示模型的使用流程
fn main() -> Result<()> {
    let _ = train_main()?;

    Ok(())
}
