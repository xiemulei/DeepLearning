use candle_core::Result;
use candle_nn::VarMap;

pub fn print_varmap(varmap: &VarMap) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let mut param_count = 0;

    for (key, value) in data.iter() {
        println!("{key}");
        println!("{:?}", value);
        param_count += value.elem_count();
    }

    println!("Parameter Count: {param_count}");

    Ok(())
}
