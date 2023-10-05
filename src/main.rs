mod test_suite;

use crate::test_suite::load_fvecs;

fn main() -> Result<(), std::io::Error> {
    let data = load_fvecs("data/siftsmall/siftsmall_base.fvecs", 128, 10_000)?;
    data.chunks(128).for_each(|data| println!("{data:?}"));

    Ok(())
}
