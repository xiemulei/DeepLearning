use candle_core::{Result, Tensor};

pub trait Dataset {
    fn len(&self) -> Result<usize>;
    fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)>;
    fn shuffle(&mut self) -> Result<()>;
}

pub struct DataLoader<'a> {
    dataset: Box<dyn Dataset + 'a>,
    batch_size: usize,
    current_index: usize,
    shuffle: bool,
}

impl<'a> DataLoader<'a> {
    pub fn new<D: Dataset + 'a>(dataset: D, batch_size: usize, shuffle: bool) -> Result<Self> {
        Ok(Self {
            dataset: Box::new(dataset),
            batch_size,
            current_index: 0,
            shuffle,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.current_index = 0;
        if self.shuffle {
            let _ = self.dataset.shuffle()?;
        }

        Ok(())
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.current_index * self.batch_size;
        let end = std::cmp::min(start + self.batch_size, self.dataset.len().ok()?);

        if start >= end {
            return None;
        }

        let batch = self.dataset.get_batch(start, end).ok()?;
        self.current_index += 1;
        Some(Ok(batch))
    }
}
