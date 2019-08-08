use std::fs::File;
use std::io::prelude::*;
use std::io::{Error, ErrorKind};

fn buf_to_u32(buf: &[u8; 4]) -> u32 {
    let mut res = 0;
    for byte in buf.iter() {
        res = res << 8 | *byte as u32;
    }
    res
}

pub fn read_images(path: &str) -> Result<Vec<[f64; 784]>, std::io::Error> {
    let mut f = File::open(path)?;
    let mut buff = [0; 4];

    f.read_exact(&mut buff)?;

    if buf_to_u32(&buff) != 2051 {
        return Err(Error::new(ErrorKind::Other, "Wrong Magic Number"));
    }

    f.read_exact(&mut buff)?;

    let nb_images = buf_to_u32(&buff);

    f.read_exact(&mut buff)?;
    if buf_to_u32(&buff) != 28 {
        return Err(Error::new(ErrorKind::Other, "Wrong Image Size"));
    }

    f.read_exact(&mut buff)?;
    if buf_to_u32(&buff) != 28 {
        return Err(Error::new(ErrorKind::Other, "Wrong Image Size"));
    }

    println!("{} images: 28x28", nb_images);

    let mut res = Vec::new();
    for _ in 0..nb_images {
        let mut image = [0; 784];
        let mut float_image = [0.; 784];

        f.read_exact(&mut image)?;

        for n in 0..784 {
            float_image[n] = image[n] as f64 / 255.;
        }
        res.push(float_image);
    }
    Ok(res)
}

pub fn read_labels(path: &str) -> Result<Vec<u8>, std::io::Error> {
    let mut f = File::open(path)?;
    let mut buff = [0; 4];

    f.read_exact(&mut buff)?;

    if buf_to_u32(&buff) != 2049 {
        return Err(Error::new(ErrorKind::Other, "Wrong Magic Number"));
    }

    f.read_exact(&mut buff)?;

    let nb_labels = buf_to_u32(&buff);

    println!("{} labels", nb_labels);

    Ok(f.take(nb_labels as u64)
        .bytes()
        .map(|x| x.unwrap())
        .collect())
}
