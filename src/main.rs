use std::fmt;
use std::i16;

use image::ImageBuffer;
use image::ImageReader;
use image::Rgb;

const FRAME_WIDTH: u32 = 720;
const FRAME_HEIGHT: u32 = 480;
const BLOCKS_WIDTH: u32 = FRAME_WIDTH / 8;
const BLOCKS_HEIGHT: u32 = FRAME_HEIGHT / 8;

#[derive(Clone, Default)]
struct Block(pub [[i16; 8]; 8]);

impl fmt::Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[ ")?;
        for y in 0..8 {
            for x in 0..8 {
                write!(f, "{:3}, ", self.0[x][y])?;
            }
            if y < 7 {
                write!(f, "\n  ")?;
            }
        }
        write!(f, " ]")?;
        Ok(())
    }
}

static DST8_MAT: [[i32; 8]; 8] = [
    [5, 10, 14, 16, 16, 14, 10, 5],
    [10, 16, 14, 5, -5, -14, -16, -10],
    [14, 14, 0, -14, -14, 0, 14, 14],
    [16, 5, -14, -10, 10, 14, -5, -16],
    [16, -5, -14, 10, 10, -14, -5, 16],
    [14, -14, 0, 14, -14, 0, 14, -14],
    [10, -16, 14, -5, -5, 14, -16, 10],
    [5, -10, 14, -16, 16, -14, 10, -5],
];

static DST_SCALE: [[f64; 8]; 8] = [
    [
        18.575,
        19.0,
        19.225,
        19.733333333333334,
        19.652173913043477,
        19.642857142857142,
        19.125,
        19.05,
    ],
    [
        18.833333333333332,
        19.093023255813954,
        19.651162790697676,
        19.77777777777778,
        19.88372093023256,
        19.736842105263158,
        19.16216216216216,
        19.18918918918919,
    ],
    [
        19.47826086956522,
        19.76595744680851,
        20.041666666666668,
        20.083333333333332,
        20.025,
        19.63888888888889,
        19.16216216216216,
        19.42105263157895,
    ],
    [
        19.964285714285715,
        20.12727272727273,
        20.444444444444443,
        20.39622641509434,
        20.574468085106382,
        20.3,
        19.952380952380953,
        20.113636363636363,
    ],
    [
        19.70967741935484,
        19.857142857142858,
        20.20967741935484,
        20.271186440677965,
        20.327272727272728,
        19.98,
        19.510204081632654,
        19.76923076923077,
    ],
    [
        19.761194029850746,
        19.791044776119403,
        20.176470588235293,
        20.227272727272727,
        20.225806451612904,
        20.05263157894737,
        19.683333333333334,
        19.746031746031747,
    ],
    [
        19.303030303030305,
        19.3768115942029,
        19.760563380281692,
        19.901408450704224,
        20.12857142857143,
        19.941176470588236,
        19.454545454545453,
        19.55072463768116,
    ],
    [
        19.18032786885246,
        19.257575757575758,
        19.791044776119403,
        19.985915492957748,
        20.0,
        19.742424242424242,
        19.348484848484848,
        19.36764705882353,
    ],
];

static mut max_s: i32 = 0;
static mut min_s: i32 = 0;

fn dst8_1d_forward(src: [i32; 8], dst: &mut [i32; 8]) {
    for k in 0..8 {
        let mut s = 0i32;
        for n in 0..8 {
            s += DST8_MAT[k][n] * src[n];
        }
        s >>= 4;
        dst[k] = s;
        unsafe {
            if s > max_s {
                max_s = s;
            }
            if s < min_s {
                min_s = s;
            }
        }
    }
}

fn dst8_2d_forward(src: &Block, dst: &mut Block) {
    let mut row_in = [[0i32; 8]; 8];
    let mut row_out = [[0i32; 8]; 8];
    let mut col_in = [[0i32; 8]; 8];

    for r in 0..8 {
        for c in 0..8 {
            row_in[r][c] = src.0[r][c] as i32;
        }
    }

    // Row transform
    for r in 0..8 {
        dst8_1d_forward(row_in[r], &mut row_out[r]);
    }

    // Transpose to col_in
    for r in 0..8 {
        for c in 0..8 {
            col_in[c][r] = row_out[r][c];
        }
    }

    // Column transform
    let mut col_out = [[0i32; 8]; 8];
    for c in 0..8 {
        dst8_1d_forward(col_in[c], &mut col_out[c]);
    }

    // Transpose back to out
    for r in 0..8 {
        for c in 0..8 {
            dst.0[r][c] = col_out[c][r].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
}

fn dst8_1d_inverse(src: [i32; 8], dst: &mut [i32; 8]) {
    for n in 0..8 {
        let mut s = 0i32;
        for k in 0..8 {
            s += DST8_MAT[k][n] * src[k];
        }
        dst[n] = s >> 4;
    }
}

fn dst8_2d_inverse(src: &Block, dst: &mut Block) {
    let mut row_in = [[0i32; 8]; 8];
    let mut row_out = [[0i32; 8]; 8];
    let mut col_in = [[0i32; 8]; 8];

    for r in 0..8 {
        for c in 0..8 {
            row_in[r][c] = src.0[r][c] as i32;
        }
    }

    // Row transform
    for c in 0..8 {
        dst8_1d_inverse(row_in[c], &mut col_in[c]);
    }

    // Transpose to col_in
    for r in 0..8 {
        for c in 0..8 {
            row_out[r][c] = col_in[c][r];
        }
    }

    // Column transform
    let mut recon = [[0i32; 8]; 8];
    for r in 0..8 {
        dst8_1d_inverse(row_out[r], &mut recon[r]);
    }

    // Transpose back to out
    for r in 0..8 {
        for c in 0..8 {
            dst.0[r][c] = recon[c][r].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
}

fn main() -> anyhow::Result<()> {
    let img = ImageReader::open("data/01.png")?.decode()?.to_rgb8();

    let mut blocks = vec![Block::default(); (BLOCKS_WIDTH * BLOCKS_HEIGHT) as usize];
    let mut blocks_out = vec![Block::default(); (BLOCKS_WIDTH * BLOCKS_HEIGHT) as usize];

    for (px, py, Rgb([r, g, b])) in img.enumerate_pixels() {
        let color = 0.299 * (*r as f64) + 0.587 * (*g as f64) + 0.114 * (*b as f64);
        let bx = (px % 8) as usize;
        let by = (py % 8) as usize;
        let id = (px / 8 + (py / 8) * BLOCKS_WIDTH) as usize;
        blocks[id].0[bx][by] = color as i16;
    }
    println!("{:?}", blocks[0]);
    let ob = blocks[0].clone();

    for (i, block) in blocks_out.iter_mut().enumerate() {
        dst8_2d_forward(&blocks[i], block);
    }

    #[allow(static_mut_refs)]
    unsafe {
        println!("max:{} min:{}", max_s, min_s);
    }

    for (i, block) in blocks_out.iter_mut().enumerate() {
        for i in 0..8 {
            for j in 0..8 {
                block.0[i][j] >>= 5;
            }
        }
    }

    println!("{:?}", blocks_out[0]);

    for (i, block) in blocks_out.iter_mut().enumerate() {
        for i in 0..8 {
            for j in 0..8 {
                block.0[i][j] <<= 5;
            }
        }
    }

    for (i, block) in blocks.iter_mut().enumerate() {
        dst8_2d_inverse(&blocks_out[i], block);
    }

    for i in 0..8 {
        for j in 0..8 {
            print!("{} ", (blocks[0].0[j][i] as f64) / (ob.0[j][i] as f64));
        }
        println!();
    }

    println!("{:?}", blocks[0]);

    let mut img_out = ImageBuffer::new(FRAME_WIDTH, FRAME_HEIGHT);

    for (px, py, pixel) in img_out.enumerate_pixels_mut() {
        let bx = (px % 8) as usize;
        let by = (py % 8) as usize;
        let id = (px / 8 + (py / 8) * BLOCKS_WIDTH) as usize;
        let y = (blocks[id].0[bx][by] as f64 / DST_SCALE[bx][by]).clamp(0.0, 255.0) as u8;
        *pixel = Rgb([y, y, y]);
    }

    img_out.save("data/01_res.png")?;
    Ok(())
}
