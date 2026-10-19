use image::{RgbImage, Rgb};
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use ratnest::{Expr, RandExprCfg};

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 576;

/// Will generate an `RgbImage` using ratnest's expression generator
pub fn generate_image(width: u32, height: u32) -> RgbImage {
    let mut img = RgbImage::new(width, height);

    let mut rng = ChaCha20Rng::from_os_rng();
    // let seed = rng.get_seed();

    let options = RandExprCfg{
        num_variables: 2,
        ..RandExprCfg::default()
    }.finalize().unwrap();

    let expr: (Expr, Expr, Expr) = (
        Expr::generate_with_rng(&mut rng, options),
        Expr::generate_with_rng(&mut rng, options),
        Expr::generate_with_rng(&mut rng, options),
    );

    let expr = (
        {
            let range = expr.0.range(&[(0.0, 1.0), (0.0, 1.0)], 8192);
            expr.0.linear_transform_to_range(range, (0.0, 255.0))
        },
        {
            let range = expr.1.range(&[(0.0, 1.0), (0.0, 1.0)], 8192);
            expr.1.linear_transform_to_range(range, (0.0, 255.0))
        },
        {
            let range = expr.2.range(&[(0.0, 1.0), (0.0, 1.0)], 8192);
            expr.2.linear_transform_to_range(range, (0.0, 255.0))
        },
    );

    // let glsl = Expr::to_glsl(&expr).unwrap();

    // let variable_names = &[String::from("x"), String::from("y")];

    // dbg!(&expr);
    // println!("red: {}", expr.0.to_formattable(variable_names));
    // println!("green: {}", expr.1.to_formattable(variable_names));
    // println!("blue: {}", expr.2.to_formattable(variable_names));
    // println!("{}", glsl);

    for (x, y, Rgb(color)) in img.enumerate_pixels_mut() {
        // 0.0 - 1.0 value coords
        let x = x as f32 / width as f32;
        let y = 1.0 - y as f32 / height as f32;

        *color = [
            expr.0.eval(&[x, y]) as u8,
            expr.1.eval(&[x, y]) as u8,
            expr.2.eval(&[x, y]) as u8,
        ];
    }

    img
}

fn main() {
    let img = generate_image(WIDTH, HEIGHT);

    img.save("out.png").unwrap();
}
