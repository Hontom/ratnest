use std::{
    f32::consts::PI,
    fmt::{Display, Write},
};

use image::{Rgb, RgbImage};
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, Uniform, weighted::WeightedIndex},
};
use rand_chacha::ChaCha20Rng;

#[derive(Debug, Clone)]
enum Expr {
    /// Indexes into provided weights array.
    Variable(usize),
    Literal(f32),
    Operation(Box<Operation>),
}

impl Expr {
    fn random<R: Rng + ?Sized>(rng: &mut R, depth: u32) -> Expr {
        // TODO: allow weights as arguments

        let variable_weights = [1.0, 1.0];
        let variable_dist = WeightedIndex::new(variable_weights).unwrap();

        // Chances that we choose to do an operation and keep recursing
        // Decreases as depth increases to lower chances of stack overflow
        let op_expr_weight = (1.0 / (depth as f32 + 1.0)) * 25.0;

        let expr_weights = [1.0, 1.0, op_expr_weight];
        let expr_dist = WeightedIndex::new(expr_weights).unwrap();

        // TODO: Disabled inverse trig functions cause of domain restrictions
        let op_weights = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0,
        ];
        let op_dist = WeightedIndex::new(op_weights).unwrap();

        // TODO: Increase literal range and bias towards lower numbers
        match expr_dist.sample(rng) {
            0 => Expr::Variable(variable_dist.sample(rng)),
            1 => Expr::Literal(Uniform::new(0.0, 255.0).unwrap().sample(rng)),
            2 => Expr::Operation(Box::new(match op_dist.sample(rng) {
                0 => Operation::Add(Expr::random(rng, depth + 1), Expr::random(rng, depth + 1)),
                1 => Operation::Sub(Expr::random(rng, depth + 1), Expr::random(rng, depth + 1)),
                2 => Operation::Mul(Expr::random(rng, depth + 1), Expr::random(rng, depth + 1)),
                3 => Operation::Div(Expr::random(rng, depth + 1), Expr::random(rng, depth + 1)),
                4 => Operation::Mod(Expr::random(rng, depth + 1), Expr::random(rng, depth + 1)),
                5 => Operation::Pow(Expr::random(rng, depth + 1), Expr::random(rng, depth + 1)),
                6 => Operation::Log(Expr::random(rng, depth + 1)),
                7 => Operation::Sin(Expr::random(rng, depth + 1)),
                8 => Operation::Cos(Expr::random(rng, depth + 1)),
                9 => Operation::Tan(Expr::random(rng, depth + 1)),
                10 => Operation::ASin(Expr::random(rng, depth + 1)),
                11 => Operation::ACos(Expr::random(rng, depth + 1)),
                12 => Operation::ATan(Expr::random(rng, depth + 1)),
                13 => Operation::Radians(Expr::random(rng, depth + 1)),
                14 => Operation::Degrees(Expr::random(rng, depth + 1)),
                15 => Operation::Sqrt(Expr::random(rng, depth + 1)),
                16 => Operation::Fract(Expr::random(rng, depth + 1)),
                17 => Operation::Reciprocal(Expr::random(rng, depth + 1)),
                _ => unreachable!(),
            })),
            _ => unreachable!(),
        }
    }

    fn to_glsl((r, g, b): &(Expr, Expr, Expr)) -> Result<String, std::fmt::Error> {
        let mut o = String::new();
        let var_names = [String::from("x"), String::from("y")];

        write!(&mut o, "precision highp float;")?;
        write!(
            &mut o,
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {{"
        )?;

        write!(&mut o, "vec2 uv = fragCoord / iResolution.xy;")?;
        write!(&mut o, "float x = uv.x;")?;
        write!(&mut o, "float y = uv.y;")?;
        
        write!(&mut o, "float r = {};", r.to_formattable(&var_names))?;
        write!(&mut o, "float g = {};", g.to_formattable(&var_names))?;
        write!(&mut o, "float b = {};", b.to_formattable(&var_names))?;

        write!(&mut o, "fragColor = vec4(r, g, b, 1.0);")?;

        write!(&mut o, "}}")?;

        Ok(o)
    }

    fn to_string(&self, var_names: &[String]) -> String {
        match self {
            Expr::Variable(idx) => var_names[*idx].clone(),
            Expr::Literal(x) if x.fract() == 0.0 => x.to_string() + ".0",
            Expr::Literal(x) => x.to_string(),
            Expr::Operation(op) => match op.as_ref() {
                Operation::Add(left, right) => {
                    '('.to_string()
                        + &left.to_string(var_names)
                        + "+"
                        + &right.to_string(var_names)
                        + ")"
                }
                Operation::Sub(left, right) => {
                    '('.to_string()
                        + &left.to_string(var_names)
                        + "-"
                        + &right.to_string(var_names)
                        + ")"
                }
                Operation::Mul(left, right) => {
                    '('.to_string()
                        + &left.to_string(var_names)
                        + "*"
                        + &right.to_string(var_names)
                        + ")"
                }
                Operation::Div(left, right) => {
                    '('.to_string()
                        + &left.to_string(var_names)
                        + "/"
                        + &right.to_string(var_names)
                        + ")"
                }
                Operation::Mod(left, right) => {
                    "mod(".to_string()
                        + &left.to_string(var_names)
                        + ","
                        + &right.to_string(var_names)
                        + ")"
                }
                Operation::Pow(left, right) => {
                    "pow(".to_string()
                        + &left.to_string(var_names)
                        + ","
                        + &right.to_string(var_names)
                        + ")"
                }
                Operation::Log(left) => "log(".to_string() + &left.to_string(var_names) + ")",
                Operation::Sin(left) => "sin(".to_string() + &left.to_string(var_names) + ")",
                Operation::Cos(left) => "cos(".to_string() + &left.to_string(var_names) + ")",
                Operation::Tan(left) => "tan(".to_string() + &left.to_string(var_names) + ")",
                Operation::ASin(left) => "asin(".to_string() + &left.to_string(var_names) + ")",
                Operation::ACos(left) => "acos(".to_string() + &left.to_string(var_names) + ")",
                Operation::ATan(left) => "atan(".to_string() + &left.to_string(var_names) + ")",
                Operation::Radians(left) => {
                    "radians(".to_string() + &left.to_string(var_names) + ")"
                }
                Operation::Degrees(left) => {
                    "degrees(".to_string() + &left.to_string(var_names) + ")"
                }
                Operation::Sqrt(left) => "sqrt(".to_string() + &left.to_string(var_names) + ")",
                Operation::Fract(left) => "fract(".to_string() + &left.to_string(var_names) + ")",
                Operation::Reciprocal(left) => {
                    "(1.0/".to_string() + &left.to_string(var_names) + ")"
                }
            },
        }
    }

    fn eval(&self, vars: &[f32]) -> f32 {
        match self {
            Expr::Variable(idx) => vars[*idx],
            Expr::Literal(num) => *num,
            Expr::Operation(op) => match op.as_ref() {
                Operation::Add(left, right) => left.eval(vars) + right.eval(vars),
                Operation::Sub(left, right) => left.eval(vars) - right.eval(vars),
                Operation::Mul(left, right) => left.eval(vars) * right.eval(vars),
                Operation::Div(left, right) => left.eval(vars) / right.eval(vars),
                Operation::Mod(left, right) => left.eval(vars) % right.eval(vars),
                Operation::Pow(left, right) => left.eval(vars).powf(right.eval(vars)),
                Operation::Log(left) => left.eval(vars).ln(),
                Operation::Sin(left) => left.eval(vars).sin(),
                Operation::Cos(left) => left.eval(vars).cos(),
                Operation::Tan(left) => left.eval(vars).tan(),
                Operation::ASin(left) => left.eval(vars).asin(),
                Operation::ACos(left) => left.eval(vars).acos(),
                Operation::ATan(left) => left.eval(vars).atan(),
                Operation::Radians(left) => PI * left.eval(vars) / 180.0,
                Operation::Degrees(left) => 180.0 * left.eval(vars) / PI,
                Operation::Sqrt(left) => left.eval(vars).sqrt(),
                Operation::Fract(left) => left.eval(vars) - left.eval(vars).floor(),
                Operation::Reciprocal(left) => 1.0 / left.eval(vars),
            },
        }
    }

    fn to_formattable<'a>(&'a self, var_names: &'a [String]) -> ExprFmt<'a> {
        ExprFmt {
            expr: self,
            var_names,
        }
    }

    fn range(&self, (x_min, x_max): (f32, f32), (y_min, y_max): (f32, f32)) -> (f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for x in 0..=100 {
            let x = lerp(x_min, x_max, x as f32 / 100.0);

            for y in 0..=100 {
                let y = lerp(y_min, y_max, y as f32 / 100.0);

                let res = self.eval(&[x, y]).clamp(-10000.0, 10000.0);
                let res = if res.is_normal() { res } else { 0.0 };

                min = res.min(min);
                max = res.max(max);
            }
        }

        (min, max)
    }

    // TODO: Fix with more ways than linear interpolation
    fn fix(mut self, (expr_min, expr_max): (f32, f32), min: f32, max: f32) -> Expr {
        let dist = min - expr_min;
        let diff = expr_max - expr_min;
        let factor = max / diff;
        let factor = if factor.is_normal() { factor } else { 1.0 };

        if dist != 0.0 {
            self = Expr::Operation(Box::new(Operation::Add(self, Expr::Literal(dist))));
        }

        if factor != 1.0 {
            self = Expr::Operation(Box::new(Operation::Mul(self, Expr::Literal(factor))));
        }

        self
    }
}

fn lerp(start: f32, end: f32, amount: f32) -> f32 {
    start + (end - start) * amount
}

struct ExprFmt<'a> {
    expr: &'a Expr,
    var_names: &'a [String],
}

impl<'a> Display for ExprFmt<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expr.to_string(self.var_names))?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
enum Operation {
    Add(Expr, Expr),
    Sub(Expr, Expr),
    Mul(Expr, Expr),
    Div(Expr, Expr),
    Mod(Expr, Expr),
    Pow(Expr, Expr),
    Log(Expr),
    Sin(Expr),
    Cos(Expr),
    Tan(Expr),
    ASin(Expr),
    ACos(Expr),
    ATan(Expr),
    Radians(Expr),
    Degrees(Expr),
    Sqrt(Expr),
    Fract(Expr),
    Reciprocal(Expr),
}

/// Will generate an `RgbImage` using ratnest's expression generator
pub fn generate_image(width: u32, height: u32) -> RgbImage {
    let mut img = RgbImage::new(width, height);

    let mut rng = ChaCha20Rng::from_os_rng();
    // let seed = rng.get_seed();

    let expr: (Expr, Expr, Expr) = (
        Expr::random(&mut rng, 0),
        Expr::random(&mut rng, 0),
        Expr::random(&mut rng, 0),
    );

    let expr = (
        {
            let range = expr.0.range((0.0, 1.0), (0.0, 1.0));
            expr.0.fix(range, 0.0, 1.0)
        },
        {
            let range = expr.1.range((0.0, 1.0), (0.0, 1.0));
            expr.1.fix(range, 0.0, 1.0)
        },
        {
            let range = expr.2.range((0.0, 1.0), (0.0, 1.0));
            expr.2.fix(range, 0.0, 1.0)
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

pub fn generate_image_with_seed() -> RgbImage {
    todo!()
}

pub fn generate_glsl() -> String {
    todo!()
}

pub fn generate_glsl_with_seed() -> String {
    todo!()
}
