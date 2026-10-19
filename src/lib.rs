use std::f32::consts::PI;

use rand::{
    Rng, SeedableRng,
    distr::{Distribution, Uniform, weighted::WeightedIndex},
};
use rand_chacha::ChaCha20Rng;

#[derive(Debug, Clone)]
enum ExprInner {
    /// Zero-indexed reference to an input variable (e.g., x = `Variable(0)` and y = `Variable(1)`).
    Variable(usize),
    /// A constant value.
    Literal(f32),
    /// An operation on one or more expressions (arity dependent on function/operator).
    Operation(Box<Operation>),
}

/// A numeric expression that accepts a set number of input parameters.
pub struct Expr {
    num_variables: usize,
    inner_expr: ExprInner,
}

/// Options used to generate a random expression. [`RandExprCfg::default()`] returns the default options.
#[derive(Clone, Copy, Debug)]
pub struct RandExprCfg<'a> {
    /// The number of input (independent) variables (parameters) the expression is going to take.
    ///
    /// DEFAULT: `1`
    pub num_variables: usize,
    /// The chance that generating an expression will go "deeper" (choose an operation instead of a constant or variable) is `1 / (depth + 1) * weight_go_deeper`, with `depth` starting at 0.
    ///
    /// DEFAULT: `25.0`
    pub weight_go_deeper: f32,
    /// The relative chances that any variable is chosen. Slice must have exactly as many elements as there are input variables in the expression. The `None` variant is interpreted as uniform chance across all variables.
    ///
    /// DEFAULT: `None`
    ///
    /// # Example
    /// Let's say you have three input variables: `x`, `y`, and `z`, which respectively get mapped to variable indices 0, 1, and 2. If you want `x` to have a 50% chance of being selected and `y` and `z` each to have a 25% chance, you could use:
    /// ```
    /// RandomExprOptions {
    ///     // ...
    ///     variable_weights: Some(&[2.0, 1.0, 1.0]),
    ///     // ...
    /// }
    /// ```
    ///
    /// You could have equivalently chosen `[0.5, 0.25, 0.25]` or `[12.2, 6.1, 6.1]`.
    pub variable_weights: Option<&'a [f32]>,
    // TODO: more! see `todo.md`
}

impl<'a> RandExprCfg<'a> {
    // Returns the default configuration for random expression generation.
    pub fn default() -> Self {
        Self {
            num_variables: 1,
            weight_go_deeper: 25.0,
            variable_weights: None,
        }
    }

    /// Finalizes the config into a wrapper that ratnest functions accept.
    ///
    /// This is just here to ensure the data entered into the config is valid. `RandExprCfgFinal` has no public fields.
    pub fn finalize(self) -> Result<RandExprCfgFinal<'a>, RandExprCfgFinalizeErr> {
        if let Some(weights) = self.variable_weights
            && weights.len() != self.num_variables
        {
            return Err(RandExprCfgFinalizeErr::VariableCountMismatch);
        }

        if !self.weight_go_deeper.is_finite() || self.weight_go_deeper < 0.0 {
            return Err(RandExprCfgFinalizeErr::InvalidDepthWeight);
        }

        Ok(RandExprCfgFinal(self))
    }
}

/// The finalized version of a [`RandExprCfg`], the existence of which guarantees its validity (no public fields!) hence it is the config type that ratnest functions accept.
#[derive(Clone, Copy)]
pub struct RandExprCfgFinal<'a>(RandExprCfg<'a>);

/// The problems that could have been encountered during the conversion from the [`RandExprCfg`] to the [`RandExprCfgFinal`].
#[derive(Debug, Clone, Copy)]
pub enum RandExprCfgFinalizeErr {
    /// There is a mismatch between the length of the per-variable lists and the number of variables.
    VariableCountMismatch,
    /// The depth was invalid (Negative, Infinity, NaN).
    InvalidDepthWeight,
}

impl Expr {
    /// Generate a random expression, using a given random number generator.
    pub fn generate_with_rng<R: Rng + ?Sized>(
        rng: &mut R,
        cfg_final: RandExprCfgFinal<'_>,
    ) -> Self {
        Self {
            num_variables: cfg_final.0.num_variables,
            inner_expr: ExprInner::generate_with_rng(rng, 0, cfg_final),
        }
    }

    /// Generate a random expression.
    ///
    /// This function uses [`rand_chacha::ChaCha20Rng`] for a random number generator internally. If you wish to use your own, see [`random_with_rng()`](Expr::random_with_rng).
    pub fn generate(config: RandExprCfgFinal<'_>) -> Self {
        let mut rng = ChaCha20Rng::from_os_rng();

        Self::generate_with_rng(&mut rng, config)
    }

    /// Evaluates the expression, passing in the variables provided.
    pub fn eval(&self, vars: &[f32]) -> f32 {
        // ugly?
        assert_eq!(vars.len(), self.num_variables);

        self.inner_expr.eval(vars)
    }

    /// Returns the range of the expression given each variable's domain, using `num_samples` random samples.
    // TODO: also pass rng here
    pub fn range(&self, domain: &[(f32, f32)], num_samples: u32) -> (f32, f32) {
        return self.inner_expr.range(domain, num_samples);
    }

    pub fn linear_transform_to_range(
        self,
        original_range: (f32, f32),
        desired_range: (f32, f32),
    ) -> Self {
        Self {
            num_variables: self.num_variables,
            inner_expr: self
                .inner_expr
                .linear_transform_to_range(original_range, desired_range),
        }
    }

    /// Returns a formatted string of the expression. Variables are represented with `x0`, `x1`, etc.
    pub fn to_string(&self) -> String {
        self.inner_expr.to_string()
    }
}

impl ExprInner {
    /// Generates the inner expression.
    fn generate_with_rng<R: Rng + ?Sized>(
        rng: &mut R,
        depth: u32,
        final_config: RandExprCfgFinal<'_>,
    ) -> ExprInner {
        // TODO: allow weights as arguments

        let RandExprCfgFinal(config) = final_config;

        let variable_dist = if let Some(ws) = config.variable_weights {
            WeightedIndex::new(ws).unwrap()
        } else {
            WeightedIndex::new(std::iter::repeat_n(1.0, config.num_variables)).unwrap()
        };

        // Chances that we choose to do an operation and keep recursing
        // Decreases as depth increases to lower chances of stack overflow
        let op_expr_weight = (1.0 / (depth as f32 + 1.0)) * config.weight_go_deeper;

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
            0 => ExprInner::Variable(variable_dist.sample(rng)),
            1 => ExprInner::Literal(Uniform::new(0.0, 255.0).unwrap().sample(rng)),
            2 => ExprInner::Operation(Box::new(match op_dist.sample(rng) {
                0 => Operation::Add(
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                ),
                1 => Operation::Sub(
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                ),
                2 => Operation::Mul(
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                ),
                3 => Operation::Div(
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                ),
                4 => Operation::Mod(
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                ),
                5 => Operation::Pow(
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                    ExprInner::generate_with_rng(rng, depth + 1, final_config),
                ),
                6 => Operation::Log(ExprInner::generate_with_rng(rng, depth + 1, final_config)),
                7 => Operation::Sin(ExprInner::generate_with_rng(rng, depth + 1, final_config)),
                8 => Operation::Cos(ExprInner::generate_with_rng(rng, depth + 1, final_config)),
                9 => Operation::Tan(ExprInner::generate_with_rng(rng, depth + 1, final_config)),
                10 => Operation::ASin(ExprInner::generate_with_rng(rng, depth + 1, final_config)),
                11 => Operation::ACos(ExprInner::generate_with_rng(rng, depth + 1, final_config)),
                12 => Operation::ATan(ExprInner::generate_with_rng(rng, depth + 1, final_config)),
                13 => {
                    Operation::Radians(ExprInner::generate_with_rng(rng, depth + 1, final_config))
                }
                14 => {
                    Operation::Degrees(ExprInner::generate_with_rng(rng, depth + 1, final_config))
                }
                15 => Operation::Sqrt(ExprInner::generate_with_rng(rng, depth + 1, final_config)),
                16 => Operation::Fract(ExprInner::generate_with_rng(rng, depth + 1, final_config)),
                17 => Operation::Reciprocal(ExprInner::generate_with_rng(
                    rng,
                    depth + 1,
                    final_config,
                )),
                _ => unreachable!(),
            })),
            _ => unreachable!(),
        }
    }

    // fn to_glsl((r, g, b): &(ExprInner, ExprInner, ExprInner)) -> Result<String, std::fmt::Error> {
    //     let mut o = String::new();
    //     let var_names = [String::from("x"), String::from("y")];

    //     write!(&mut o, "precision highp float;")?;
    //     write!(
    //         &mut o,
    //         "void mainImage(out vec4 fragColor, in vec2 fragCoord) {{"
    //     )?;

    //     write!(&mut o, "vec2 uv = fragCoord / iResolution.xy;")?;
    //     write!(&mut o, "float x = uv.x;")?;
    //     write!(&mut o, "float y = uv.y;")?;

    //     write!(&mut o, "float r = {};", r.to_formattable(&var_names))?;
    //     write!(&mut o, "float g = {};", g.to_formattable(&var_names))?;
    //     write!(&mut o, "float b = {};", b.to_formattable(&var_names))?;

    //     write!(&mut o, "fragColor = vec4(r, g, b, 1.0);")?;

    //     write!(&mut o, "}}")?;

    //     Ok(o)
    // }

    fn to_string(&self) -> String {
        match self {
            ExprInner::Variable(idx) => format!("x{}", idx),
            ExprInner::Literal(x) if x.fract() == 0.0 => x.to_string() + ".0",
            ExprInner::Literal(x) => x.to_string(),
            ExprInner::Operation(op) => match op.as_ref() {
                Operation::Add(left, right) => {
                    '('.to_string() + &left.to_string() + "+" + &right.to_string() + ")"
                }
                Operation::Sub(left, right) => {
                    '('.to_string() + &left.to_string() + "-" + &right.to_string() + ")"
                }
                Operation::Mul(left, right) => {
                    '('.to_string() + &left.to_string() + "*" + &right.to_string() + ")"
                }
                Operation::Div(left, right) => {
                    '('.to_string() + &left.to_string() + "/" + &right.to_string() + ")"
                }
                Operation::Mod(left, right) => {
                    "mod(".to_string() + &left.to_string() + "," + &right.to_string() + ")"
                }
                Operation::Pow(left, right) => {
                    "pow(".to_string() + &left.to_string() + "," + &right.to_string() + ")"
                }
                Operation::Log(left) => "log(".to_string() + &left.to_string() + ")",
                Operation::Sin(left) => "sin(".to_string() + &left.to_string() + ")",
                Operation::Cos(left) => "cos(".to_string() + &left.to_string() + ")",
                Operation::Tan(left) => "tan(".to_string() + &left.to_string() + ")",
                Operation::ASin(left) => "asin(".to_string() + &left.to_string() + ")",
                Operation::ACos(left) => "acos(".to_string() + &left.to_string() + ")",
                Operation::ATan(left) => "atan(".to_string() + &left.to_string() + ")",
                Operation::Radians(left) => "radians(".to_string() + &left.to_string() + ")",
                Operation::Degrees(left) => "degrees(".to_string() + &left.to_string() + ")",
                Operation::Sqrt(left) => "sqrt(".to_string() + &left.to_string() + ")",
                Operation::Fract(left) => "fract(".to_string() + &left.to_string() + ")",
                Operation::Reciprocal(left) => "(1.0/".to_string() + &left.to_string() + ")",
            },
        }
    }

    fn eval(&self, vars: &[f32]) -> f32 {
        match self {
            &ExprInner::Variable(idx) => vars[idx],
            &ExprInner::Literal(num) => num,
            ExprInner::Operation(op) => match op.as_ref() {
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

    /// Accepts the domain of the parameters and returns the range of the expression over that domain, established with `num_samples` samples randomly distributed.
    // TODO: pass rng
    fn range(&self, domain: &[(f32, f32)], num_samples: u32) -> (f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        let mut rng = rand::rng();

        let mut test_vars = vec![0.0; domain.len()].into_boxed_slice();
        for _ in 0..num_samples {
            for (var_domain, var) in domain.iter().zip(test_vars.iter_mut()) {
                *var = rng.random_range(var_domain.0..var_domain.1);
            }

            let val = self.eval(&test_vars);

            if val.is_normal() {
                if val > max {
                    max = val;
                }
                if val < min {
                    min = val;
                }
            }
        }

        (min, max)
    }

    // TODO: Fix with more ways than linear interpolation
    fn linear_transform_to_range(
        mut self,
        (original_min, original_max): (f32, f32),
        (desired_min, desired_max): (f32, f32),
    ) -> ExprInner {
        let dist = desired_min - original_min;
        let diff = original_max - original_min;
        let factor = desired_max / diff;
        let factor = if factor.is_normal() { factor } else { 1.0 };

        if dist != 0.0 {
            self = ExprInner::Operation(Box::new(Operation::Add(self, ExprInner::Literal(dist))));
        }

        if factor != 1.0 {
            self = ExprInner::Operation(Box::new(Operation::Mul(self, ExprInner::Literal(factor))));
        }

        self
    }
}

#[derive(Debug, Clone)]
enum Operation {
    Add(ExprInner, ExprInner),
    Sub(ExprInner, ExprInner),
    Mul(ExprInner, ExprInner),
    Div(ExprInner, ExprInner),
    Mod(ExprInner, ExprInner),
    Pow(ExprInner, ExprInner),
    Log(ExprInner),
    Sin(ExprInner),
    Cos(ExprInner),
    Tan(ExprInner),
    ASin(ExprInner),
    ACos(ExprInner),
    ATan(ExprInner),
    Radians(ExprInner),
    Degrees(ExprInner),
    Sqrt(ExprInner),
    Fract(ExprInner),
    Reciprocal(ExprInner),
}
