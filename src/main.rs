const WIDTH: u32 = 1024;
const HEIGHT: u32 = 576;

fn main() {
    let img = ratnest::generate_image(WIDTH, HEIGHT);

    img.save("out.png").unwrap();
}
